"""Audit execution, retrieval, report inspection, and PDF export endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_user_role, get_db, get_orchestrator, require_role
from backend.app.models.audit_models import AuditModel
from backend.app.repositories.audit_repository import AuditRepository
from backend.app.schemas.audit_report import AuditReport
from backend.app.schemas.orchestrator import (
    AuditJobStatusResponse,
    AuditRunRequest,
    ReviewerDispositionRequest,
)
from backend.app.services.audit_ledger import AuditLedgerService
from backend.app.services.export_pdf import PDFExportService
from backend.app.services.orchestrator import RunAuditOrchestrator

router = APIRouter(prefix="/api/audits", tags=["Audits"])


@router.post("", response_model=AuditJobStatusResponse, status_code=status.HTTP_201_CREATED)
def create_audit(
    request: AuditRunRequest,
    db: Session = Depends(get_db),
    orchestrator: RunAuditOrchestrator = Depends(get_orchestrator),
    user_role: str = Depends(get_current_user_role),
) -> AuditJobStatusResponse:
    """Trigger a Run Audit pipeline execution (synchronous or background job)."""
    # Override role with authenticated role
    req_with_role = AuditRunRequest(
        applicant_data=request.applicant_data,
        applicant_ref=request.applicant_ref,
        idempotency_key=request.idempotency_key,
        user_role=user_role,
        run_async=request.run_async,
    )
    job_status = orchestrator.execute_audit(req_with_role, db)
    return job_status


@router.get("")
def list_audits(
    db: Session = Depends(get_db),
    user_role: str = Depends(get_current_user_role),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """List historical audit records with metadata, models, index versions, and statuses."""
    repo = AuditRepository(db)
    records = repo.list_audits(limit=limit, offset=offset)

    audits_summary = []
    for r in records:
        pred_dict = {}
        try:
            pred_dict = json.loads(r.prediction_json) if r.prediction_json else {}
        except Exception:
            pass

        audits_summary.append({
            "audit_id": r.audit_id,
            "applicant_ref": r.applicant_ref,
            "generated_at": r.generated_at.isoformat() if r.generated_at else None,
            "mode": r.mode,
            "status": r.status,
            "model_name": r.model_name,
            "model_version": r.model_version,
            "default_probability": pred_dict.get("default_probability"),
            "decision_label": pred_dict.get("label"),
            "human_review_required": r.human_review_required,
        })

    return {
        "total": len(audits_summary),
        "limit": limit,
        "offset": offset,
        "audits": audits_summary,
    }


@router.get("/{audit_id}")
def get_audit_summary(
    audit_id: str,
    db: Session = Depends(get_db),
    user_role: str = Depends(get_current_user_role),
) -> dict[str, Any]:
    """Read audit details, tamper-evident lineage events, and chain integrity verification."""
    repo = AuditRepository(db)
    ledger = AuditLedgerService(repo)

    # Reopen audit with metadata integrity and access checks
    try:
        report, chain_valid, warnings = ledger.reopen_audit(
            audit_id=audit_id,
            requesting_role=user_role,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if "not found" in str(exc).lower() else status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        )

    events = repo.get_events_for_audit(audit_id)
    lineage_events = [
        {
            "sequence_num": ev.sequence_num,
            "event_type": ev.event_type,
            "timestamp": ev.timestamp.isoformat() if ev.timestamp else None,
            "actor": ev.actor,
            "event_hash": ev.event_hash,
            "previous_event_hash": ev.previous_event_hash,
            "payload_summary": json.loads(ev.payload_json) if ev.payload_json else {},
        }
        for ev in events
    ]

    return {
        "audit_id": report.audit_id,
        "applicant_ref": report.applicant_ref,
        "generated_at": report.generated_at.isoformat(),
        "mode": report.mode,
        "status": report.status,
        "model": report.model.model_dump(),
        "prediction": report.prediction.model_dump(),
        "chain_integrity_valid": chain_valid,
        "chain_warnings": warnings,
        "events_count": len(lineage_events),
        "events": lineage_events,
        "human_review_required": report.human_review_required,
    }


@router.get("/{audit_id}/report", response_model=AuditReport)
def get_audit_report(
    audit_id: str,
    db: Session = Depends(get_db),
    user_role: str = Depends(get_current_user_role),
) -> AuditReport:
    """Return strict dossier JSON adhering to versioned AuditReport contract."""
    repo = AuditRepository(db)
    ledger = AuditLedgerService(repo)

    try:
        report, _, _ = ledger.reopen_audit(audit_id=audit_id, requesting_role=user_role)
        return report
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if "not found" in str(exc).lower() else status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        )


@router.get("/{audit_id}/trace")
def get_audit_retrieval_trace(
    audit_id: str,
    db: Session = Depends(get_db),
    user_role: str = Depends(get_current_user_role),
) -> dict[str, Any]:
    """Retrieve raw search trace separately from narrative so auditors can inspect retrieval logic."""
    repo = AuditRepository(db)
    trace_rec = repo.get_retrieval_trace(audit_id)
    if not trace_rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No retrieval trace found for audit ID '{audit_id}'.",
        )

    return {
        "trace_id": trace_rec.trace_id,
        "audit_id": trace_rec.audit_id,
        "dense_query": trace_rec.dense_query,
        "bm25_terms": json.loads(trace_rec.bm25_terms_json) if trace_rec.bm25_terms_json else [],
        "metadata_filters": json.loads(trace_rec.metadata_filters_json) if trace_rec.metadata_filters_json else {},
        "retrieved_chunk_ids": json.loads(trace_rec.retrieved_chunk_ids_json) if trace_rec.retrieved_chunk_ids_json else [],
        "dense_scores": json.loads(trace_rec.dense_scores_json) if trace_rec.dense_scores_json else {},
        "bm25_scores": json.loads(trace_rec.bm25_scores_json) if trace_rec.bm25_scores_json else {},
        "fused_scores": json.loads(trace_rec.fused_scores_json) if trace_rec.fused_scores_json else {},
        "execution_time_ms": trace_rec.execution_time_ms,
    }


@router.get("/{audit_id}/export.pdf")
def export_audit_pdf(
    audit_id: str,
    db: Session = Depends(get_db),
    user_role: str = Depends(get_current_user_role),
) -> FileResponse:
    """Generate and download a traceable, printable PDF compliance audit dossier."""
    repo = AuditRepository(db)
    ledger = AuditLedgerService(repo)

    try:
        report, _, _ = ledger.reopen_audit(audit_id=audit_id, requesting_role=user_role)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if "not found" in str(exc).lower() else status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        )

    exporter = PDFExportService(output_dir="data/exports")
    pdf_path = exporter.export_report_pdf(report)

    if not pdf_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate PDF dossier export file.",
        )

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"dossier_{audit_id}.pdf",
    )


@router.post("/{audit_id}/disposition")
def record_disposition(
    audit_id: str,
    request: ReviewerDispositionRequest,
    db: Session = Depends(get_db),
    user_role: str = Depends(require_role(["reviewer", "compliance_reviewer", "admin"])),
) -> dict[str, Any]:
    """Record a human reviewer disposition (approve, reject_for_rework, escalate) in the audit ledger."""
    repo = AuditRepository(db)
    ledger = AuditLedgerService(repo)

    # Check audit exists
    report = repo.get_audit(audit_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audit '{audit_id}' not found.",
        )

    action_id = f"act-{uuid.uuid4().hex[:10]}"
    action_rec = repo.record_reviewer_action(
        action_id=action_id,
        audit_id=audit_id,
        reviewer_id=request.reviewer_id,
        action_type=request.disposition,
        notes=request.notes,
    )

    # Append immutable event to hash chain
    ledger.append_event(
        audit_id=audit_id,
        event_type="REVIEWER_DISPOSITION_RECORDED",
        actor=request.reviewer_id,
        payload={
            "action_id": action_id,
            "disposition": request.disposition,
            "notes": request.notes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    return {
        "status": "recorded",
        "action_id": action_id,
        "audit_id": audit_id,
        "disposition": request.disposition,
        "reviewer_id": request.reviewer_id,
    }
