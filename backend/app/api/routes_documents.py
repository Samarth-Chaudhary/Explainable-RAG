"""Document registration, upload, reindexing, and status management endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_user_role, get_db, get_orchestrator, require_role
from backend.app.models.audit_models import DocumentModel
from backend.app.repositories.audit_repository import AuditRepository
from backend.app.schemas.handoff import IngestionResult
from backend.app.schemas.source_registry import SourceRegistryRecord
from backend.app.services.document_ingestion import DocumentIngestionError, DocumentIngestionService
from backend.app.services.orchestrator import RunAuditOrchestrator
from backend.app.services.retrieval import HybridRetrievalService

router = APIRouter(prefix="/api/documents", tags=["Policy Documents"])


class DocumentStatusUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str  # "active" or "superseded"


@router.get("")
def list_documents(
    db: Session = Depends(get_db),
    user_role: str = Depends(get_current_user_role),
) -> dict[str, Any]:
    """List registered policy documents, ingestion states, and active/superseded statuses."""
    repo = AuditRepository(db)
    docs = repo.list_documents()

    doc_list = []
    for d in docs:
        doc_list.append({
            "source_id": d.source_id,
            "title": d.title,
            "issuer": d.issuer,
            "jurisdiction": d.jurisdiction,
            "source_family": d.source_family,
            "version": d.version,
            "effective_date": d.effective_date,
            "status": "superseded" if d.superseded else "active",
            "sha256_hash": d.sha256_hash,
            "total_pages": d.total_pages,
            "total_chunks": len(d.chunks),
            "parser_status": d.parser_status,
            "access_classification": d.access_classification,
            "ingestion_timestamp": d.created_at.isoformat() if d.created_at else None,
        })

    return {
        "total": len(doc_list),
        "documents": doc_list,
        "disclaimer": "All demonstration documents must carry the 'SYNTHETIC DEMO POLICY - NOT LAW' prefix.",
    }


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_and_ingest_document(
    file: UploadFile = File(...),
    source_id: str = Form(...),
    title: str = Form(...),
    issuer: str = Form(default="Internal Credit Policy Committee"),
    jurisdiction: str = Form(default="IN"),
    source_family: str = Form(default="underwriting_rules"),
    version: str = Form(default="v1.0"),
    effective_date: str = Form(default="2026-02-01"),
    access_classification: str = Form(default="internal_confidential"),
    db: Session = Depends(get_db),
    orchestrator: RunAuditOrchestrator = Depends(get_orchestrator),
    user_role: str = Depends(require_role(["admin", "compliance_reviewer"])),
) -> dict[str, Any]:
    """Upload and ingest an authorized PDF document with strict magic-byte and content validation."""
    # Read file content safely
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds maximum allowed upload limit (50 MB).",
        )

    # Magic byte check
    if not contents.startswith(b"%PDF"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format: file does not begin with valid PDF magic bytes '%PDF'.",
        )

    # Save to managed storage path with sanitized name
    clean_filename = f"{source_id}_{uuid.uuid4().hex[:8]}.pdf".replace("/", "_").replace("\\", "_")
    storage_dir = Path("data/policies/uploaded")
    storage_dir.mkdir(parents=True, exist_ok=True)
    target_path = storage_dir / clean_filename

    with open(target_path, "wb") as f:
        f.write(contents)

    sha256 = hashlib.sha256(contents).hexdigest()
    now = datetime.now(timezone.utc)

    metadata = SourceRegistryRecord(
        source_id=source_id,
        title=title,
        issuer=issuer,
        jurisdiction=jurisdiction,
        source_family=source_family,
        publication_date=now.strftime("%Y-%m-%d"),
        effective_date=effective_date,
        version_or_update_date=version,
        status="active",
        official_url_or_owner="compliance-admin@internal.demo",
        sha256_hash=sha256,
        ingestion_timestamp=now,
        access_classification="internal_confidential" if access_classification not in ("public", "restricted") else access_classification,
        parser_status="pending",
    )

    ingestion_service = orchestrator._ingestion_service or DocumentIngestionService()
    try:
        ingest_res = ingestion_service.ingest_pdf(target_path, metadata)
    except DocumentIngestionError as die:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(die))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ingestion failed: {exc}")

    # Persist in DB
    repo = AuditRepository(db)
    repo.save_document(
        ingest_res.document,
        parser_status=ingest_res.parser_status,
        access_classification=metadata.access_classification,
        superseded=False,
    )
    repo.save_chunks(ingest_res.chunks)

    # Reindex with newly added chunks
    all_chunks = repo.get_all_chunks()
    if orchestrator._retrieval_service and all_chunks:
        build_res = orchestrator._retrieval_service.build_indexes(all_chunks)
        repo.save_index_build(build_res, {"reindex_trigger": "upload", "doc_id": source_id})

    return {
        "status": "ingested",
        "source_id": source_id,
        "title": title,
        "parser_status": ingest_res.parser_status,
        "chunks_extracted": len(ingest_res.chunks),
        "total_pages": ingest_res.document.total_pages,
        "sha256_hash": sha256,
    }


@router.post("/{document_id}/reindex")
def reindex_documents(
    document_id: str,
    db: Session = Depends(get_db),
    orchestrator: RunAuditOrchestrator = Depends(get_orchestrator),
    user_role: str = Depends(require_role(["admin", "compliance_reviewer"])),
) -> dict[str, Any]:
    """Rebuild hybrid FAISS + BM25 search indexes and persist versioned index build metadata."""
    repo = AuditRepository(db)
    all_chunks = repo.get_all_chunks()

    if not all_chunks and orchestrator._retrieval_service and orchestrator._retrieval_service.chunks:
        repo.save_chunks(orchestrator._retrieval_service.chunks)
        all_chunks = repo.get_all_chunks()

    if not all_chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document chunks found in database to index.",
        )

    retrieval_service = orchestrator._retrieval_service or HybridRetrievalService()
    build_res = retrieval_service.build_indexes(all_chunks)

    repo.save_index_build(build_res, {"triggered_by": user_role, "target_doc": document_id})

    return {
        "status": "reindexed",
        "index_version": build_res.index_version,
        "document_count": build_res.document_count,
        "chunk_count": build_res.chunk_count,
        "embedding_model": build_res.embedding_model,
        "timestamp": build_res.build_timestamp.isoformat(),
    }


@router.post("/{document_id}/status")
def update_document_status(
    document_id: str,
    update: DocumentStatusUpdate,
    db: Session = Depends(get_db),
    user_role: str = Depends(require_role(["admin", "compliance_reviewer"])),
) -> dict[str, Any]:
    """Update document lifecycle status between active and superseded."""
    repo = AuditRepository(db)
    doc = repo.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    if update.status not in ("active", "superseded"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Status must be either 'active' or 'superseded'.",
        )

    doc.superseded = (update.status == "superseded")
    db.commit()

    return {
        "source_id": document_id,
        "new_status": update.status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
