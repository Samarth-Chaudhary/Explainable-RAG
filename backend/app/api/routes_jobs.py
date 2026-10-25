"""Asynchronous job polling and stage progress endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.api.deps import get_orchestrator
from backend.app.schemas.orchestrator import AuditJobStatusResponse
from backend.app.services.orchestrator import RunAuditOrchestrator

router = APIRouter(prefix="/api/jobs", tags=["Background Jobs"])


@router.get("/{job_id}", response_model=AuditJobStatusResponse)
def get_job_status(
    job_id: str,
    orchestrator: RunAuditOrchestrator = Depends(get_orchestrator),
) -> AuditJobStatusResponse:
    """Poll asynchronous Run Audit job progress, per-stage timing, and completed report."""
    job = orchestrator.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return job
