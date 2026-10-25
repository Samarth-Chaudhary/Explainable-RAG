"""Health and readiness check endpoints."""

from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db, get_orchestrator
from backend.app.config import get_settings
from backend.app.services.orchestrator import RunAuditOrchestrator

app_health_router = APIRouter(tags=["Health & Status"])


@app_health_router.get("/health")
def health_check(
    db: Session = Depends(get_db),
    orchestrator: RunAuditOrchestrator = Depends(get_orchestrator),
) -> JSONResponse:
    """System health check, dependency diagnostics, and runtime mode status."""
    settings = get_settings()

    # Check model and artifacts
    model_loaded = orchestrator._model_loader is not None and orchestrator._model_loader.model is not None
    schema_loaded = orchestrator._schema is not None
    ontology_loaded = orchestrator._ontology_service is not None
    indexes_loaded = (
        orchestrator._retrieval_service is not None
        and orchestrator._retrieval_service.index_version != "uninitialized"
    )

    # Check database liveness
    db_connected = False
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False

    is_ready = model_loaded and schema_loaded and db_connected
    overall_status = "healthy" if is_ready else "degraded"

    return JSONResponse(
        content={
            "status": overall_status,
            "mode": settings.mode,
            "ready": is_ready,
            "dependencies": {
                "model_artifact": "ready" if model_loaded else "missing",
                "feature_schema": "ready" if schema_loaded else "missing",
                "domain_ontology": "ready" if ontology_loaded else "missing",
                "hybrid_indexes": "ready" if indexes_loaded else "empty_or_bootstrap",
                "audit_ledger_db": "connected" if db_connected else "disconnected",
            },
            "synthetic_flags": {
                "is_synthetic_inputs": settings.is_synthetic_inputs,
                "is_synthetic_model": settings.is_synthetic_model,
                "is_synthetic_policy": settings.is_synthetic_policy,
                "is_synthetic_llm": settings.is_synthetic_llm,
            },
            "threshold": {
                "value": settings.threshold_value,
                "source": settings.threshold_source,
                "owner": settings.threshold_owner,
            },
        }
    )
