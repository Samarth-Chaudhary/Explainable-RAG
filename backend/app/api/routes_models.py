"""Model catalog and schema endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from backend.app.api.deps import get_current_user_role, get_orchestrator
from backend.app.services.orchestrator import RunAuditOrchestrator

router = APIRouter(prefix="/api/models", tags=["Models & Schemas"])


@router.get("")
def list_models(
    orchestrator: RunAuditOrchestrator = Depends(get_orchestrator),
    user_role: str = Depends(get_current_user_role),
) -> JSONResponse:
    """List registered model artifacts, active schemas, calibration, and threshold metadata."""
    manifest = orchestrator._manifest
    schema = orchestrator._schema

    if not manifest or not schema:
        return JSONResponse(
            content={
                "active_model": None,
                "models": [],
                "message": "No model manifest or feature schema currently loaded.",
            }
        )

    model_entry = {
        "name": manifest.name,
        "version": manifest.version,
        "framework": getattr(manifest, "framework", "xgboost"),
        "artifact_sha256": manifest.artifact_sha256,
        "calibration_status": manifest.calibration_status,
        "features_count": len(schema.features),
        "features": [f.model_dump() for f in schema.features],
        "default_threshold": {
            "value": orchestrator.settings.threshold_value,
            "source": orchestrator.settings.threshold_source,
            "owner": orchestrator.settings.threshold_owner,
        },
    }

    return JSONResponse(
        content={
            "active_model": manifest.name,
            "models": [model_entry],
        }
    )
