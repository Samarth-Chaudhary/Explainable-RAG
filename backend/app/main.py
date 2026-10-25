"""FastAPI application entrypoint for XAI-RAG Platform."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import uuid
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes_applicants import router as applicants_router
from backend.app.api.routes_audits import router as audits_router
from backend.app.api.routes_auth import router as auth_router
from backend.app.api.routes_documents import router as documents_router
from backend.app.api.routes_evaluation import router as evaluation_router
from backend.app.api.routes_health import app_health_router
from backend.app.api.routes_jobs import router as jobs_router
from backend.app.api.routes_models import router as models_router
from backend.app.config import get_settings
from backend.app.repositories.database import init_db
from backend.app.schemas.handoff import ErrorEnvelope

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("xai_rag.api")

settings = get_settings()

app = FastAPI(
    title="XAI-RAG Platform API",
    version="1.0.0",
    description=(
        "Full-stack, explainable credit-risk decision-auditing and remediation platform combining "
        "tabular ML, local explanations (TreeSHAP), constrained counterfactuals (DiCE), "
        "ontology-guided hybrid retrieval (FAISS+BM25), citation-locked structured reports, and "
        "a tamper-evident hash-chained audit ledger."
    ),
    openapi_tags=[
        {"name": "Health & Status", "description": "Liveness, readiness, and runtime mode indicators"},
        {"name": "Audits", "description": "Full decision audit orchestration, reports, traces, and PDF export"},
        {"name": "Models & Schemas", "description": "Registered model artifacts, schemas, and threshold configs"},
        {"name": "Applicants", "description": "Applicant search and inspection with PII minimization"},
        {"name": "Policy Documents", "description": "Document ingestion, indexing, and status lifecycle"},
        {"name": "Background Jobs", "description": "Asynchronous pipeline stage polling"},
        {"name": "Evaluation & Benchmarks", "description": "Quantitative evaluation metrics and benchmarks"},
    ],
)

# ------------------------------------------------------------------------------
# Security & CORS Middleware
# ------------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Open for local dev & dashboard testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "Content-Disposition"],
)


@app.middleware("http")
async def request_id_and_logging_middleware(request: Request, call_next):
    """Attach unique X-Request-ID and structured audit logging to every request."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ------------------------------------------------------------------------------
# Consistent ErrorEnvelope Exception Handlers
# ------------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Format HTTP errors into consistent ErrorEnvelope schema."""
    envelope = ErrorEnvelope(
        error_code=f"HTTP_{exc.status_code}",
        message=str(exc.detail),
        details=[{"status_code": exc.status_code}],
        timestamp=datetime.now(timezone.utc),
    )
    return JSONResponse(status_code=exc.status_code, content=envelope.model_dump(mode="json"))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Format Pydantic / FastAPI request validation errors into ErrorEnvelope."""
    errors_formatted = []
    for err in exc.errors():
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        errors_formatted.append({"field": loc, "issue": err.get("msg"), "type": err.get("type")})

    envelope = ErrorEnvelope(
        error_code="VALIDATION_ERROR",
        message="Request payload failed schema validation.",
        details=errors_formatted,
        timestamp=datetime.now(timezone.utc),
    )
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=envelope.model_dump(mode="json"))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Safely catch unhandled internal server exceptions without leaking internal secrets."""
    logger.exception("Unhandled server exception: %s", exc)
    envelope = ErrorEnvelope(
        error_code="INTERNAL_SERVER_ERROR",
        message="An internal processing error occurred. Check server logs with request ID.",
        details=[{"request_id": getattr(request.state, "request_id", "unknown")}],
        timestamp=datetime.now(timezone.utc),
    )
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=envelope.model_dump(mode="json"))


# ------------------------------------------------------------------------------
# Startup and Shutdown Lifecycle Hooks
# ------------------------------------------------------------------------------
@app.on_event("startup")
def on_startup() -> None:
    """Initialize SQLite/Postgres database tables and verify core assets."""
    logger.info("Initializing XAI-RAG database schemas and storage directories...")
    init_db()


# ------------------------------------------------------------------------------
# Include API Routers
# ------------------------------------------------------------------------------
app.include_router(app_health_router)
app.include_router(auth_router)
app.include_router(models_router)
app.include_router(applicants_router)
app.include_router(audits_router)
app.include_router(documents_router)
app.include_router(jobs_router)
app.include_router(evaluation_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
