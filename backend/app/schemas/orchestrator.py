"""Schemas for Run Audit Orchestrator, stage execution tracking, and background jobs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

from backend.app.schemas.audit_report import AuditReport

StageStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]
JobState = Literal["queued", "running", "completed", "failed"]


class StageExecutionRecord(BaseModel):
    """Execution status and timing for an individual orchestrator pipeline stage."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_name: str = Field(description="Pipeline stage identifier: e.g. 'validate_input', 'predict'")
    status: StageStatus = Field(default="pending", description="Current execution state of this stage")
    started_at: datetime | None = Field(default=None, description="Stage start timestamp")
    completed_at: datetime | None = Field(default=None, description="Stage completion timestamp")
    duration_ms: float | None = Field(default=None, description="Execution duration in milliseconds")
    error_message: str | None = Field(default=None, description="Error details if stage failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Stage-specific diagnostic metrics")


class AuditRunRequest(BaseModel):
    """Payload to trigger a complete Run Audit execution."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    applicant_data: dict[str, Any] = Field(description="Applicant feature input data dictionary")
    applicant_ref: str | None = Field(default=None, description="Optional pseudonymized applicant reference")
    idempotency_key: str | None = Field(default=None, description="Unique client request ID to avoid duplicate runs")
    user_role: str = Field(default="analyst", description="Role of requesting user: analyst, reviewer, admin")
    run_async: bool = Field(default=False, description="Whether to execute as a background job")


class AuditJobStatusResponse(BaseModel):
    """Complete status report for an audit job or completed execution."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str = Field(description="Unique job identifier")
    audit_id: str | None = Field(default=None, description="Associated audit ID once initialized")
    idempotency_key: str | None = Field(default=None, description="Client idempotency key")
    status: JobState = Field(description="Overall job state")
    progress_percentage: int = Field(ge=0, le=100, description="Truthful completion percentage 0-100")
    current_stage: str = Field(description="Currently executing or final stage name")
    stages: list[StageExecutionRecord] = Field(default_factory=list, description="Per-stage execution records")
    report: AuditReport | None = Field(default=None, description="Final AuditReport if completed")
    error: str | None = Field(default=None, description="Error message if run failed")
    created_at: datetime = Field(description="Job creation timestamp")
    completed_at: datetime | None = Field(default=None, description="Job completion timestamp")


class ReviewerDispositionRequest(BaseModel):
    """Reviewer disposition action recording."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    disposition: Literal["approve", "reject_for_rework", "escalate"] = Field(
        description="Reviewer action decision"
    )
    notes: str = Field(description="Reviewer rationale and observations")
    reviewer_id: str = Field(description="Identifier of reviewing officer")
