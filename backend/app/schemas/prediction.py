"""Prediction contracts and error schemas."""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class ModelMetadataSummary(BaseModel):
    """Essential model identification and calibration summary."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Model name")
    version: str = Field(description="Model version")
    artifact_sha256: str = Field(description="Cryptographic SHA-256 hash of model artifact")
    calibration_status: str = Field(description="Calibration status: 'none', 'platt', 'isotonic'")


class ThresholdConfig(BaseModel):
    """Decision threshold configuration and ownership."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float = Field(ge=0.0, le=1.0, description="Decision threshold cutoff value")
    source: str = Field(description="Policy or committee document establishing threshold")
    owner: str = Field(description="Accountable executive or committee owner")


class FieldError(BaseModel):
    """Detailed error description for an individual input field."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str = Field(description="Feature name")
    value: Any = Field(description="Received value that failed validation")
    issue: str = Field(description="Validation error description")
    constraint: str = Field(description="Specific rule or bound violated")


class InputValidationError(Exception):
    """Raised when applicant input fails feature schema validation."""

    def __init__(self, errors: list[FieldError], message: str | None = None):
        self.errors = errors
        self.message = message or f"Input validation failed for {len(errors)} field(s)."
        super().__init__(self.message)


class PredictionResult(BaseModel):
    """Typed result of deterministic model inference (P1.0 handoff)."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default="1.0.0", description="Contract schema version")
    model: ModelMetadataSummary = Field(description="Model lineage and integrity summary")
    raw_output: float = Field(description="Raw model score (log-odds margin for logistic XGBoost)")
    default_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Predicted default probability [0.0, 1.0]",
    )
    calibrated_probability: float | None = Field(
        default=None,
        description="Calibrated probability if model supports calibration; otherwise None",
    )
    threshold: ThresholdConfig = Field(description="Configured decision threshold with named owner")
    decision_label: str = Field(
        description="Operational decision: e.g. 'REJECT_HIGH_RISK' or 'APPROVE_LOW_RISK'"
    )
    output_notes: str = Field(description="Explanatory notes on calibration, link function, and threshold rule")
    inference_timestamp: datetime = Field(description="UTC timestamp of inference execution")
    feature_order_hash: str = Field(description="SHA-256 hash of canonical evaluated feature order")
    features_evaluated: dict[str, float] = Field(
        description="Final evaluated feature vector (primitive and derived values)"
    )
