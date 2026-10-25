"""Model manifest schema capturing lineage, artifact hash, and calibration status."""

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


CalibrationStatus = Literal["none", "platt", "isotonic", "beta"]


class ModelManifest(BaseModel):
    """Immutable model metadata and governance manifest."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Model identifier (e.g. 'SYNTHETIC-DEMO-XGB')")
    version: str = Field(description="Model version tag (e.g. '1.0.0-synthetic')")
    artifact_path: str = Field(description="Relative path to serialized model file")
    artifact_sha256: str = Field(description="SHA-256 cryptographic digest of the model artifact")
    feature_schema_hash: str = Field(description="SHA-256 hash of the canonical feature order and schema")
    training_dataset_id: str = Field(
        default="synthetic-demo",
        description="Lineage ID of training dataset or 'synthetic demo'",
    )
    explainer_background_data_id: str = Field(
        default="synthetic-bg-100",
        description="Dataset identifier for background reference distribution",
    )
    calibration_status: CalibrationStatus = Field(
        default="none",
        description="Probability calibration method ('none', 'platt', 'isotonic')",
    )
    created_timestamp: datetime = Field(
        description="Timestamp when model was trained and packaged",
    )
    owner: str = Field(
        default="Credit Model Risk Management",
        description="Organizational unit responsible for model validation and lifecycle",
    )
    notes: str = Field(
        default="Model calibration is not implemented; raw logistic margins are mapped to probabilities.",
        description="Governance and calibration disclosure notes",
    )
