"""Application configuration module using pydantic-settings.

Enforces strict demo vs configured boundary.
"""

from pathlib import Path
from typing import Literal
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """XAI-RAG system settings and environment configuration."""

    model_config = SettingsConfigDict(
        env_prefix="XAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    # Runtime mode
    mode: Literal["demo", "configured"] = Field(
        default="demo",
        description="Operating mode: 'demo' uses synthetic fixtures; 'configured' requires production assets",
    )

    # Model & Feature Assets
    model_path: str = Field(
        default="data/demo/model.json",
        description="Path to serialized model artifact",
    )
    model_manifest_path: str = Field(
        default="data/demo/model_manifest.json",
        description="Path to model manifest JSON metadata",
    )
    feature_schema_path: str = Field(
        default="data/demo/feature_schema.json",
        description="Path to feature schema JSON definition",
    )
    background_data_path: str = Field(
        default="data/demo/explainer_background.json",
        description="Path to background reference data for SHAP TreeExplainer",
    )
    ontology_path: str = Field(
        default="data/ontology/credit_risk_ontology_v1.json",
        description="Path to domain ontology definition",
    )
    document_storage_path: str = Field(
        default="data/policies",
        description="Directory for ingested policy documents",
    )

    # Operational Decision Threshold
    # Default 0.35 is an illustrative demo value, never a regulatory fact
    threshold_value: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Decision threshold for classification (probability of default)",
    )
    threshold_source: str = Field(
        default="Credit Risk Committee Policy Manual v2.4 (Synthetic Fixture)",
        description="Institutional governance policy authorizing this threshold",
    )
    threshold_owner: str = Field(
        default="Head of Credit Underwriting & Risk Governance",
        description="Named institutional officer accountable for the decision threshold",
    )

    # SHAP Explainer Settings
    shap_output_space: Literal["log_odds", "probability"] = Field(
        default="log_odds",
        description="Coordinate space for SHAP attribution calculations",
    )
    shap_tolerance: float = Field(
        default=1e-4,
        description="Numerical tolerance for SHAP additivity residual checks",
    )
    shap_top_k: int = Field(
        default=5,
        ge=1,
        description="Top-k number of features to highlight as primary drivers",
    )
    shap_contribution_cutoff: float = Field(
        default=0.05,
        ge=0.0,
        description="Minimum absolute attribution cutoff |phi_j| for top-driver inclusion",
    )

    # DiCE Counterfactual Settings
    dice_desired_class: int = Field(
        default=0,
        description="Target prediction class for counterfactuals (0: non-default / approval)",
    )
    dice_total_cfs: int = Field(
        default=3,
        ge=1,
        description="Desired number of diverse counterfactual candidates",
    )
    dice_target_probability_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Upper bound target for predicted default probability",
    )
    dice_method: Literal["genetic", "random", "kdtree"] = Field(
        default="genetic",
        description="DiCE search optimization algorithm",
    )

    # Persistence & Indexes (Parts 2 & 3)
    database_url: str = Field(
        default="sqlite:///./xai_rag.db",
        description="Database connection URL",
    )
    vector_index_path: str = Field(
        default="data/indexes/faiss_index.bin",
        description="Path to FAISS dense index file",
    )
    bm25_index_path: str = Field(
        default="data/indexes/bm25_index.pkl",
        description="Path to BM25 lexical index file",
    )
    max_upload_size_bytes: int = Field(
        default=10 * 1024 * 1024,
        description="Maximum permitted document upload size (10MB default)",
    )

    # LLM Provider Configuration (Part 2)
    llm_provider: str = Field(
        default="template_local",
        description="LLM provider name: 'template_local', 'openai', 'gemini', etc.",
    )
    llm_api_key: str | None = Field(
        default=None,
        description="API key for configured LLM provider",
    )

    # Security & Auth (Part 3)
    auth_enabled: bool = Field(
        default=False,
        description="Whether API authentication is enforced",
    )
    api_secret_key: str | None = Field(
        default=None,
        description="Secret key for JWT validation when auth is enabled",
    )

    @property
    def is_synthetic_inputs(self) -> bool:
        """Indicates whether synthetic test inputs are active."""
        return self.mode == "demo"

    @property
    def is_synthetic_model(self) -> bool:
        """Indicates whether a synthetic demo model is active."""
        return self.mode == "demo"

    @property
    def is_synthetic_policy(self) -> bool:
        """Indicates whether synthetic policy fixtures are active."""
        return self.mode == "demo"

    @property
    def is_synthetic_llm(self) -> bool:
        """Indicates whether synthetic local template LLM is active."""
        return self.mode == "demo"

    @model_validator(mode="after")
    def validate_mode_integrity(self) -> "Settings":
        """Strict validation of configured production vs demo mode boundaries."""
        if self.mode == "configured":
            # In configured mode, production assets and governance ownership must exist
            errors: list[str] = []

            if not self.threshold_owner or not self.threshold_owner.strip():
                errors.append("Configured mode requires an explicit, non-empty 'threshold_owner'.")

            if not self.threshold_source or not self.threshold_source.strip():
                errors.append("Configured mode requires an explicit, non-empty 'threshold_source'.")

            model_file = Path(self.model_path)
            if not model_file.is_file():
                errors.append(
                    f"Configured mode requires a valid model file at '{self.model_path}', but none was found."
                )

            manifest_file = Path(self.model_manifest_path)
            if not manifest_file.is_file():
                errors.append(
                    f"Configured mode requires a valid manifest file at '{self.model_manifest_path}', but none was found."
                )

            schema_file = Path(self.feature_schema_path)
            if not schema_file.is_file():
                errors.append(
                    f"Configured mode requires a valid feature schema file at '{self.feature_schema_path}', but none was found."
                )

            if errors:
                raise ValueError("Configured Mode Validation Failed:\n  - " + "\n  - ".join(errors))

        return self


# Global singleton factory
def get_settings() -> Settings:
    """Retrieve application settings."""
    return Settings()
