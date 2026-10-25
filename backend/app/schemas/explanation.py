"""SHAP explanation contracts and mathematical verification schemas."""

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


OutputSpace = Literal["log_odds", "probability"]
AdditivityStatus = Literal["valid", "degraded", "invalid"]
Direction = Literal["risk_increasing", "risk_reducing", "neutral"]


class FeatureContributionSummary(BaseModel):
    """Local attribution for a single feature."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Programmatic feature identifier")
    label: str = Field(description="Human-readable business display label")
    value: float = Field(description="Observed applicant feature value")
    unit: str = Field(description="Measurement unit")
    shap_value: float = Field(description="Local SHAP attribution contribution in declared output space")
    direction: Direction = Field(description="'risk_increasing' (> 0), 'risk_reducing' (< 0), or 'neutral'")
    rank: int = Field(ge=1, description="Absolute attribution importance rank (1 = largest impact)")


FeatureContribution = FeatureContributionSummary


class ExplanationResult(BaseModel):
    """Typed result of SHAP explanation and additivity audit (P1.0 handoff)."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default="1.0.0", description="Contract schema version")
    output_space: OutputSpace = Field(
        description="Declared coordinate space: 'log_odds' (margin space) or 'probability'"
    )
    base_value: float = Field(description="Expected model output E[f(x)] over reference background")
    shap_additivity_residual: float = Field(
        description="Absolute difference between (base_value + sum(shap)) and model target output"
    )
    tolerance: float = Field(description="Configured maximum acceptable numerical residual")
    additivity_status: AdditivityStatus = Field(
        description="Status: 'valid' passes residual check; 'degraded' or 'invalid' blocks downstream narrative"
    )
    features: list[FeatureContributionSummary] = Field(
        description="Complete ranked list of all feature contributions (all features, not top-3)"
    )
    top_risk_increasing: list[FeatureContributionSummary] = Field(
        description="Primary features pushing prediction toward higher credit risk"
    )
    top_risk_reducing: list[FeatureContributionSummary] = Field(
        description="Primary features mitigating credit risk toward approval"
    )
    explainer_config: dict[str, Any] = Field(
        description="Explainer metadata: model_output, feature_perturbation, background_id, shap_version"
    )
    methodology_notes: str = Field(
        description="Scientific caveats: SHAP measures attribution relative to background, not real-world causality"
    )
