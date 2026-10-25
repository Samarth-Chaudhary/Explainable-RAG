"""Counterfactual contracts, candidate evaluations, and feasibility metrics."""

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


FeasibilityStatus = Literal["feasible", "infeasible", "partially_feasible"]


class FeatureChange(BaseModel):
    """Specific proposed adjustment to an actionable feature."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: str = Field(description="Name of actionable primitive feature")
    current: float = Field(description="Original observed applicant value")
    proposed: float = Field(description="Proposed counterfactual value")
    delta: float = Field(description="Proposed change: proposed - current")


class DerivedRecalculation(BaseModel):
    """Automatic recalculation of a derived ratio resulting from primitive changes."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: str = Field(description="Name of derived feature")
    formula: str = Field(description="Mathematical formula used for recomputation")
    previous_value: float = Field(description="Original derived value")
    new_value: float = Field(description="Recomputed derived value after primitive modifications")


class CounterfactualCandidate(BaseModel):
    """An individual candidate remediation scenario."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Unique candidate scenario identifier (e.g. 'CF-1')")
    changes: list[FeatureChange] = Field(description="List of proposed feature changes on actionable levers")
    derived_recalculations: list[DerivedRecalculation] = Field(
        description="Recalculated derived ratios (e.g. FOIR, LTV)"
    )
    predicted_probability: float = Field(
        description="Exact re-scored default probability from production inference pipeline"
    )
    target_met: bool = Field(description="Whether re-scored probability satisfies target threshold")
    constraints_passed: list[str] = Field(description="Validated business, range, and immutability constraints")
    constraints_failed: list[str] = Field(
        default_factory=list,
        description="Any business or domain constraints that failed validation",
    )
    feasibility_status: FeasibilityStatus = Field(
        description="'feasible' (all constraints met), 'infeasible', or 'partially_feasible'"
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Mandatory advisory disclosures (e.g. 'Candidate scenario for human review; not a guarantee')",
    )
    unmodelled_dependencies: list[str] = Field(
        default_factory=list,
        description="Real-world dependencies not captured in tabular model (e.g. collateral valuation verification)",
    )


class CounterfactualMetrics(BaseModel):
    """Summary metrics of counterfactual optimization and feasibility."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    validity: float = Field(description="Proportion of evaluated candidates meeting target threshold")
    actionability: float = Field(
        description="Proportion of evaluated candidates meeting target AND all feasibility constraints"
    )
    generation_failure_rate: float = Field(description="1.0 - validity rate")
    time_to_generate_ms: float = Field(description="Search and validation duration in milliseconds")
    number_of_candidates: int = Field(description="Total valid candidates returned")
    proximity: float = Field(description="Continuous L1 distance normalized by feature ranges")
    diversity: float = Field(description="Mean pairwise distance across returned candidate vectors")
    ood_flags: int = Field(description="Count of feature values flagged as out-of-distribution")
    target_miss_magnitude: float = Field(description="Mean probability gap above target for non-satisfying profiles")
    denominator: str = Field(description="Explicit explanation of calculation denominator (e.g. '3 candidates evaluated')")
    dice_succeeded: bool = Field(default=False, description="Whether DiCE optimization succeeded")
    fallback_used: bool = Field(default=False, description="Whether constrained domain fallback was used")
    sparsity: float = Field(default=0.0, description="Mean number of actionable features altered")


class CounterfactualResult(BaseModel):
    """Typed result of constrained counterfactual optimization (P1.0 handoff)."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default="1.0.0", description="Contract schema version")
    request_config: dict[str, Any] = Field(description="Configuration used for generation")
    candidates: list[CounterfactualCandidate] = Field(
        description="Candidate profiles satisfying target and constraints"
    )
    no_feasible_found: bool = Field(
        description="True if no candidate satisfied both the target threshold and all constraints"
    )
    no_feasible_message: str | None = Field(
        default=None,
        description="Exact string: 'No feasible counterfactual found under configured constraints'",
    )
    closest_non_satisfying: list[CounterfactualCandidate] = Field(
        default_factory=list,
        description="Nearest non-qualifying alternatives (never labelled as approval paths)",
    )
    metrics: CounterfactualMetrics = Field(description="Performance and quality metrics")
