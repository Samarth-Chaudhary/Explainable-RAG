"""Authoritative AuditReport schema shared by every module."""

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


AuditMode = Literal["demo", "configured"]
AuditStatus = Literal["complete", "partial", "failed"]
OutputSpace = Literal["log_odds", "probability"]
FeatureDirection = Literal["risk_increasing", "risk_reducing", "neutral"]
FeasibilityStatus = Literal["feasible", "infeasible", "partially_feasible"]
EvidenceStatus = Literal["verified", "unverified", "failed"]


class ReportModelSummary(BaseModel):
    """Model lineage summary in AuditReport."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Model name")
    version: str = Field(description="Model version")
    artifact_sha256: str = Field(description="Cryptographic SHA-256 hash of model artifact")
    calibration_status: str = Field(description="Calibration status disclosure")


class ReportPredictionSummary(BaseModel):
    """Prediction summary in AuditReport."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_output: float = Field(description="Raw model score / margin")
    default_probability: float = Field(description="Probability of default")
    threshold: float = Field(description="Operational decision threshold")
    label: str = Field(description="Operational classification label")
    output_notes: str = Field(description="Notes on calibration and score derivation")


class ReportFeatureContribution(BaseModel):
    """Individual feature attribution entry in AuditReport."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Feature name")
    label: str = Field(description="Display label")
    value: float = Field(description="Feature value")
    unit: str = Field(description="Unit of measurement")
    shap_value: float = Field(description="Attribution value in declared output space")
    direction: FeatureDirection = Field(description="'risk_increasing', 'risk_reducing', or 'neutral'")


class ReportExplanationSummary(BaseModel):
    """Explanation summary in AuditReport."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    output_space: OutputSpace = Field(description="'log_odds' or 'probability'")
    base_value: float = Field(description="Background base value E[f(x)]")
    shap_additivity_residual: float = Field(description="Residual of additivity equation")
    features: list[ReportFeatureContribution] = Field(description="All feature attributions")


class ReportCounterfactualCandidate(BaseModel):
    """Counterfactual scenario summary in AuditReport."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Candidate identifier")
    changes: list[dict[str, Any]] = Field(description="Feature changes [{feature, current, proposed, delta}]")
    derived_recalculations: list[dict[str, Any]] = Field(
        description="Derived ratio recalculations [{feature, formula, previous_value, new_value}]"
    )
    predicted_probability: float = Field(description="Directly re-scored model probability")
    target_met: bool = Field(description="Whether candidate meets target risk threshold")
    constraints_passed: list[str] = Field(description="List of passed constraints")
    feasibility_status: FeasibilityStatus = Field(description="Feasibility classification")
    caveats: list[str] = Field(description="Required caveats and disclosures")


class ReportSearchTraceSummary(BaseModel):
    """Retrieval trace summary in AuditReport."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    ontology_version: str = Field(description="Ontology version used for query translation")
    features_used: list[str] = Field(description="Features used to construct search query")
    dense_query: str = Field(description="Dense semantic query text")
    bm25_terms: list[str] = Field(description="Lexical search terms")
    metadata_filters: dict[str, Any] = Field(description="Applied metadata filters")
    retrieved_chunk_ids: list[str] = Field(description="List of chunk IDs retrieved")
    index_version: str = Field(description="Vector and lexical index version")


class ReportEvidenceItem(BaseModel):
    """Individual evidence item in AuditReport."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(description="Source document identifier")
    title: str = Field(description="Title of document")
    issuer: str = Field(description="Issuing authority")
    jurisdiction: str = Field(description="Jurisdiction")
    source_type: str = Field(description="Category or source family")
    version: str = Field(description="Document version")
    effective_date: str = Field(description="Effective date")
    page: int = Field(description="Document page number")
    section: str = Field(description="Section heading or clause identifier")
    quote: str = Field(description="Extracted quote text")
    chunk_id: str = Field(description="Specific chunk ID in index")
    url: str = Field(description="Official URL or document reference")
    verification_status: EvidenceStatus = Field(description="'verified', 'unverified', or 'failed'")


class ReportDossierSummary(BaseModel):
    """Structured synthesis dossier in AuditReport."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    adverse_action_explanation: str = Field(description="Synthesis of primary adverse action drivers")
    candidate_review_paths: list[str] = Field(description="Actionable restructuring paths for credit officer review")
    regulatory_justification: str = Field(description="Policy and regulatory grounding summary")
    evidence_gaps: list[str] = Field(description="Any identified gaps in available policy evidence")
    reviewer_notes: str = Field(description="Notes for the human compliance reviewer")


class ReportVerifierSummary(BaseModel):
    """Quote matching and citation verification metrics in AuditReport."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    quote_checks_total: int = Field(description="Total quotes verified against source text")
    quote_checks_passed: int = Field(description="Quotes confirmed verbatim")
    quote_checks_failed: int = Field(description="Quotes failing exact match check")
    faithfulness_score: float = Field(description="Faithfulness score (passed / total)")
    normalized_match_policy: str = Field(description="Policy used for text normalization")


class AuditReport(BaseModel):
    """The authoritative cross-module audit report contract."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    audit_id: str = Field(description="Unique audit execution identifier")
    applicant_ref: str = Field(description="Pseudonymous / display applicant reference (no unnecessary PII)")
    generated_at: datetime = Field(description="Audit completion timestamp")
    mode: AuditMode = Field(description="'demo' or 'configured'")
    status: AuditStatus = Field(description="'complete', 'partial', or 'failed'")
    model: ReportModelSummary = Field(description="Model lineage metadata")
    prediction: ReportPredictionSummary = Field(description="Prediction results")
    explanation: ReportExplanationSummary = Field(description="TreeSHAP local explanation")
    counterfactuals: list[ReportCounterfactualCandidate] = Field(description="Constrained counterfactual candidates")
    search_trace: ReportSearchTraceSummary = Field(description="Ontology-to-retrieval trace")
    evidence: list[ReportEvidenceItem] = Field(description="Retrieved and verified policy evidence")
    dossier: ReportDossierSummary = Field(description="Structured report synthesis")
    verifier: ReportVerifierSummary = Field(description="Citation verification audit summary")
    human_review_required: bool = Field(
        default=True,
        description="Mandatory advisory flag: system is decision support and requires human judgment",
    )
