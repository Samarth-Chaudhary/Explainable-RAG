"""Unified schema exports for XAI-RAG."""

from backend.app.schemas.audit_report import (
    AuditMode,
    AuditReport,
    AuditStatus,
    EvidenceStatus,
    FeasibilityStatus,
    FeatureDirection,
    OutputSpace,
    ReportCounterfactualCandidate,
    ReportDossierSummary,
    ReportEvidenceItem,
    ReportExplanationSummary,
    ReportFeatureContribution,
    ReportModelSummary,
    ReportPredictionSummary,
    ReportSearchTraceSummary,
    ReportVerifierSummary,
)
from backend.app.schemas.counterfactual import (
    CounterfactualCandidate,
    CounterfactualMetrics,
    CounterfactualResult,
    DerivedRecalculation,
    FeatureChange,
)
from backend.app.schemas.explanation import (
    AdditivityStatus,
    ExplanationResult,
    FeatureContributionSummary,
)
from backend.app.schemas.feature_schema import (
    DataType,
    FeatureDefinition,
    FeatureSchema,
    MutabilityClass,
    NullHandlingRule,
    NumericRange,
)
from backend.app.schemas.handoff import (
    AuditEvent,
    Chunk,
    DraftReport,
    ErrorEnvelope,
    GeneratorInputPayload,
    HybridQueryPayload,
    IndexBuildResult,
    IngestionResult,
    JobStatus,
    RetrievalResult,
    RetrievalTrace,
    RetrievedEvidence,
    SourceDocument,
    TranslationResult,
    VerifierResult,
)
from backend.app.schemas.model_manifest import CalibrationStatus, ModelManifest
from backend.app.schemas.ontology import FeatureType, OntologyFile, OntologyMutability, OntologyRecord
from backend.app.schemas.orchestrator import (
    AuditJobStatusResponse,
    AuditRunRequest,
    JobState,
    ReviewerDispositionRequest,
    StageExecutionRecord,
    StageStatus,
)
from backend.app.schemas.prediction import (
    FieldError,
    InputValidationError,
    ModelMetadataSummary,
    PredictionResult,
    ThresholdConfig,
)
from backend.app.schemas.source_registry import (
    AccessClassification,
    ParserStatus,
    SourceRegistry,
    SourceRegistryRecord,
    SourceStatus,
)

__all__ = [
    # Feature Schema
    "FeatureSchema",
    "FeatureDefinition",
    "NumericRange",
    "MutabilityClass",
    "DataType",
    "NullHandlingRule",
    # Model Manifest
    "ModelManifest",
    "CalibrationStatus",
    # Ontology
    "OntologyRecord",
    "OntologyFile",
    "FeatureType",
    "OntologyMutability",
    # Source Registry
    "SourceRegistryRecord",
    "SourceRegistry",
    "SourceStatus",
    "ParserStatus",
    "AccessClassification",
    # Prediction
    "PredictionResult",
    "ThresholdConfig",
    "ModelMetadataSummary",
    "FieldError",
    "InputValidationError",
    # Explanation
    "ExplanationResult",
    "FeatureContributionSummary",
    "OutputSpace",
    "AdditivityStatus",
    # Counterfactuals
    "CounterfactualResult",
    "CounterfactualCandidate",
    "FeatureChange",
    "DerivedRecalculation",
    "CounterfactualMetrics",
    "FeasibilityStatus",
    # Audit Report
    "AuditReport",
    "AuditMode",
    "AuditStatus",
    "ReportModelSummary",
    "ReportPredictionSummary",
    "ReportFeatureContribution",
    "ReportExplanationSummary",
    "ReportCounterfactualCandidate",
    "ReportSearchTraceSummary",
    "ReportEvidenceItem",
    "ReportDossierSummary",
    "ReportVerifierSummary",
    "FeatureDirection",
    "EvidenceStatus",
    # Handoff Objects
    "HybridQueryPayload",
    "Chunk",
    "SourceDocument",
    "RetrievalTrace",
    "RetrievedEvidence",
    "VerifierResult",
    "AuditEvent",
    "ErrorEnvelope",
    "JobStatus",
    "TranslationResult",
    "IngestionResult",
    "IndexBuildResult",
    "RetrievalResult",
    "GeneratorInputPayload",
    "DraftReport",
    # Orchestrator & Job Schemas
    "AuditRunRequest",
    "AuditJobStatusResponse",
    "StageExecutionRecord",
    "ReviewerDispositionRequest",
    "StageStatus",
    "JobState",
]
