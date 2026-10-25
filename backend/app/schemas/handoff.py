from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

from backend.app.schemas.audit_report import (
    ReportCounterfactualCandidate,
    ReportDossierSummary,
    ReportEvidenceItem,
    ReportExplanationSummary,
    ReportModelSummary,
    ReportPredictionSummary,
    ReportSearchTraceSummary,
    ReportVerifierSummary,
)
from backend.app.schemas.counterfactual import CounterfactualResult
from backend.app.schemas.explanation import ExplanationResult
from backend.app.schemas.model_manifest import ModelManifest
from backend.app.schemas.prediction import PredictionResult


class HybridQueryPayload(BaseModel):
    """Payload produced by ontology translator for hybrid FAISS + BM25 search."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    dense_query: str = Field(description="Dense semantic search query text")
    bm25_terms: list[str] = Field(description="Lexical keyword search terms")
    metadata_filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata filters: e.g. jurisdiction, source_family, status",
    )
    top_k: int = Field(default=5, ge=1, description="Number of chunks to retrieve")
    source_families: list[str] = Field(default_factory=list, description="Target source categories")


class Chunk(BaseModel):
    """An individual indexed passage of an authoritative policy document."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(description="Unique chunk identifier: e.g. 'DOC-001-P4-C2'")
    source_id: str = Field(description="Parent source document ID")
    text: str = Field(description="Extracted textual content of the chunk")
    page_number: int = Field(ge=1, description="Original PDF page number")
    section_title: str = Field(description="Heading or clause label")
    token_count: int = Field(ge=0, description="Approximate token length")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom chunk metadata tags")


class SourceDocument(BaseModel):
    """Complete ingested policy document."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(description="Unique source identifier")
    title: str = Field(description="Official title")
    issuer: str = Field(description="Issuing authority")
    jurisdiction: str = Field(description="Jurisdiction")
    version: str = Field(description="Version or circular reference")
    effective_date: str = Field(description="Effective date")
    file_path: str = Field(description="Local storage path")
    sha256_hash: str = Field(description="SHA-256 integrity hash")
    total_pages: int = Field(ge=1, description="Total page count")
    chunks: list[Chunk] = Field(default_factory=list, description="Extracted chunks")


class RetrievalTrace(BaseModel):
    """Trace of retrieval execution and score fusion."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str = Field(description="Unique trace identifier")
    query_payload: HybridQueryPayload = Field(description="Input search payload")
    dense_scores: dict[str, float] = Field(description="FAISS similarity scores by chunk_id")
    bm25_scores: dict[str, float] = Field(description="BM25 relevance scores by chunk_id")
    fused_scores: dict[str, float] = Field(description="Reciprocal rank fusion scores")
    retrieved_chunks: list[Chunk] = Field(description="Final retrieved chunks in rank order")
    execution_time_ms: float = Field(description="Retrieval latency in milliseconds")


class RetrievedEvidence(BaseModel):
    """Verified evidence item bound to a specific policy claim."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(description="Unique evidence identifier")
    chunk: Chunk = Field(description="Source chunk containing text")
    extracted_quote: str = Field(description="Verbatim quote used in report")
    relevance_justification: str = Field(description="Explanation of why this clause applies")
    verification_status: Literal["verified", "unverified", "failed"] = Field(
        default="unverified",
        description="Quote verification status",
    )


class VerifierResult(BaseModel):
    """Result of exact substring and citation verification."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    total_citations: int = Field(description="Total cited quotes evaluated")
    passed_citations: int = Field(description="Quotes confirmed in source text")
    failed_citations: int = Field(description="Quotes not found in source text")
    faithfulness_score: float = Field(description="Ratio of passed to total citations")
    citation_details: list[dict[str, Any]] = Field(description="Itemized match outcomes")


class AuditEvent(BaseModel):
    """Tamper-evident audit event log record."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(description="Unique event ID")
    timestamp: datetime = Field(description="Event timestamp")
    event_type: str = Field(description="Category: e.g. 'PREDICTION', 'EXPLANATION', 'COUNTERFACTUAL'")
    actor: str = Field(description="Initiating user or system component")
    payload_hash: str = Field(description="SHA-256 hash of event payload")
    previous_event_hash: str = Field(description="Cryptographic hash of preceding event in ledger")
    event_hash: str = Field(description="SHA-256 hash of this complete block")


class ErrorEnvelope(BaseModel):
    """Standardized API and service error envelope."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    error_code: str = Field(description="Standard error identifier: e.g. 'VALIDATION_ERROR'")
    message: str = Field(description="Human-readable error explanation")
    details: list[dict[str, Any]] = Field(default_factory=list, description="Field-specific error details")
    timestamp: datetime = Field(description="Error occurrence timestamp")


class JobStatus(BaseModel):
    """Status tracking for asynchronous audit jobs."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str = Field(description="Unique job identifier")
    status: Literal["queued", "running", "completed", "failed"] = Field(description="Current status")
    progress_percentage: int = Field(ge=0, le=100, description="Completion percentage")
    current_step: str = Field(description="Active pipeline stage")
    error: str | None = Field(default=None, description="Failure reason if status is failed")
    completed_at: datetime | None = Field(default=None, description="Completion timestamp")


class TranslationResult(BaseModel):
    """Result of translating SHAP drivers and counterfactuals to hybrid queries."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_payload: HybridQueryPayload = Field(description="Primary hybrid search payload")
    features_selected: list[str] = Field(description="Features selected for search translation")
    ontology_matches: list[str] = Field(description="Features mapped successfully in ontology")
    unmapped_features: list[str] = Field(description="Features without ontology mapping")
    normalized_query: str = Field(description="Normalized dense search query text")
    lexical_terms: list[str] = Field(description="Lexical terms for BM25 ranking")
    metadata_filters: dict[str, Any] = Field(description="Metadata filter parameters")
    ontology_version: str = Field(description="Ontology version used")
    ontology_content_hash: str = Field(description="Cryptographic hash of ontology definition")


class IngestionResult(BaseModel):
    """Result of document ingestion process."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    document: SourceDocument = Field(description="Parsed source document")
    chunks: list[Chunk] = Field(description="Extracted chunks with page lineage")
    parser_status: Literal["parsed", "failed"] = Field(description="Parser execution status")
    ocr_applied: bool = Field(default=False, description="Whether OCR was required and used")
    error_message: str | None = Field(default=None, description="Failure reason if parsing failed")


class IndexBuildResult(BaseModel):
    """Result of building FAISS and BM25 search indexes."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    index_version: str = Field(description="Deterministic hash identifying index build")
    document_count: int = Field(ge=0, description="Total documents indexed")
    chunk_count: int = Field(ge=0, description="Total chunks indexed")
    embedding_model: str = Field(description="Embedding model used")
    build_timestamp: datetime = Field(description="UTC timestamp of build")


class RetrievalResult(BaseModel):
    """Result of hybrid FAISS + BM25 retrieval with fusion."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence: list[RetrievedEvidence] = Field(description="Retrieved evidence chunks")
    retrieval_trace: RetrievalTrace = Field(description="Full retrieval and fusion trace")
    no_evidence_found: bool = Field(description="True if no matching chunks found")
    missing_coverage: list[str] = Field(default_factory=list, description="Missing policy topics or features")
    applicability_warnings: list[str] = Field(default_factory=list, description="Conflict/date/jurisdiction warnings")


class GeneratorInputPayload(BaseModel):
    """Validated input payload passed to report_generator.generate()."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    audit_id: str = Field(description="Audit execution identifier")
    applicant_ref: str = Field(description="Pseudonymous applicant reference (no unnecessary PII)")
    mode: Literal["demo", "configured"] = Field(description="'demo' or 'configured'")
    model_manifest: ModelManifest = Field(description="Model metadata manifest")
    prediction: PredictionResult = Field(description="Model prediction output")
    explanation: ExplanationResult = Field(description="TreeSHAP explanation")
    counterfactuals: CounterfactualResult = Field(description="Constrained counterfactual results")
    translation: TranslationResult = Field(description="Ontology translation result")
    retrieval: RetrievalResult = Field(description="Hybrid retrieval result")
    index_version: str = Field(description="Index version used for retrieval")
    user_role: str = Field(default="credit_officer", description="Requesting user role")


class DraftReport(BaseModel):
    """Draft report generated prior to citation verification."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    audit_id: str = Field(description="Unique audit execution identifier")
    applicant_ref: str = Field(description="Pseudonymous applicant reference")
    generated_at: datetime = Field(description="Generation timestamp")
    mode: Literal["demo", "configured"] = Field(description="'demo' or 'configured'")
    status: Literal["complete", "partial", "failed"] = Field(description="Draft status")
    model: ReportModelSummary = Field(description="Model lineage metadata")
    prediction: ReportPredictionSummary = Field(description="Prediction results")
    explanation: ReportExplanationSummary = Field(description="TreeSHAP local explanation")
    counterfactuals: list[ReportCounterfactualCandidate] = Field(description="Constrained counterfactual candidates")
    search_trace: ReportSearchTraceSummary = Field(description="Ontology-to-retrieval trace")
    evidence: list[ReportEvidenceItem] = Field(description="Candidate evidence items with quotes")
    dossier: ReportDossierSummary = Field(description="Synthesized dossier sections")
    verifier: ReportVerifierSummary = Field(description="Initial verifier summary before verification")
    human_review_required: bool = Field(default=True, description="Human review required flag")
    prompt_template_hash: str = Field(description="SHA-256 hash of generator prompt template")
    llm_provider: str = Field(description="LLM provider name or 'local_deterministic'")
    llm_model: str = Field(description="LLM model identifier or 'template'")

