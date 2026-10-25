"""SQLAlchemy ORM models for XAI-RAG audit persistence (Part 2 implementation).

Implements tamper-evident audit ledger, document registry, index metadata,
retrieval traces, and reviewer actions. Compatible with SQLite and PostgreSQL.
"""

from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class AuditModel(Base):
    """Primary persistent audit record containing serialized AuditReport."""

    __tablename__ = "audits"

    audit_id = Column(String(64), primary_key=True)
    applicant_ref = Column(String(64), nullable=False, index=True)
    generated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    mode = Column(String(16), nullable=False)  # "demo" or "configured"
    status = Column(String(16), nullable=False)  # "complete", "partial", "failed"
    model_name = Column(String(128), nullable=False)
    model_version = Column(String(32), nullable=False)
    model_hash = Column(String(64), nullable=False)
    prediction_json = Column(Text, nullable=False)
    explanation_json = Column(Text, nullable=False)
    counterfactuals_json = Column(Text, nullable=False)
    search_trace_json = Column(Text, nullable=False)
    evidence_json = Column(Text, nullable=False)
    dossier_json = Column(Text, nullable=False)
    verifier_json = Column(Text, nullable=False)
    report_json = Column(Text, nullable=False)
    human_review_required = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    events = relationship("AuditEventModel", back_populates="audit", cascade="all, delete-orphan")


class AuditEventModel(Base):
    """Cryptographic hash-chained audit event."""

    __tablename__ = "audit_events"

    event_id = Column(String(64), primary_key=True)
    audit_id = Column(String(64), ForeignKey("audits.audit_id"), nullable=False, index=True)
    sequence_num = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    event_type = Column(String(64), nullable=False)
    actor = Column(String(64), nullable=False)
    payload_hash = Column(String(64), nullable=False)
    previous_event_hash = Column(String(64), nullable=False)
    event_hash = Column(String(64), nullable=False)
    payload_json = Column(Text, nullable=False)

    audit = relationship("AuditModel", back_populates="events")


class ModelManifestModel(Base):
    """Registered machine learning model manifest."""

    __tablename__ = "model_manifests"

    model_id = Column(String(64), primary_key=True)
    name = Column(String(128), nullable=False)
    version = Column(String(32), nullable=False)
    artifact_sha256 = Column(String(64), nullable=False)
    calibration_status = Column(String(64), nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class DocumentModel(Base):
    """Authoritative ingested policy document."""

    __tablename__ = "documents"

    source_id = Column(String(64), primary_key=True)
    title = Column(String(256), nullable=False)
    issuer = Column(String(128), nullable=False)
    jurisdiction = Column(String(16), nullable=False)
    source_type = Column(String(64), nullable=False)
    version = Column(String(64), nullable=False)
    effective_date = Column(String(32), nullable=False)
    file_path = Column(String(512), nullable=False)
    sha256_hash = Column(String(64), nullable=False)
    total_pages = Column(Integer, nullable=False)
    parser_status = Column(String(32), nullable=False)
    access_classification = Column(String(32), nullable=False)
    superseded = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    chunks = relationship("ChunkModel", back_populates="document", cascade="all, delete-orphan")


class ChunkModel(Base):
    """Extracted text chunk from an authoritative document."""

    __tablename__ = "chunks"

    chunk_id = Column(String(64), primary_key=True)
    source_id = Column(String(64), ForeignKey("documents.source_id"), nullable=False, index=True)
    page_number = Column(Integer, nullable=False)
    section_title = Column(String(256), nullable=False)
    text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)
    metadata_json = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    document = relationship("DocumentModel", back_populates="chunks")


class IndexBuildModel(Base):
    """Record of vector and lexical index builds."""

    __tablename__ = "index_builds"

    index_version = Column(String(64), primary_key=True)
    document_count = Column(Integer, nullable=False)
    chunk_count = Column(Integer, nullable=False)
    embedding_model = Column(String(128), nullable=False)
    parameters_json = Column(Text, nullable=False)
    build_timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class RetrievalTraceModel(Base):
    """Trace of hybrid search execution and score fusion."""

    __tablename__ = "retrieval_traces"

    trace_id = Column(String(64), primary_key=True)
    audit_id = Column(String(64), nullable=False, index=True)
    dense_query = Column(Text, nullable=False)
    bm25_terms_json = Column(Text, nullable=False)
    metadata_filters_json = Column(Text, nullable=False)
    retrieved_chunk_ids_json = Column(Text, nullable=False)
    dense_scores_json = Column(Text, nullable=False)
    bm25_scores_json = Column(Text, nullable=False)
    fused_scores_json = Column(Text, nullable=False)
    execution_time_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class ReviewerActionModel(Base):
    """Audit log of human credit/compliance reviewer actions."""

    __tablename__ = "reviewer_actions"

    action_id = Column(String(64), primary_key=True)
    audit_id = Column(String(64), nullable=False, index=True)
    reviewer_id = Column(String(64), nullable=False)
    action_type = Column(String(64), nullable=False)
    notes = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
