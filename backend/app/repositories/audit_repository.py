"""Repository implementation for audit persistence and query operations."""

from __future__ import annotations

import json
from typing import Any
from sqlalchemy.orm import Session

from backend.app.models.audit_models import (
    AuditEventModel,
    AuditModel,
    ChunkModel,
    DocumentModel,
    IndexBuildModel,
    RetrievalTraceModel,
    ReviewerActionModel,
)
from backend.app.schemas.audit_report import AuditReport
from backend.app.schemas.handoff import AuditEvent, Chunk, IndexBuildResult, RetrievalTrace, SourceDocument


class AuditRepository:
    """Repository handling CRUD operations for audits, events, documents, and traces."""

    def __init__(self, session: Session):
        self.session = session

    def save_audit(self, report: AuditReport) -> AuditModel:
        """Persist or update an AuditReport."""
        record = self.session.query(AuditModel).filter_by(audit_id=report.audit_id).first()
        if record is None:
            record = AuditModel(
                audit_id=report.audit_id,
                applicant_ref=report.applicant_ref,
                generated_at=report.generated_at,
                mode=report.mode,
                status=report.status,
                model_name=report.model.name,
                model_version=report.model.version,
                model_hash=report.model.artifact_sha256,
                prediction_json=report.prediction.model_dump_json(),
                explanation_json=report.explanation.model_dump_json(),
                counterfactuals_json=json.dumps([c.model_dump() for c in report.counterfactuals]),
                search_trace_json=report.search_trace.model_dump_json(),
                evidence_json=json.dumps([e.model_dump() for e in report.evidence]),
                dossier_json=report.dossier.model_dump_json(),
                verifier_json=report.verifier.model_dump_json(),
                report_json=report.model_dump_json(),
                human_review_required=report.human_review_required,
            )
            self.session.add(record)
        else:
            record.status = report.status
            record.report_json = report.model_dump_json()
            record.dossier_json = report.dossier.model_dump_json()
            record.verifier_json = report.verifier.model_dump_json()
            record.evidence_json = json.dumps([e.model_dump() for e in report.evidence])

        self.session.commit()
        self.session.refresh(record)
        return record

    def get_audit(self, audit_id: str) -> AuditReport | None:
        """Retrieve and parse AuditReport by audit_id."""
        record = self.session.query(AuditModel).filter_by(audit_id=audit_id).first()
        if record is None:
            return None
        return AuditReport.model_validate_json(record.report_json)

    def list_audits(self, limit: int = 50, offset: int = 0) -> list[AuditModel]:
        """List persisted audit summaries."""
        return (
            self.session.query(AuditModel)
            .order_by(AuditModel.generated_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def append_event(self, event: AuditEvent, payload_json: str) -> AuditEventModel:
        """Append an immutable, hash-chained audit event."""
        # Find next sequence number
        last_event = (
            self.session.query(AuditEventModel)
            .filter_by(audit_id=event.event_id.split("-")[0] if "-" in event.event_id else "")
            .order_by(AuditEventModel.sequence_num.desc())
            .first()
        )
        # We also support explicit audit_id passed in payload_json or parsed
        audit_id = "global"
        try:
            p = json.loads(payload_json)
            audit_id = p.get("audit_id", "global")
        except Exception:
            pass

        seq_num = 0
        last_audit_event = (
            self.session.query(AuditEventModel)
            .filter_by(audit_id=audit_id)
            .order_by(AuditEventModel.sequence_num.desc())
            .first()
        )
        if last_audit_event is not None:
            seq_num = last_audit_event.sequence_num + 1

        record = AuditEventModel(
            event_id=event.event_id,
            audit_id=audit_id,
            sequence_num=seq_num,
            timestamp=event.timestamp,
            event_type=event.event_type,
            actor=event.actor,
            payload_hash=event.payload_hash,
            previous_event_hash=event.previous_event_hash,
            event_hash=event.event_hash,
            payload_json=payload_json,
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def get_events_for_audit(self, audit_id: str) -> list[AuditEventModel]:
        """Get all hash-chained events for an audit ordered by sequence_num."""
        return (
            self.session.query(AuditEventModel)
            .filter_by(audit_id=audit_id)
            .order_by(AuditEventModel.sequence_num.asc())
            .all()
        )

    def get_last_event_for_audit(self, audit_id: str) -> AuditEventModel | None:
        """Get the latest event in the chain for an audit."""
        return (
            self.session.query(AuditEventModel)
            .filter_by(audit_id=audit_id)
            .order_by(AuditEventModel.sequence_num.desc())
            .first()
        )

    def save_document(
        self,
        doc: SourceDocument,
        parser_status: str = "parsed",
        access_classification: str = "internal",
        superseded: bool = False,
    ) -> DocumentModel:
        """Register or update an authoritative source document."""
        record = self.session.query(DocumentModel).filter_by(source_id=doc.source_id).first()
        if record is None:
            record = DocumentModel(
                source_id=doc.source_id,
                title=doc.title,
                issuer=doc.issuer,
                jurisdiction=doc.jurisdiction,
                source_type="regulatory_policy",
                version=doc.version,
                effective_date=doc.effective_date,
                file_path=doc.file_path,
                sha256_hash=doc.sha256_hash,
                total_pages=doc.total_pages,
                parser_status=parser_status,
                access_classification=access_classification,
                superseded=superseded,
            )
            self.session.add(record)
        else:
            record.total_pages = doc.total_pages
            record.parser_status = parser_status
            record.superseded = superseded

        self.session.commit()
        return record

    def list_documents(self) -> list[DocumentModel]:
        """List all registered document models."""
        return self.session.query(DocumentModel).order_by(DocumentModel.created_at.desc()).all()

    def get_document(self, source_id: str) -> DocumentModel | None:
        """Fetch document model by source_id."""
        return self.session.query(DocumentModel).filter_by(source_id=source_id).first()

    def update_document_status(self, source_id: str, superseded: bool) -> DocumentModel | None:
        """Update superseded status for a document."""
        rec = self.get_document(source_id)
        if rec:
            rec.superseded = superseded
            self.session.commit()
            self.session.refresh(rec)
        return rec

    def save_chunks(self, chunks: list[Chunk]) -> None:
        """Persist or update chunk records."""
        for c in chunks:
            existing = self.session.query(ChunkModel).filter_by(chunk_id=c.chunk_id).first()
            if existing is None:
                new_chunk = ChunkModel(
                    chunk_id=c.chunk_id,
                    source_id=c.source_id,
                    page_number=c.page_number,
                    section_title=c.section_title,
                    text=c.text,
                    token_count=c.token_count,
                    metadata_json=json.dumps(c.metadata),
                )
                self.session.add(new_chunk)
            else:
                existing.text = c.text
                existing.page_number = c.page_number
                existing.section_title = c.section_title
                existing.metadata_json = json.dumps(c.metadata)
        self.session.commit()

    def get_chunk(self, chunk_id: str) -> ChunkModel | None:
        """Retrieve chunk by chunk_id."""
        return self.session.query(ChunkModel).filter_by(chunk_id=chunk_id).first()

    def get_all_chunks(self) -> list[Chunk]:
        """Fetch all chunks as Pydantic Chunk objects."""
        records = self.session.query(ChunkModel).all()
        chunks: list[Chunk] = []
        for r in records:
            chunks.append(
                Chunk(
                    chunk_id=r.chunk_id,
                    source_id=r.source_id,
                    text=r.text,
                    page_number=r.page_number,
                    section_title=r.section_title,
                    token_count=r.token_count,
                    metadata=json.loads(r.metadata_json) if r.metadata_json else {},
                )
            )
        return chunks

    def save_retrieval_trace(self, trace: RetrievalTrace, audit_id: str) -> RetrievalTraceModel:
        """Persist retrieval trace separately from narrative."""
        record = RetrievalTraceModel(
            trace_id=trace.trace_id,
            audit_id=audit_id,
            dense_query=trace.query_payload.dense_query,
            bm25_terms_json=json.dumps(trace.query_payload.bm25_terms),
            metadata_filters_json=json.dumps(trace.query_payload.metadata_filters),
            retrieved_chunk_ids_json=json.dumps([c.chunk_id for c in trace.retrieved_chunks]),
            dense_scores_json=json.dumps(trace.dense_scores),
            bm25_scores_json=json.dumps(trace.bm25_scores),
            fused_scores_json=json.dumps(trace.fused_scores),
            execution_time_ms=trace.execution_time_ms,
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def get_retrieval_trace(self, audit_id: str) -> RetrievalTraceModel | None:
        """Fetch persisted retrieval trace by audit_id."""
        return self.session.query(RetrievalTraceModel).filter_by(audit_id=audit_id).first()

    def save_index_build(self, build: IndexBuildResult, params: dict[str, Any]) -> IndexBuildModel:
        """Save or update index build metadata."""
        record = self.session.query(IndexBuildModel).filter_by(index_version=build.index_version).first()
        if record is None:
            record = IndexBuildModel(
                index_version=build.index_version,
                document_count=build.document_count,
                chunk_count=build.chunk_count,
                embedding_model=build.embedding_model,
                parameters_json=json.dumps(params),
                build_timestamp=build.build_timestamp,
            )
            self.session.add(record)
        else:
            record.document_count = build.document_count
            record.chunk_count = build.chunk_count
            record.embedding_model = build.embedding_model
            record.parameters_json = json.dumps(params)
            record.build_timestamp = build.build_timestamp
        self.session.commit()
        self.session.refresh(record)
        return record

    def record_reviewer_action(
        self,
        action_id: str,
        audit_id: str,
        reviewer_id: str,
        action_type: str,
        notes: str,
    ) -> ReviewerActionModel:
        """Record a human reviewer action."""
        record = ReviewerActionModel(
            action_id=action_id,
            audit_id=audit_id,
            reviewer_id=reviewer_id,
            action_type=action_type,
            notes=notes,
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record
