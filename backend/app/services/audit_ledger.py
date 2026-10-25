"""Tamper-evident audit ledger service (Part 2 implementation).

Implements append-only event logging with cryptographic SHA-256 hash chaining
across all execution stages (including failures). Provides chain verification,
payload integrity checking, and reproducible audit reopening.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from typing import Any, Protocol
import uuid

from backend.app.models.audit_models import AuditEventModel
from backend.app.repositories.audit_repository import AuditRepository
from backend.app.schemas.audit_report import AuditReport
from backend.app.schemas.handoff import AuditEvent, RetrievalTrace

logger = logging.getLogger(__name__)

GENESIS_PREV_HASH = "0" * 64


class AuditLedgerProtocol(Protocol):
    """Protocol for cryptographic hash-chained audit logging."""

    def append_event(
        self,
        audit_id: str,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
    ) -> AuditEvent:
        ...

    def verify_chain(self, audit_id: str) -> tuple[bool, list[str]]:
        ...


def canonical_json(data: Any) -> str:
    """Serialize data to deterministic, canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def compute_sha256(text: str) -> str:
    """Compute SHA-256 hex digest."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_timestamp(dt: datetime | str) -> str:
    """Format datetime consistently for canonical hash computation."""
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except Exception:
            return dt
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class TamperDetectionError(Exception):
    """Raised when ledger tamper or broken hash chain is detected."""


class AuditLedgerService:
    """Append-only, tamper-evident audit ledger service."""

    def __init__(self, repository: AuditRepository):
        self.repository = repository

    def append_event(
        self,
        audit_id: str,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
    ) -> AuditEvent:
        """Append an immutable, hash-chained audit event to the ledger."""
        # 1. Canonical payload hash with audit_id bound
        persisted_payload = dict(payload)
        persisted_payload["audit_id"] = audit_id
        payload_canonical = canonical_json(persisted_payload)
        payload_hash = compute_sha256(payload_canonical)

        # 2. Get latest preceding event in chain for this audit_id
        last_event: AuditEventModel | None = self.repository.get_last_event_for_audit(audit_id)
        if last_event is None:
            sequence_num = 0
            previous_event_hash = GENESIS_PREV_HASH
        else:
            sequence_num = last_event.sequence_num + 1
            previous_event_hash = last_event.event_hash

        event_id = f"{audit_id}-evt-{sequence_num:04d}-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc)
        timestamp_str = format_timestamp(now)

        # 3. Canonical block for event hash
        block = {
            "event_id": event_id,
            "audit_id": audit_id,
            "sequence_num": sequence_num,
            "timestamp": timestamp_str,
            "event_type": event_type,
            "actor": actor,
            "payload_hash": payload_hash,
            "previous_event_hash": previous_event_hash,
        }
        event_hash = compute_sha256(canonical_json(block))

        # 4. Construct Pydantic event object
        event_obj = AuditEvent(
            event_id=event_id,
            timestamp=now,
            event_type=event_type,
            actor=actor,
            payload_hash=payload_hash,
            previous_event_hash=previous_event_hash,
            event_hash=event_hash,
        )

        # 5. Persist to repository
        self.repository.append_event(event_obj, payload_canonical)

        logger.info(
            "Appended audit event %s (seq=%d, type=%s, hash=%s)",
            event_id,
            sequence_num,
            event_type,
            event_hash[:12],
        )
        return event_obj

    def append(self, event: AuditEvent, payload: dict[str, Any], audit_id: str) -> AuditEvent:
        """Compatibility wrapper for P2.0 handoff interface."""
        return self.append_event(
            audit_id=audit_id,
            event_type=event.event_type,
            actor=event.actor,
            payload=payload,
        )

    def verify_chain(self, audit_id: str) -> tuple[bool, list[str]]:
        """Verify the complete cryptographic hash chain for an audit run.

        Checks:
        1. Non-empty event history.
        2. Sequence numbers are strictly monotonically increasing (0, 1, 2, ...).
        3. Genesis event points to GENESIS_PREV_HASH.
        4. Each subsequent event's previous_event_hash equals previous event's event_hash.
        5. Payload hash recomputed from stored JSON matches event.payload_hash.
        6. Event hash recomputed from canonical block matches event.event_hash.
        """
        events = self.repository.get_events_for_audit(audit_id)
        if not events:
            return False, [f"No events found for audit_id: {audit_id}"]

        tamper_records: list[str] = []
        expected_prev_hash = GENESIS_PREV_HASH

        for expected_seq, evt in enumerate(events):
            # Check 1: Sequence number continuity
            if evt.sequence_num != expected_seq:
                tamper_records.append(
                    f"Sequence broken at event {evt.event_id}: expected seq {expected_seq}, got {evt.sequence_num}"
                )

            # Check 2: Previous event hash chaining
            if evt.previous_event_hash != expected_prev_hash:
                tamper_records.append(
                    f"Hash chain broken at event {evt.event_id} (seq {evt.sequence_num}): "
                    f"stored previous_hash={evt.previous_event_hash[:12]} != expected={expected_prev_hash[:12]}"
                )

            # Check 3: Payload hash integrity
            try:
                payload_data = json.loads(evt.payload_json)
                recomputed_payload_hash = compute_sha256(canonical_json(payload_data))
                if recomputed_payload_hash != evt.payload_hash:
                    tamper_records.append(
                        f"Payload tampering detected at event {evt.event_id} (seq {evt.sequence_num}): "
                        f"recomputed payload hash {recomputed_payload_hash[:12]} != stored {evt.payload_hash[:12]}"
                    )
            except Exception as e:
                tamper_records.append(f"Payload parse failure at event {evt.event_id}: {e}")

            # Check 4: Event hash integrity
            block = {
                "event_id": evt.event_id,
                "audit_id": evt.audit_id,
                "sequence_num": evt.sequence_num,
                "timestamp": format_timestamp(evt.timestamp),
                "event_type": evt.event_type,
                "actor": evt.actor,
                "payload_hash": evt.payload_hash,
                "previous_event_hash": evt.previous_event_hash,
            }
            recomputed_event_hash = compute_sha256(canonical_json(block))
            if recomputed_event_hash != evt.event_hash:
                tamper_records.append(
                    f"Event block tampering detected at event {evt.event_id} (seq {evt.sequence_num}): "
                    f"recomputed hash {recomputed_event_hash[:12]} != stored {evt.event_hash[:12]}"
                )

            expected_prev_hash = evt.event_hash

        is_valid = len(tamper_records) == 0
        return is_valid, tamper_records

    def persist_audit(
        self,
        report: AuditReport,
        retrieval_trace: RetrievalTrace,
        actor: str = "system",
    ) -> str:
        """Persist full audit record, separate retrieval trace, and record completion event."""
        # 1. Save AuditReport in primary repository
        self.repository.save_audit(report)

        # 2. Save retrieval trace separately from LLM narrative
        self.repository.save_retrieval_trace(retrieval_trace, audit_id=report.audit_id)

        # 3. Append completion audit event
        self.append_event(
            audit_id=report.audit_id,
            event_type="AUDIT_PERSISTED",
            actor=actor,
            payload={
                "audit_id": report.audit_id,
                "applicant_ref": report.applicant_ref,
                "status": report.status,
                "model_name": report.model.name,
                "model_version": report.model.version,
                "model_hash": report.model.artifact_sha256,
                "faithfulness_score": report.verifier.faithfulness_score,
                "evidence_count": len(report.evidence),
                "human_review_required": report.human_review_required,
            },
        )

        return report.audit_id

    def reopen_audit(
        self,
        audit_id: str,
        expected_model_hash: str | None = None,
        expected_index_version: str | None = None,
        requesting_role: str = "credit_officer",
    ) -> tuple[AuditReport, bool, list[str]]:
        """Reopen a saved audit, verifying authorization, hash chain, and metadata consistency."""
        authorized_roles = {
            "analyst",
            "reviewer",
            "credit_officer",
            "compliance_reviewer",
            "auditor",
            "model_risk_analyst",
            "admin",
        }
        if requesting_role not in authorized_roles:
            raise PermissionError(
                f"Role '{requesting_role}' is not authorized to reopen audit records. Required roles: {authorized_roles}"
            )

        report = self.repository.get_audit(audit_id)
        if report is None:
            raise ValueError(f"Audit record not found: {audit_id}")

        # Verify hash chain
        chain_valid, chain_issues = self.verify_chain(audit_id)

        warnings: list[str] = []
        if not chain_valid:
            warnings.extend(chain_issues)

        # Metadata consistency checks
        if expected_model_hash and report.model.artifact_sha256 != expected_model_hash:
            warnings.append(
                f"Model hash mismatch on reopen: record has {report.model.artifact_sha256[:12]}, expected {expected_model_hash[:12]}"
            )

        if expected_index_version and report.search_trace.index_version != expected_index_version:
            warnings.append(
                f"Index version mismatch on reopen: record has {report.search_trace.index_version[:12]}, expected {expected_index_version[:12]}"
            )

        # Record audit reopened access event
        self.append_event(
            audit_id=audit_id,
            event_type="AUDIT_REOPENED",
            actor=requesting_role,
            payload={
                "audit_id": audit_id,
                "chain_valid": chain_valid,
                "warning_count": len(warnings),
            },
        )

        return report, chain_valid, warnings
