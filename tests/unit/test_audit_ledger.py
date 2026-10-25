"""Unit tests for Tamper-Evident Audit Ledger and Persistence (Part 2 implementation)."""

import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.models.audit_models import AuditEventModel, Base
from backend.app.repositories.audit_repository import AuditRepository
from backend.app.schemas.audit_report import (
    AuditReport,
    ReportDossierSummary,
    ReportExplanationSummary,
    ReportModelSummary,
    ReportPredictionSummary,
    ReportSearchTraceSummary,
    ReportVerifierSummary,
)
from backend.app.schemas.handoff import HybridQueryPayload, RetrievalTrace
from backend.app.services.audit_ledger import AuditLedgerService


@pytest.fixture
def in_memory_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


@pytest.fixture
def ledger_service(in_memory_session) -> AuditLedgerService:
    repo = AuditRepository(in_memory_session)
    return AuditLedgerService(repo)


@pytest.fixture
def sample_audit_report() -> AuditReport:
    return AuditReport(
        audit_id="AUDIT-LEDGER-001",
        applicant_ref="APP-10492-SYNTHETIC",
        generated_at=datetime.now(timezone.utc),
        mode="demo",
        status="complete",
        model=ReportModelSummary(
            name="SYNTHETIC-DEMO-XGB",
            version="1.0.0",
            artifact_sha256="4dfcbc08479e0a6d59f71c4c990264b38d35fbf638b7efea39d3fa6a3782eb9a",
            calibration_status="platt",
        ),
        prediction=ReportPredictionSummary(
            raw_output=1.45,
            default_probability=0.81,
            threshold=0.35,
            label="REJECT_HIGH_RISK",
            output_notes="Probability calibrated via Platt scaling.",
        ),
        explanation=ReportExplanationSummary(
            output_space="log_odds",
            base_value=0.20,
            shap_additivity_residual=0.0001,
            features=[],
        ),
        counterfactuals=[],
        search_trace=ReportSearchTraceSummary(
            ontology_version="1.0.0",
            features_used=["FOIR_RATIO"],
            dense_query="repayment capacity query",
            bm25_terms=["FOIR"],
            metadata_filters={},
            retrieved_chunk_ids=["SYN-POL-FOIR-001-P1-C1"],
            index_version="idx_v1_hash",
        ),
        evidence=[],
        dossier=ReportDossierSummary(
            adverse_action_explanation="Draft decision explanation",
            candidate_review_paths=[],
            regulatory_justification="Section 2.1 policy quote",
            evidence_gaps=[],
            reviewer_notes="Decision support only",
        ),
        verifier=ReportVerifierSummary(
            quote_checks_total=1,
            quote_checks_passed=1,
            quote_checks_failed=0,
            faithfulness_score=1.0,
            normalized_match_policy="NFKC",
        ),
        human_review_required=True,
    )


def test_hash_chain_creation_and_continuity(ledger_service: AuditLedgerService):
    """Test sequential event appending with cryptographic previous_hash chaining."""
    audit_id = "AUDIT-CHAIN-001"
    evt0 = ledger_service.append_event(audit_id, "PREDICTION", "system", {"score": 0.81})
    evt1 = ledger_service.append_event(audit_id, "EXPLANATION", "system", {"top_driver": "FOIR"})
    evt2 = ledger_service.append_event(audit_id, "REPORT_GENERATED", "system", {"status": "complete"})

    assert evt0.previous_event_hash == "0" * 64
    assert evt1.previous_event_hash == evt0.event_hash
    assert evt2.previous_event_hash == evt1.event_hash

    is_valid, issues = ledger_service.verify_chain(audit_id)
    assert is_valid is True
    assert len(issues) == 0


def test_tamper_detection_on_modified_payload(ledger_service: AuditLedgerService, in_memory_session):
    """Test that modifying an event's payload in database breaks chain verification."""
    audit_id = "AUDIT-TAMPER-001"
    ledger_service.append_event(audit_id, "PREDICTION", "system", {"score": 0.81})
    ledger_service.append_event(audit_id, "COUNTERFACTUAL", "system", {"target": 0.35})

    # Tamper with the database row directly
    event_row = in_memory_session.query(AuditEventModel).filter_by(audit_id=audit_id, sequence_num=0).first()
    assert event_row is not None
    # Change the payload in the database without updating the hash
    event_row.payload_json = '{"score": 0.10, "audit_id": "AUDIT-TAMPER-001"}'
    in_memory_session.commit()

    is_valid, issues = ledger_service.verify_chain(audit_id)
    assert is_valid is False
    assert any("Payload tampering detected" in issue for issue in issues)


def test_tamper_detection_on_modified_hash(ledger_service: AuditLedgerService, in_memory_session):
    """Test that altering an event's event_hash in database is detected."""
    audit_id = "AUDIT-TAMPER-002"
    ledger_service.append_event(audit_id, "STEP_1", "system", {"data": 1})
    ledger_service.append_event(audit_id, "STEP_2", "system", {"data": 2})

    event_row = in_memory_session.query(AuditEventModel).filter_by(audit_id=audit_id, sequence_num=1).first()
    assert event_row is not None
    event_row.event_hash = "f" * 64
    in_memory_session.commit()

    is_valid, issues = ledger_service.verify_chain(audit_id)
    assert is_valid is False
    assert any("tampering detected" in issue or "Hash chain broken" in issue for issue in issues)


def test_tamper_detection_on_deleted_event(ledger_service: AuditLedgerService, in_memory_session):
    """Test that deleting an intermediate event in database breaks sequence and chain."""
    audit_id = "AUDIT-TAMPER-003"
    ledger_service.append_event(audit_id, "STEP_0", "system", {"val": 0})
    ledger_service.append_event(audit_id, "STEP_1", "system", {"val": 1})
    ledger_service.append_event(audit_id, "STEP_2", "system", {"val": 2})

    # Delete intermediate event sequence 1
    event_1 = in_memory_session.query(AuditEventModel).filter_by(audit_id=audit_id, sequence_num=1).first()
    assert event_1 is not None
    in_memory_session.delete(event_1)
    in_memory_session.commit()

    is_valid, issues = ledger_service.verify_chain(audit_id)
    assert is_valid is False
    assert any("Sequence broken" in issue for issue in issues)


def test_persist_audit_and_reopen_metadata_check(
    ledger_service: AuditLedgerService,
    sample_audit_report: AuditReport,
):
    """Test persisting full audit report, verifying chain, and checking metadata on reopen."""
    trace = RetrievalTrace(
        trace_id="TRACE-001",
        query_payload=HybridQueryPayload(dense_query="test", bm25_terms=["test"]),
        dense_scores={},
        bm25_scores={},
        fused_scores={},
        retrieved_chunks=[],
        execution_time_ms=10.0,
    )

    audit_id = ledger_service.persist_audit(sample_audit_report, trace, actor="system")
    assert audit_id == sample_audit_report.audit_id

    # Reopen as credit officer
    report, chain_valid, warnings = ledger_service.reopen_audit(
        audit_id,
        expected_model_hash=sample_audit_report.model.artifact_sha256,
        expected_index_version=sample_audit_report.search_trace.index_version,
        requesting_role="credit_officer",
    )

    assert report.audit_id == audit_id
    assert chain_valid is True
    assert len(warnings) == 0

    # Test metadata mismatch warning on reopen
    _, _, warnings_mismatch = ledger_service.reopen_audit(
        audit_id,
        expected_model_hash="wrong_hash_12345",
        expected_index_version="wrong_index_v2",
        requesting_role="compliance_reviewer",
    )
    assert any("Model hash mismatch" in w for w in warnings_mismatch)
    assert any("Index version mismatch" in w for w in warnings_mismatch)


def test_reopen_unauthorized_role_rejected(
    ledger_service: AuditLedgerService,
    sample_audit_report: AuditReport,
):
    """Test that unauthorized roles cannot reopen audit records."""
    trace = RetrievalTrace(
        trace_id="TRACE-001",
        query_payload=HybridQueryPayload(dense_query="test", bm25_terms=["test"]),
        dense_scores={},
        bm25_scores={},
        fused_scores={},
        retrieved_chunks=[],
        execution_time_ms=10.0,
    )
    ledger_service.persist_audit(sample_audit_report, trace)

    with pytest.raises(PermissionError, match="not authorized to reopen audit records"):
        ledger_service.reopen_audit(sample_audit_report.audit_id, requesting_role="unauthorized_public_user")
