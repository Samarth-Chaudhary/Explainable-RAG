"""Unit tests for Structured Report Generator service (Part 2 implementation)."""

import pytest
from datetime import datetime, timezone

from backend.app.schemas.explanation import ExplanationResult, FeatureContributionSummary
from backend.app.schemas.counterfactual import (
    CounterfactualCandidate,
    CounterfactualMetrics,
    CounterfactualResult,
    FeatureChange,
)
from backend.app.schemas.handoff import (
    Chunk,
    DraftReport,
    GeneratorInputPayload,
    HybridQueryPayload,
    RetrievalResult,
    RetrievalTrace,
    RetrievedEvidence,
    TranslationResult,
)
from backend.app.schemas.model_manifest import ModelManifest
from backend.app.schemas.prediction import ModelMetadataSummary, PredictionResult, ThresholdConfig
from backend.app.services.report_generator import (
    ExplanationAdditivityError,
    GenericLLMProvider,
    ReportGeneratorService,
)


def create_sample_payload() -> GeneratorInputPayload:
    manifest = ModelManifest(
        name="SYNTHETIC-DEMO-XGB",
        version="1.0.0",
        artifact_path="data/demo/model.json",
        artifact_sha256="4dfcbc08479e0a6d59f71c4c990264b38d35fbf638b7efea39d3fa6a3782eb9a",
        feature_schema_hash="schema_hash_123",
        calibration_status="platt",
        created_timestamp=datetime.now(timezone.utc),
    )
    meta_summary = ModelMetadataSummary(
        name="SYNTHETIC-DEMO-XGB",
        version="1.0.0",
        artifact_sha256="4dfcbc08479e0a6d59f71c4c990264b38d35fbf638b7efea39d3fa6a3782eb9a",
        calibration_status="platt",
    )
    threshold_cfg = ThresholdConfig(
        value=0.35,
        source="Credit Committee Policy Sec 4.1",
        owner="Chief Risk Officer",
    )
    pred = PredictionResult(
        model=meta_summary,
        raw_output=1.45,
        default_probability=0.81,
        threshold=threshold_cfg,
        decision_label="REJECT_HIGH_RISK",
        output_notes="Probability calibrated via isotonic scaling.",
        inference_timestamp=datetime.now(timezone.utc),
        feature_order_hash="dummy_order_hash",
        features_evaluated={"FOIR_RATIO": 0.68, "LTV_RATIO": 0.89},
    )

    foir = FeatureContributionSummary(
        name="FOIR_RATIO",
        label="Fixed Obligation to Income Ratio",
        value=0.68,
        unit="ratio",
        shap_value=0.34,
        direction="risk_increasing",
        rank=1,
    )
    ltv = FeatureContributionSummary(
        name="LTV_RATIO",
        label="Loan to Value Ratio",
        value=0.89,
        unit="ratio",
        shap_value=0.22,
        direction="risk_increasing",
        rank=2,
    )

    expl = ExplanationResult(
        output_space="log_odds",
        base_value=0.20,
        shap_additivity_residual=0.0001,
        tolerance=0.01,
        additivity_status="valid",
        features=[foir, ltv],
        top_risk_increasing=[foir, ltv],
        top_risk_reducing=[],
        explainer_config={"shap_version": "0.45.0"},
        methodology_notes="TreeSHAP log-odds decomposition.",
    )

    cf = CounterfactualResult(
        request_config={"target_threshold": 0.35},
        candidates=[
            CounterfactualCandidate(
                id="CF-001",
                changes=[
                    FeatureChange(feature="FOIR_RATIO", current=0.68, proposed=0.49, delta=-0.19),
                    FeatureChange(feature="LTV_RATIO", current=0.89, proposed=0.75, delta=-0.14),
                ],
                derived_recalculations=[],
                predicted_probability=0.31,
                target_met=True,
                constraints_passed=["mutability", "bounds"],
                constraints_failed=[],
                feasibility_status="feasible",
                caveats=["Model scenario only; not a credit approval guarantee."],
                unmodelled_dependencies=[],
            )
        ],
        no_feasible_found=False,
        no_feasible_message=None,
        closest_non_satisfying=[],
        metrics=CounterfactualMetrics(
            validity=1.0,
            actionability=1.0,
            generation_failure_rate=0.0,
            time_to_generate_ms=100.0,
            number_of_candidates=1,
            proximity=0.15,
            diversity=0.8,
            ood_flags=0,
            target_miss_magnitude=0.0,
            denominator="1 candidate",
        ),
    )

    query_payload = HybridQueryPayload(
        dense_query="FOIR and LTV policy limits",
        bm25_terms=["FOIR", "LTV"],
        metadata_filters={"jurisdiction": "IN"},
        top_k=2,
        source_families=["underwriting_rules"],
    )

    trans = TranslationResult(
        query_payload=query_payload,
        features_selected=["FOIR_RATIO", "LTV_RATIO"],
        ontology_matches=["FOIR_RATIO", "LTV_RATIO"],
        unmapped_features=[],
        normalized_query="FOIR and LTV limits",
        lexical_terms=["FOIR", "LTV"],
        metadata_filters={"jurisdiction": "IN"},
        ontology_version="1.0.0",
        ontology_content_hash="mock_hash_123",
    )

    chunk1 = Chunk(
        chunk_id="SYN-POL-FOIR-001-P1-C1",
        source_id="SYN-POL-FOIR-001",
        text="Section 2.1 FOIR Ceiling. Borrowers must maintain a Fixed Obligation to Income Ratio not exceeding 65 percent.",
        page_number=1,
        section_title="Section 2.1 FOIR Ceiling",
        token_count=18,
        metadata={"title": "FOIR Policy", "issuer": "Policy Committee"},
    )

    ret = RetrievalResult(
        evidence=[
            RetrievedEvidence(
                evidence_id="EV-001",
                chunk=chunk1,
                extracted_quote="Borrowers must maintain a Fixed Obligation to Income Ratio not exceeding 65 percent.",
                relevance_justification="Directly establishes maximum debt service burden for retail applicants.",
                verification_status="unverified",
            )
        ],
        retrieval_trace=RetrievalTrace(
            trace_id="TRACE-001",
            query_payload=query_payload,
            dense_scores={"SYN-POL-FOIR-001-P1-C1": 0.95},
            bm25_scores={"SYN-POL-FOIR-001-P1-C1": 12.5},
            fused_scores={"SYN-POL-FOIR-001-P1-C1": 0.033},
            retrieved_chunks=[chunk1],
            execution_time_ms=15.0,
        ),
        no_evidence_found=False,
        missing_coverage=[],
        applicability_warnings=[],
    )

    return GeneratorInputPayload(
        audit_id="AUDIT-TEST-001",
        applicant_ref="APP-10492-SYNTHETIC",
        mode="demo",
        model_manifest=manifest,
        prediction=pred,
        explanation=expl,
        counterfactuals=cf,
        translation=trans,
        retrieval=ret,
        index_version="idx_version_test_123",
        user_role="credit_officer",
    )


@pytest.fixture
def sample_payload() -> GeneratorInputPayload:
    """Fixture providing sample valid GeneratorInputPayload."""
    return create_sample_payload()


def test_deterministic_local_report_generation(sample_payload: GeneratorInputPayload):
    """Test deterministic report generator produces schema-valid DraftReport with required sections."""
    service = ReportGeneratorService(mode="demo")
    draft = service.generate(sample_payload)

    assert isinstance(draft, DraftReport)
    assert draft.audit_id == "AUDIT-TEST-001"
    assert draft.status == "complete"
    assert draft.human_review_required is True
    assert len(draft.prompt_template_hash) == 64

    # 1. Adverse action explanation
    adverse = draft.dossier.adverse_action_explanation
    assert "Draft decision explanation" in adverse
    assert "for human review" in adverse
    assert "FOIR_RATIO" in adverse

    # 2. Candidate review paths
    assert len(draft.dossier.candidate_review_paths) == 1
    assert "Model scenario, not a promise of approval" in draft.dossier.candidate_review_paths[0]

    # 3. Regulatory justification
    assert "[SYNTHETIC DEMO POLICY - NOT LAW]" in draft.dossier.regulatory_justification

    # 4. Reviewer notes
    assert "decision-support mode" in draft.dossier.reviewer_notes


def test_invalid_shap_additivity_blocks_generation(sample_payload: GeneratorInputPayload):
    """Test that invalid SHAP additivity raises ExplanationAdditivityError and blocks report."""
    invalid_expl = sample_payload.explanation.model_copy(
        update={"additivity_status": "invalid", "shap_additivity_residual": 0.55}
    )
    bad_payload = sample_payload.model_copy(update={"explanation": invalid_expl})

    service = ReportGeneratorService()
    with pytest.raises(ExplanationAdditivityError, match="SHAP explanation additivity is INVALID"):
        service.generate(bad_payload)


def test_degraded_shap_additivity_labels_report_partial(sample_payload: GeneratorInputPayload):
    """Test that degraded SHAP additivity labels the report status as partial."""
    degraded_expl = sample_payload.explanation.model_copy(
        update={"additivity_status": "degraded", "shap_additivity_residual": 0.03}
    )
    degraded_payload = sample_payload.model_copy(update={"explanation": degraded_expl})

    service = ReportGeneratorService()
    draft = service.generate(degraded_payload)
    assert draft.status == "partial"


def test_missing_evidence_labels_report_partial_with_gap(sample_payload: GeneratorInputPayload):
    """Test that missing evidence marks report partial and records explicit gap without inventing law."""
    empty_ret = sample_payload.retrieval.model_copy(
        update={"evidence": [], "no_evidence_found": True, "missing_coverage": ["repayment_capacity"]}
    )
    gap_payload = sample_payload.model_copy(update={"retrieval": empty_ret})

    service = ReportGeneratorService()
    draft = service.generate(gap_payload)

    assert draft.status == "partial"
    assert "No authoritative policy evidence found" in draft.dossier.regulatory_justification
    assert any("repayment_capacity" in g for g in draft.dossier.evidence_gaps)


def test_external_llm_provider_bounded_retry():
    """Test that external LLM adapter retries invalid JSON and recovers on valid response."""
    calls = 0

    def mock_llm_func(prompt: str, sys: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return "This is malformed non-JSON response from LLM"
        return """
        {
            "adverse_action_explanation": "Draft decision notice for human review",
            "candidate_review_paths": ["Path 1: Tenor extension"],
            "regulatory_justification": "Section 2.1 policy quote",
            "evidence_gaps": ["None"],
            "reviewer_notes": "Decision support only"
        }
        """

    provider = GenericLLMProvider(mock_llm_func, max_retries=2)
    dossier = provider.generate_dossier("Test prompt", "System prompt")

    assert calls == 2
    assert "Draft decision notice" in dossier.adverse_action_explanation


def test_build_prompt_isolation_and_external_llm_fallback(sample_payload: GeneratorInputPayload):
    """Test prompt building data isolation and external provider execution/fallback."""
    service = ReportGeneratorService()
    prompt = service._build_prompt(sample_payload)

    assert "AUDIT ID: AUDIT-TEST-001" in prompt
    assert "<untrusted_policy_data nonce=" in prompt
    assert "CRITICAL SECURITY DIRECTIVE FOR RETRIEVED PASSAGES" in prompt

    # Test with external provider that fails; verifies graceful fallback
    class FailingProvider:
        provider_name = "failing_llm"
        model_name = "broken_v1"
        def generate_dossier(self, p: str, s: str):
            raise RuntimeError("API quota exceeded")

    service_with_ext = ReportGeneratorService(llm_provider=FailingProvider())
    draft = service_with_ext.generate(sample_payload)
    assert draft.status == "partial"
    assert any("External LLM generation failed" in g for g in draft.dossier.evidence_gaps)
