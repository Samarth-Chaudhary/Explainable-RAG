"""Unit tests for Pillar C: Ontology Service and Translator."""

import pytest
from pathlib import Path

from backend.app.schemas.explanation import ExplanationResult, FeatureContributionSummary
from backend.app.schemas.counterfactual import (
    CounterfactualCandidate,
    CounterfactualMetrics,
    CounterfactualResult,
    DerivedRecalculation,
    FeatureChange,
)
from backend.app.services.ontology_translator import (
    OntologyService,
    OntologyTranslatorService,
    build_naive_baseline_query,
)


@pytest.fixture
def ontology_path() -> Path:
    return Path("data/ontology/credit_risk_ontology_v1.json")


@pytest.fixture
def sample_explanation() -> ExplanationResult:
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
    ext2 = FeatureContributionSummary(
        name="EXT_SOURCE_2",
        label="External Credit Bureau Score 2",
        value=0.24,
        unit="score",
        shap_value=0.21,
        direction="risk_increasing",
        rank=3,
    )
    amt = FeatureContributionSummary(
        name="AMT_CREDIT",
        label="Loan Amount Requested",
        value=1500000.0,
        unit="INR",
        shap_value=-0.05,
        direction="risk_reducing",
        rank=4,
    )
    return ExplanationResult(
        output_space="log_odds",
        base_value=0.20,
        shap_additivity_residual=0.0001,
        tolerance=0.01,
        additivity_status="valid",
        features=[foir, ltv, ext2, amt],
        top_risk_increasing=[foir, ltv],
        top_risk_reducing=[amt],
        explainer_config={"shap_version": "0.45.0", "background": "sample_100"},
        methodology_notes="SHAP measures local linear attribution relative to background reference population.",
    )


@pytest.fixture
def sample_counterfactuals() -> CounterfactualResult:
    return CounterfactualResult(
        request_config={"target_threshold": 0.35, "max_changes": 2},
        candidates=[
            CounterfactualCandidate(
                id="CF-001",
                changes=[
                    FeatureChange(
                        feature="FOIR_RATIO",
                        current=0.68,
                        proposed=0.49,
                        delta=-0.19,
                    ),
                    FeatureChange(
                        feature="LTV_RATIO",
                        current=0.89,
                        proposed=0.75,
                        delta=-0.14,
                    ),
                ],
                derived_recalculations=[],
                predicted_probability=0.31,
                target_met=True,
                constraints_passed=["mutability", "bounds", "re_score"],
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
            time_to_generate_ms=120.0,
            number_of_candidates=1,
            proximity=0.15,
            diversity=0.8,
            ood_flags=0,
            target_miss_magnitude=0.0,
            denominator="1 candidate evaluated",
        ),
    )


def test_ontology_service_load_and_validation(ontology_path: Path):
    """Test loading and validating the domain ontology."""
    service = OntologyService(ontology_path)
    assert service.ontology_version == "1.0.0"
    assert len(service.content_hash) == 64

    # Verify known features exist
    for feat in ["FOIR_RATIO", "LTV_RATIO", "EXT_SOURCE_2", "AMT_CREDIT"]:
        rec = service.get_record(feat)
        assert rec is not None
        assert rec.feature_name == feat
        assert rec.plain_language_definition != ""
        assert rec.unit != ""


def test_ontology_derived_formulas_and_tags(ontology_path: Path):
    """Test that derived ratios expose formulas, tags, and legal topics."""
    service = OntologyService(ontology_path)
    foir = service.get_record("FOIR_RATIO")
    assert foir is not None
    assert foir.formula is not None
    assert "repayment_capacity" in foir.legal_topics
    assert "FOIR_Cap" in foir.metadata_tags

    ltv = service.get_record("LTV_RATIO")
    assert ltv is not None
    assert ltv.formula is not None
    assert any("collateral" in t.lower() for t in ltv.legal_topics)


def test_ontology_unmapped_feature_safe_handling(ontology_path: Path):
    """Test that unmapped features return None without inventing definitions."""
    service = OntologyService(ontology_path)
    assert service.get_record("NON_EXISTENT_FEATURE_123") is None


def test_translator_translation_algorithm(
    ontology_path: Path,
    sample_explanation: ExplanationResult,
    sample_counterfactuals: CounterfactualResult,
):
    """Test full 5-step translation algorithm with validated counterfactuals."""
    ontology_service = OntologyService(ontology_path)
    translator = OntologyTranslatorService(ontology_service)

    result = translator.translate(sample_explanation, sample_counterfactuals)

    assert "FOIR_RATIO" in result.features_selected
    assert "LTV_RATIO" in result.features_selected
    assert "FOIR_RATIO" in result.ontology_matches
    assert "LTV_RATIO" in result.ontology_matches
    assert len(result.unmapped_features) == 0

    payload = result.query_payload
    assert any(term in payload.dense_query.lower() for term in ["obligation", "foir", "repayment"])
    assert any("foir" in term.lower() for term in payload.bm25_terms)
    assert "IN" in payload.metadata_filters.get("jurisdiction", "")


def test_translator_unmapped_features_logged_safely(
    ontology_path: Path,
    sample_counterfactuals: CounterfactualResult,
):
    """Test that explanation with unknown features logs them as unmapped without inventing law."""
    ontology_service = OntologyService(ontology_path)
    translator = OntologyTranslatorService(ontology_service)

    unknown_feat = FeatureContributionSummary(
        name="UNKNOWN_FEATURE_X",
        label="Unknown Feature X",
        value=999.0,
        unit="units",
        shap_value=0.45,
        direction="risk_increasing",
        rank=1,
    )
    explanation_with_unknown = ExplanationResult(
        output_space="log_odds",
        base_value=0.20,
        shap_additivity_residual=0.0001,
        tolerance=0.01,
        additivity_status="valid",
        features=[unknown_feat],
        top_risk_increasing=[unknown_feat],
        top_risk_reducing=[],
        explainer_config={"shap_version": "0.45.0"},
        methodology_notes="Test explanation",
    )

    result = translator.translate(explanation_with_unknown, sample_counterfactuals)
    assert "UNKNOWN_FEATURE_X" in result.unmapped_features
    # Check that generic query does not invent specific legal clauses
    assert "unknown_feature_x" in result.normalized_query.lower()
    assert "rbi" not in result.normalized_query.lower()


def test_naive_baseline_query_builder(sample_explanation: ExplanationResult):
    """Test naive baseline query builder concatenates features + 'loan rejection'."""
    baseline = build_naive_baseline_query(sample_explanation)
    assert "FOIR_RATIO" in baseline.dense_query
    assert "LTV_RATIO" in baseline.dense_query
    assert "loan rejection" in baseline.dense_query.lower()
    assert "loan" in baseline.bm25_terms
    assert "rejection" in baseline.bm25_terms
