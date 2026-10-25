"""Unit tests for TreeSHAP local explanations, output-space fidelity, and additivity verification."""

import numpy as np
import pytest
from scipy.special import expit

from backend.app.schemas.explanation import ExplanationResult
from backend.app.services.explanation import (
    CrossSpaceComparisonError,
    DegradedExplanationError,
    ExplanationService,
)
from backend.app.services.prediction import PredictionService


def test_shap_additivity_in_log_odds_space(
    explanation_service: ExplanationService,
    prediction_service: PredictionService,
    valid_applicant_dict: dict,
):
    """base_value + sum(shap) must reconstruct raw margin within tolerance in log-odds space."""
    prediction = prediction_service.predict(valid_applicant_dict)
    query_df, _ = prediction_service.prepare_feature_vector(valid_applicant_dict)

    result: ExplanationResult = explanation_service.explain(query_df, prediction)

    # 1. Coordinate space check
    assert result.output_space == "log_odds"

    # 2. Additivity equation in log-odds: base_value + sum(shap) == raw_output
    shap_sum = sum(f.shap_value for f in result.features)
    reconstructed_margin = result.base_value + shap_sum

    margin_diff = abs(reconstructed_margin - prediction.raw_output)
    assert margin_diff <= explanation_service.tolerance
    assert result.shap_additivity_residual <= explanation_service.tolerance
    assert result.additivity_status == "valid"

    # 3. Sigmoid link transformation check
    # sigmoid(base_value + sum(shap)) == default_probability
    sigmoid_prob = float(expit(reconstructed_margin))
    prob_diff = abs(sigmoid_prob - prediction.default_probability)
    assert prob_diff <= 1e-4


def test_detect_cross_space_comparison_failure(explanation_service: ExplanationService):
    """Directly comparing log-odds and probability spaces without transformation must raise CrossSpaceComparisonError."""
    log_odds_val = 2.45
    prob_val = 0.92

    with pytest.raises(CrossSpaceComparisonError):
        explanation_service.assert_valid_cross_space_comparison(
            val_a=log_odds_val,
            space_a="log_odds",
            val_b=prob_val,
            space_b="probability",
        )


def test_degraded_additivity_blocks_narrative_generation(
    explanation_service: ExplanationService,
    prediction_service: PredictionService,
    valid_applicant_dict: dict,
):
    """An explanation failing the additivity check must be marked invalid/degraded and block narrative synthesis."""
    prediction = prediction_service.predict(valid_applicant_dict)
    query_df, _ = prediction_service.prepare_feature_vector(valid_applicant_dict)

    # Artificially set an impossible tolerance (1e-12) to trigger degraded/invalid status
    strict_service = ExplanationService(
        model=explanation_service.model,
        feature_schema=explanation_service.feature_schema,
        background_data=explanation_service.background_data,
        output_space="log_odds",
        tolerance=1e-12,  # Impossibly tight tolerance
        top_k=5,
        contribution_cutoff=0.05,
    )
    result = strict_service.explain(query_df, prediction)
    assert result.additivity_status in ("degraded", "invalid")

    # verify that is_safe_for_narrative raises DegradedExplanationError
    with pytest.raises(DegradedExplanationError):
        strict_service.is_safe_for_narrative(result)


def test_feature_contributions_ranking(
    explanation_service: ExplanationService,
    prediction_service: PredictionService,
    valid_applicant_dict: dict,
):
    """Features must be ranked by absolute magnitude, with complete 1-based ranks."""
    prediction = prediction_service.predict(valid_applicant_dict)
    query_df, _ = prediction_service.prepare_feature_vector(valid_applicant_dict)

    result = explanation_service.explain(query_df, prediction)

    # Verify all canonical features are present
    assert len(result.features) == len(explanation_service.feature_schema.feature_order)

    # Check rank ordering
    ranks = [f.rank for f in result.features]
    assert ranks == list(range(1, len(result.features) + 1))

    # Check that magnitude is non-increasing
    magnitudes = [abs(f.shap_value) for f in result.features]
    for i in range(len(magnitudes) - 1):
        assert magnitudes[i] >= magnitudes[i + 1]


def test_shap_additivity_sanity_check_declaration(
    explanation_service: ExplanationService,
    prediction_service: PredictionService,
    valid_applicant_dict: dict,
):
    """TreeSHAP additivity must be explicitly declared as an exact-by-construction sanity check."""
    prediction = prediction_service.predict(valid_applicant_dict)
    query_df, _ = prediction_service.prepare_feature_vector(valid_applicant_dict)

    result = explanation_service.explain(query_df, prediction)
    assert "exact-by-construction sanity check" in result.methodology_notes
    assert result.additivity_status == "valid"


def test_explanation_validity_empirical_metrics():
    """Verify that empirical explanation validity metrics recorded in evaluation_results.json meet standards."""
    import json
    from pathlib import Path

    eval_path = Path(__file__).resolve().parent.parent / "evaluation" / "evaluation_results.json"
    assert eval_path.is_file(), f"Missing evaluation results file at {eval_path}"

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "phase_4_explanation_validity" in data
    phase4 = data["phase_4_explanation_validity"]

    # 1. SHAP vs Permutation Importance Rank Correlation
    shap_perm = phase4["shap_vs_permutation_importance"]
    assert shap_perm["spearman_rho"] >= 0.80, f"Expected high rank concordance, got {shap_perm['spearman_rho']}"
    assert shap_perm["p_value"] < 0.05

    # 2. Stability
    boot_stab = phase4["bootstrap_stability"]
    assert boot_stab["mean_jaccard_vs_full"] >= 0.90
    pert_stab = phase4["perturbation_stability"]
    assert pert_stab["mean_top5_jaccard"] >= 0.70
    assert pert_stab["mean_spearman_correlation"] >= 0.70

    # 3. Faithfulness gains
    faith = phase4["faithfulness_and_monotonicity"]
    assert faith["deletion_faithfulness_gain"] > 0, "SHAP deletion should cause faster drop than random"
    assert faith["insertion_faithfulness_gain"] > 0, "SHAP insertion should recover prediction faster than random"
    assert faith["mean_monotonicity_spearman"] > 0, "Marginal feature drops should correlate with SHAP magnitudes"
