"""Unit tests for DiCE constrained counterfactual optimization and feasibility validation."""

import pytest
from backend.app.schemas.counterfactual import CounterfactualResult
from backend.app.schemas.prediction import InputValidationError
from backend.app.services.counterfactual import CounterfactualService
from backend.app.services.prediction import PredictionService


def test_immutable_field_alteration_is_rejected(
    counterfactual_service: CounterfactualService,
    valid_applicant_dict: dict,
):
    """Post-generation validator must strictly reject any candidate where an immutable field was altered."""
    _, orig_eval = counterfactual_service.prediction_service.prepare_feature_vector(valid_applicant_dict)

    # Malicious / invalid candidate: artificially reduced applicant age and erased delinquencies
    bad_candidate = dict(valid_applicant_dict)
    bad_candidate["AGE"] = 25.0  # Original was 38
    bad_candidate["PAST_DELINQUENCIES"] = 0  # Original was 2
    bad_candidate["AMT_ANNUITY"] = 30000.0  # Actionable lever

    validated = counterfactual_service.validate_candidate(
        candidate_id="BAD-CF",
        original_dict=orig_eval,
        candidate_dict=bad_candidate,
    )

    # Candidate must have failed constraints
    assert len(validated.constraints_failed) > 0
    assert any("AGE" in f for f in validated.constraints_failed)
    assert any("PAST_DELINQUENCIES" in f for f in validated.constraints_failed)
    assert validated.feasibility_status != "feasible"


def test_derived_ratios_recomputed_from_primitives(
    counterfactual_service: CounterfactualService,
    valid_applicant_dict: dict,
):
    """When actionable primitive levers change, derived ratios (FOIR and LTV) must be recomputed before re-scoring."""
    _, orig_eval = counterfactual_service.prediction_service.prepare_feature_vector(valid_applicant_dict)

    # Candidate modifying annuity (tenor) and collateral value
    candidate_inputs = dict(valid_applicant_dict)
    candidate_inputs["AMT_ANNUITY"] = 31500.0  # Reduced annuity
    candidate_inputs["AMT_GOODS_PRICE"] = 2000000.0  # Additional collateral

    validated = counterfactual_service.validate_candidate(
        candidate_id="CF-RECOMPUTE",
        original_dict=orig_eval,
        candidate_dict=candidate_inputs,
    )

    recalcs = {r.feature: r.new_value for r in validated.derived_recalculations}
    assert "FOIR_RATIO" in recalcs
    assert "LTV_RATIO" in recalcs

    # Expected FOIR: (31500 + 2600) / 70000 = 34100 / 70000 = 0.4871
    assert abs(recalcs["FOIR_RATIO"] - 0.4871) < 1e-3

    # Expected LTV: 1500000 / 2000000 = 0.75
    assert abs(recalcs["LTV_RATIO"] - 0.75) < 1e-3


def test_production_rescoring_integrity(
    counterfactual_service: CounterfactualService,
    valid_applicant_dict: dict,
):
    """Predicted probability on counterfactual candidates must come from direct model re-scoring."""
    res: CounterfactualResult = counterfactual_service.generate_counterfactuals(
        applicant_data=valid_applicant_dict,
        total_cfs=2,
    )

    for cand in res.candidates:
        # Reconstruct candidate dictionary
        cand_dict = dict(valid_applicant_dict)
        for ch in cand.changes:
            cand_dict[ch.feature] = ch.proposed

        # Re-score independently via production service
        independent_res = counterfactual_service.prediction_service.predict(cand_dict)
        assert abs(cand.predicted_probability - independent_res.default_probability) < 1e-4


def test_no_feasible_found_exact_message(
    counterfactual_service: CounterfactualService,
    valid_applicant_dict: dict,
):
    """When target threshold is set impossibly low (e.g. 0.0001), exact required message must be returned."""
    strict_service = CounterfactualService(
        prediction_service=counterfactual_service.prediction_service,
        feature_schema=counterfactual_service.feature_schema,
        training_df=counterfactual_service.training_df,
        target_threshold=0.00001,  # Impossibly low risk target
    )

    res = strict_service.generate_counterfactuals(valid_applicant_dict, total_cfs=2)

    assert res.no_feasible_found is True
    assert res.no_feasible_message == CounterfactualService.EXACT_NO_FEASIBLE_MESSAGE
    assert len(res.candidates) == 0
    assert len(res.closest_non_satisfying) > 0


def test_out_of_range_counterfactual_rejected(
    counterfactual_service: CounterfactualService,
    valid_applicant_dict: dict,
):
    """Candidate proposing an out-of-range value (e.g. negative loan amount) must fail constraints."""
    _, orig_eval = counterfactual_service.prediction_service.prepare_feature_vector(valid_applicant_dict)

    bad_cand = dict(valid_applicant_dict)
    bad_cand["AMT_CREDIT"] = -50000.0  # Negative credit amount

    validated = counterfactual_service.validate_candidate(
        candidate_id="OUT-OF-BOUNDS",
        original_dict=orig_eval,
        candidate_dict=bad_cand,
    )

    assert any("AMT_CREDIT" in f and "below minimum" in f for f in validated.constraints_failed)
    assert validated.feasibility_status != "feasible"


def test_unsupported_coapplicant_field_rejected(
    prediction_service: PredictionService,
    valid_applicant_dict: dict,
):
    """Attempting to alter score by injecting an unsupported co-applicant field must be rejected by schema."""
    bad_cand = dict(valid_applicant_dict)
    bad_cand["CO_APPLICANT_INCOME"] = 100000.0

    with pytest.raises(InputValidationError):
        prediction_service.prepare_feature_vector(bad_cand)


@pytest.mark.parametrize(
    "immutable_field,mutated_value",
    [
        ("AGE", 21.0),
        ("CREDIT_BUREAU_SCORE", 850.0),
        ("PAST_DELINQUENCIES", 0),
        ("DAYS_EMPLOYED", -5000.0),
        ("AMT_INCOME_TOTAL", 500000.0),
        ("EXISTING_DEBT_OBLIGATIONS", 0.0),
    ],
)
def test_all_immutable_and_bureau_fields_strictly_fail_mutation(
    counterfactual_service: CounterfactualService,
    valid_applicant_dict: dict,
    immutable_field: str,
    mutated_value: float,
):
    """Enforce that every protected demographic and bureau-derived field cannot be mutated."""
    _, orig_eval = counterfactual_service.prediction_service.prepare_feature_vector(valid_applicant_dict)

    bad_candidate = dict(valid_applicant_dict)
    bad_candidate[immutable_field] = mutated_value

    validated = counterfactual_service.validate_candidate(
        candidate_id=f"MUTATE-{immutable_field}",
        original_dict=orig_eval,
        candidate_dict=bad_candidate,
    )

    # Must be marked as failure and infeasible
    assert len(validated.constraints_failed) > 0, f"Expected constraint failure for mutating {immutable_field}"
    assert any(immutable_field in f for f in validated.constraints_failed)
    assert validated.feasibility_status != "feasible"


def test_real_test_cohort_counterfactual_evaluation_metrics_loaded():
    """Verify that real counterfactual evaluation on >=200 test rejections is recorded in evaluation_results.json."""
    import json
    from pathlib import Path

    eval_path = Path(__file__).resolve().parent.parent / "evaluation" / "evaluation_results.json"
    assert eval_path.is_file(), f"Missing evaluation results at {eval_path}"

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "phase_5_counterfactual_evaluation" in data
    phase5 = data["phase_5_counterfactual_evaluation"]

    # Must evaluate on >= 200 real rejected applicants
    assert phase5["metadata"]["cohort_sample_size"] >= 200

    metrics = phase5["metrics_with_95_ci"]
    assert "validity_rate" in metrics
    assert "actionability_rate" in metrics
    assert "proximity_l1_normalized" in metrics
    assert "sparsity_features_changed" in metrics
    assert "latency_ms" in metrics
    assert "dice_failure_rate" in metrics
    assert "fallback_search_rate" in metrics

    # Honest disclosure of fallback rate
    assert metrics["fallback_search_rate"] > 0.0, "DiCE fallback rate must be honestly reported"
