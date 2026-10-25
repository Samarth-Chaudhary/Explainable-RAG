"""Unit tests for deterministic prediction, threshold edge cases, and integrity verification."""

import copy
import pytest
from backend.app.schemas.feature_schema import FeatureSchema
from backend.app.schemas.prediction import ThresholdConfig
from backend.app.services.prediction import (
    ModelIntegrityError,
    ModelLoader,
    PredictionService,
    SchemaMismatchError,
)


def test_prediction_stability_and_bounds(prediction_service: PredictionService, valid_applicant_dict: dict):
    """Prediction output must be invariant across multiple runs and probability must be in [0, 1]."""
    res1 = prediction_service.predict(valid_applicant_dict)
    res2 = prediction_service.predict(valid_applicant_dict)

    assert res1.default_probability == res2.default_probability
    assert res1.raw_output == res2.raw_output
    assert 0.0 <= res1.default_probability <= 1.0
    assert res1.feature_order_hash == res2.feature_order_hash


def test_derived_features_computed_correctly(prediction_service: PredictionService, valid_applicant_dict: dict):
    """Derived ratios (FOIR_RATIO and LTV_RATIO) must be recomputed from primitives accurately."""
    _, combined = prediction_service.prepare_feature_vector(valid_applicant_dict)

    # FOIR = (AMT_ANNUITY + EXISTING_DEBT_OBLIGATIONS) / AMT_INCOME_TOTAL
    # (45000 + 2600) / 70000 = 47600 / 70000 = 0.68
    expected_foir = round((45000.0 + 2600.0) / 70000.0, 4)
    assert combined["FOIR_RATIO"] == expected_foir

    # LTV = AMT_CREDIT / AMT_GOODS_PRICE
    # 1500000 / 1685393 = 0.889999... -> 0.89
    expected_ltv = round(1500000.0 / 1685393.0, 4)
    assert combined["LTV_RATIO"] == expected_ltv


def test_threshold_comparison_edge_cases(prediction_service: PredictionService, valid_applicant_dict: dict):
    """Verify decision threshold comparison edge cases (==, >, <).

    Policy rule: default_probability >= threshold -> REJECT_HIGH_RISK
                 default_probability < threshold  -> APPROVE_LOW_RISK
    """
    res = prediction_service.predict(valid_applicant_dict)
    actual_prob = res.default_probability

    # Case 1: Threshold set exactly equal to predicted probability
    exact_threshold = ThresholdConfig(
        value=actual_prob,
        source="Test Policy (Exact Edge Case)",
        owner="Chief Risk Officer",
    )
    svc_exact = PredictionService(
        loader=prediction_service.loader,
        feature_schema=prediction_service.feature_schema,
        threshold_config=exact_threshold,
    )
    res_exact = svc_exact.predict(valid_applicant_dict)
    assert res_exact.decision_label == "REJECT_HIGH_RISK"  # >= rule rejects when equal

    # Case 2: Threshold set just above predicted probability (+0.001)
    above_threshold = ThresholdConfig(
        value=min(1.0, actual_prob + 0.001),
        source="Test Policy (Just Above)",
        owner="Chief Risk Officer",
    )
    svc_above = PredictionService(
        loader=prediction_service.loader,
        feature_schema=prediction_service.feature_schema,
        threshold_config=above_threshold,
    )
    res_above = svc_above.predict(valid_applicant_dict)
    assert res_above.decision_label == "APPROVE_LOW_RISK"

    # Case 3: Threshold set just below predicted probability (-0.001)
    below_threshold = ThresholdConfig(
        value=max(0.0, actual_prob - 0.001),
        source="Test Policy (Just Below)",
        owner="Chief Risk Officer",
    )
    svc_below = PredictionService(
        loader=prediction_service.loader,
        feature_schema=prediction_service.feature_schema,
        threshold_config=below_threshold,
    )
    res_below = svc_below.predict(valid_applicant_dict)
    assert res_below.decision_label == "REJECT_HIGH_RISK"


def test_model_loader_detects_hash_tampering(tmp_path, feature_schema: FeatureSchema):
    """Model loader must fail with ModelIntegrityError if artifact SHA-256 is tampered."""
    fake_model = tmp_path / "model.json"
    fake_model.write_text('{"mock": "corrupted"}', encoding="utf-8")

    fake_manifest = tmp_path / "model_manifest.json"
    manifest_data = {
        "name": "TAMPERED-XGB",
        "version": "1.0",
        "artifact_path": str(fake_model),
        "artifact_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "feature_schema_hash": feature_schema.compute_feature_order_hash(),
        "training_dataset_id": "test",
        "explainer_background_data_id": "test",
        "calibration_status": "none",
        "created_timestamp": "2026-10-19T00:00:00Z",
        "owner": "Risk Ops",
        "notes": "Tampered test",
    }
    fake_manifest.write_text(copy.deepcopy(str(manifest_data).replace("'", '"')), encoding="utf-8")

    with pytest.raises(ModelIntegrityError):
        ModelLoader(
            model_path=str(fake_model),
            manifest_path=str(fake_manifest),
            feature_schema=feature_schema,
        )
