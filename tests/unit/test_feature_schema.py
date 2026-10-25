"""Unit tests for feature schema validation, bounds checking, and order hashing."""

import pytest
from backend.app.schemas.feature_schema import FeatureSchema
from backend.app.schemas.prediction import InputValidationError
from backend.app.services.prediction import PredictionService


def test_feature_order_hash_consistency(feature_schema: FeatureSchema):
    """Ensure feature order hash is deterministic and reproducible."""
    hash1 = feature_schema.compute_feature_order_hash()
    hash2 = feature_schema.compute_feature_order_hash()
    assert hash1 == hash2
    assert len(hash1) == 64  # Valid SHA-256 string


def test_reject_missing_required_feature(prediction_service: PredictionService, valid_applicant_dict: dict):
    """Feature schema must reject missing required primitive features with structured FieldError."""
    bad_input = dict(valid_applicant_dict)
    del bad_input["AMT_CREDIT"]

    with pytest.raises(InputValidationError) as exc_info:
        prediction_service.validate_applicant_data(bad_input)

    errors = exc_info.value.errors
    assert any(e.field == "AMT_CREDIT" and "missing" in e.issue for e in errors)


def test_reject_extra_unexpected_features(prediction_service: PredictionService, valid_applicant_dict: dict):
    """Feature schema must reject unexpected extra features to prevent silent pollution."""
    bad_input = dict(valid_applicant_dict)
    bad_input["UNEXPECTED_FIELD_XYZ"] = 999.0
    bad_input["CO_APPLICANT_FLAG"] = 1.0

    with pytest.raises(InputValidationError) as exc_info:
        prediction_service.validate_applicant_data(bad_input)

    errors = exc_info.value.errors
    offending_fields = [e.field for e in errors]
    assert "UNEXPECTED_FIELD_XYZ" in offending_fields
    assert "CO_APPLICANT_FLAG" in offending_fields


def test_reject_non_numeric_values(prediction_service: PredictionService, valid_applicant_dict: dict):
    """Feature schema must reject non-numeric strings where float/int is expected."""
    bad_input = dict(valid_applicant_dict)
    bad_input["AMT_INCOME_TOTAL"] = "NOT_A_NUMBER"

    with pytest.raises(InputValidationError) as exc_info:
        prediction_service.validate_applicant_data(bad_input)

    errors = exc_info.value.errors
    assert any(e.field == "AMT_INCOME_TOTAL" and "numeric" in e.issue for e in errors)


def test_reject_out_of_range_values(prediction_service: PredictionService, valid_applicant_dict: dict):
    """Feature schema must reject values violating specified min/max bounds."""
    bad_input = dict(valid_applicant_dict)
    bad_input["AGE"] = 12.0  # Min allowed is 18
    bad_input["AMT_CREDIT"] = 500000000.0  # Max allowed is 10,000,000

    with pytest.raises(InputValidationError) as exc_info:
        prediction_service.validate_applicant_data(bad_input)

    errors = exc_info.value.errors
    offending = {e.field: e.issue for e in errors}
    assert "AGE" in offending and "below minimum" in offending["AGE"]
    assert "AMT_CREDIT" in offending and "above maximum" in offending["AMT_CREDIT"]


def test_mutability_classifications(feature_schema: FeatureSchema):
    """Ensure mutability classes are correctly partitioned across features."""
    immutables = [f.name for f in feature_schema.get_immutable_features()]
    actionables = [f.name for f in feature_schema.get_actionable_features()]
    derived = [f.name for f in feature_schema.get_derived_features()]

    # Age, past delinquency, income, and bureau score must be immutable
    assert "AGE" in immutables
    assert "PAST_DELINQUENCIES" in immutables
    assert "AMT_INCOME_TOTAL" in immutables
    assert "CREDIT_BUREAU_SCORE" in immutables

    # Loan terms must be actionable
    assert "AMT_CREDIT" in actionables
    assert "AMT_ANNUITY" in actionables

    # Ratios must be derived
    assert "FOIR_RATIO" in derived
    assert "LTV_RATIO" in derived
