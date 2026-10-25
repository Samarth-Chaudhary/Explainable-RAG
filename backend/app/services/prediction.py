"""Deterministic prediction service with schema validation, model loader, and derived feature handling."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.special import expit
import xgboost as xgb

from backend.app.schemas.feature_schema import FeatureDefinition, FeatureSchema
from backend.app.schemas.model_manifest import ModelManifest
from backend.app.schemas.prediction import (
    FieldError,
    InputValidationError,
    ModelMetadataSummary,
    PredictionResult,
    ThresholdConfig,
)


class ModelIntegrityError(Exception):
    """Raised when model artifact integrity verification fails."""
    pass


class SchemaMismatchError(Exception):
    """Raised when model manifest feature hash does not match active feature schema."""
    pass


class ModelLoader:
    """Manages loading, integrity validation, and caching of versioned model artifacts."""

    def __init__(self, model_path: str, manifest_path: str, feature_schema: FeatureSchema):
        self.model_path = Path(model_path)
        self.manifest_path = Path(manifest_path)
        self.feature_schema = feature_schema
        self.manifest: ModelManifest | None = None
        self.model: xgb.XGBClassifier | xgb.Booster | None = None
        self._load_and_validate()

    def _compute_sha256(self, file_path: Path) -> str:
        """Calculate cryptographic SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_and_validate(self) -> None:
        """Load manifest and model artifact, verifying cryptographic hashes."""
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Model manifest not found at {self.manifest_path}")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")

        # Load and parse manifest
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        self.manifest = ModelManifest.model_validate(manifest_data)

        # Verify model artifact SHA-256
        actual_model_hash = self._compute_sha256(self.model_path)
        if actual_model_hash.lower() != self.manifest.artifact_sha256.lower():
            raise ModelIntegrityError(
                f"Model artifact SHA-256 mismatch!\n"
                f"  Manifest: {self.manifest.artifact_sha256}\n"
                f"  Actual:   {actual_model_hash}"
            )

        # Verify feature schema hash
        actual_schema_hash = self.feature_schema.compute_feature_order_hash()
        if actual_schema_hash.lower() != self.manifest.feature_schema_hash.lower():
            raise SchemaMismatchError(
                f"Feature schema hash mismatch!\n"
                f"  Manifest: {self.manifest.feature_schema_hash}\n"
                f"  Actual:   {actual_schema_hash}"
            )

        # Load XGBoost model
        # Use Booster for maximum stability and precise margin extraction
        booster = xgb.Booster()
        booster.load_model(str(self.model_path))
        self.model = booster


class PredictionService:
    """Deterministic prediction service with rigorous pre-inference validation."""

    def __init__(
        self,
        loader: ModelLoader,
        feature_schema: FeatureSchema,
        threshold_config: ThresholdConfig,
    ):
        self.loader = loader
        self.manifest = loader.manifest
        self.model = loader.model
        self.feature_schema = feature_schema
        self.threshold_config = threshold_config

    def validate_applicant_data(self, data: dict[str, Any]) -> dict[str, float]:
        """Validate input features against FeatureSchema and return cleaned primitive map.

        Rejects missing, extra, non-numeric, out-of-range, or null-violating fields.
        """
        errors: list[FieldError] = []
        validated: dict[str, float] = {}

        # 1. Check for unexpected / extra fields
        known_names = {f.name for f in self.feature_schema.features}
        # Ignore metadata fields, target ground truth, and demographic audit columns
        ignored_meta_keys = {
            "applicant_id",
            "applicant_ref",
            "id",
            "name",
            "_notes",
            "data_mode",
            "TARGET",
            "target",
            "SEX",
            "EDUCATION",
            "MARRIAGE",
        }
        input_feature_keys = {
            k for k in data.keys()
            if k not in ignored_meta_keys and not k.startswith("_")
        }

        extra_keys = input_feature_keys - known_names
        if extra_keys:
            for extra in sorted(extra_keys):
                errors.append(
                    FieldError(
                        field=extra,
                        value=data[extra],
                        issue="Unexpected feature present in input",
                        constraint="Feature is not defined in FeatureSchema",
                    )
                )

        # 2. Check each primitive feature in schema
        for feat in self.feature_schema.get_primitive_features():
            val = data.get(feat.name)

            # Check missing
            if val is None:
                if not feat.nullability:
                    errors.append(
                        FieldError(
                            field=feat.name,
                            value=None,
                            issue="Required feature is missing",
                            constraint="nullability is False",
                        )
                    )
                continue

            # Check numeric type
            if feat.dtype in ("float", "int"):
                try:
                    num_val = float(val)
                except (ValueError, TypeError):
                    errors.append(
                        FieldError(
                            field=feat.name,
                            value=val,
                            issue=f"Value cannot be parsed as numeric {feat.dtype}",
                            constraint=f"dtype must be {feat.dtype}",
                        )
                    )
                    continue

                # Check range bounds
                if feat.allowed_range:
                    if feat.allowed_range.min_value is not None and num_val < feat.allowed_range.min_value:
                        errors.append(
                            FieldError(
                                field=feat.name,
                                value=num_val,
                                issue=f"Value {num_val} is below minimum allowed bound {feat.allowed_range.min_value}",
                                constraint=f"min_value >= {feat.allowed_range.min_value}",
                            )
                        )
                    if feat.allowed_range.max_value is not None and num_val > feat.allowed_range.max_value:
                        errors.append(
                            FieldError(
                                field=feat.name,
                                value=num_val,
                                issue=f"Value {num_val} is above maximum allowed bound {feat.allowed_range.max_value}",
                                constraint=f"max_value <= {feat.allowed_range.max_value}",
                            )
                        )

                validated[feat.name] = num_val

        if errors:
            raise InputValidationError(errors=errors)

        return validated

    def compute_derived_features(self, primitives: dict[str, float]) -> dict[str, float]:
        """Calculate derived ratios deterministically from primitives.

        FOIR_RATIO = (AMT_ANNUITY + EXISTING_DEBT_OBLIGATIONS) / AMT_INCOME_TOTAL
        LTV_RATIO = AMT_CREDIT / AMT_GOODS_PRICE
        """
        derived: dict[str, float] = {}

        # FOIR_RATIO derivation
        if "AMT_ANNUITY" in primitives and "AMT_INCOME_TOTAL" in primitives:
            annuity = primitives["AMT_ANNUITY"]
            obligations = primitives.get("EXISTING_DEBT_OBLIGATIONS", 0.0)
            income = primitives["AMT_INCOME_TOTAL"]
            if income <= 0:
                raise ValueError("AMT_INCOME_TOTAL must be positive to compute FOIR_RATIO.")
            foir = (annuity + obligations) / income
            derived["FOIR_RATIO"] = round(foir, 4)

        # LTV_RATIO derivation
        if "AMT_CREDIT" in primitives and "AMT_GOODS_PRICE" in primitives:
            credit = primitives["AMT_CREDIT"]
            goods_price = primitives["AMT_GOODS_PRICE"]
            if goods_price <= 0:
                raise ValueError("AMT_GOODS_PRICE must be positive to compute LTV_RATIO.")
            ltv = credit / goods_price
            derived["LTV_RATIO"] = round(ltv, 4)

        return derived

    def prepare_feature_vector(self, data: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
        """Validate, compute derivations, and order columns in canonical sequence."""
        primitives = self.validate_applicant_data(data)
        derived = self.compute_derived_features(primitives)

        combined = {**primitives, **derived}

        # Ensure all canonical features are present
        canonical_order = self.feature_schema.feature_order
        missing_canonical = set(canonical_order) - set(combined.keys())
        if missing_canonical:
            errors = [
                FieldError(
                    field=m,
                    value=None,
                    issue="Feature missing from canonical vector",
                    constraint="All canonical model features must be resolved",
                )
                for m in sorted(missing_canonical)
            ]
            raise InputValidationError(errors=errors)

        # Construct single-row DataFrame ordered strictly by canonical sequence
        df = pd.DataFrame([{col: combined[col] for col in canonical_order}])
        return df, combined

    def predict(self, data: dict[str, Any]) -> PredictionResult:
        """Run deterministic prediction on applicant data."""
        df, features_evaluated = self.prepare_feature_vector(data)
        dmatrix = xgb.DMatrix(df)

        # 1. Raw margin / log-odds score
        raw_margin = float(self.model.predict(dmatrix, output_margin=True)[0])

        # 2. Probability of default via logistic link (sigmoid)
        # For standard binary:logistic objective in XGBoost, probability is sigmoid(raw_margin)
        default_prob = round(float(expit(raw_margin)), 6)

        # 3. Calibration status
        calibrated_prob: float | None = None
        output_notes = (
            f"Raw log-odds margin: {raw_margin:.4f}. "
            f"Default probability calculated via logistic sigmoid: {default_prob:.4f}. "
            f"Calibration status: {self.manifest.calibration_status}. "
            f"Decision threshold rule: probability >= {self.threshold_config.value:.2f} -> REJECT_HIGH_RISK."
        )

        # 4. Threshold comparison (Edge case: >= vs >)
        # Policy rule: default_probability >= threshold -> REJECT_HIGH_RISK (conservative underwriting policy)
        if default_prob >= self.threshold_config.value:
            decision_label = "REJECT_HIGH_RISK"
        else:
            decision_label = "APPROVE_LOW_RISK"

        return PredictionResult(
            schema_version="1.0.0",
            model=ModelMetadataSummary(
                name=self.manifest.name,
                version=self.manifest.version,
                artifact_sha256=self.manifest.artifact_sha256,
                calibration_status=self.manifest.calibration_status,
            ),
            raw_output=round(raw_margin, 6),
            default_probability=default_prob,
            calibrated_probability=calibrated_prob,
            threshold=self.threshold_config,
            decision_label=decision_label,
            output_notes=output_notes,
            inference_timestamp=datetime.now(timezone.utc),
            feature_order_hash=self.feature_schema.compute_feature_order_hash(),
            features_evaluated=features_evaluated,
        )
