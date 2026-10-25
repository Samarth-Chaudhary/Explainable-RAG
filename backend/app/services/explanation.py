"""SHAP explanation service with TreeExplainer, explicit output-space checks, and additivity verification."""

from typing import Any, Literal
import numpy as np
import pandas as pd
from scipy.special import expit
import shap
import xgboost as xgb

try:
    from backend.app.schemas.explanation import (
        AdditivityStatus,
        ExplanationResult,
        FeatureContributionSummary,
        OutputSpace,
    )
    from backend.app.schemas.feature_schema import FeatureSchema
    from backend.app.schemas.prediction import PredictionResult
except (ImportError, ModuleNotFoundError):
    from app.schemas.explanation import (  # type: ignore
        AdditivityStatus,
        ExplanationResult,
        FeatureContributionSummary,
        OutputSpace,
    )
    from app.schemas.feature_schema import FeatureSchema  # type: ignore
    from app.schemas.prediction import PredictionResult  # type: ignore


class CrossSpaceComparisonError(Exception):
    """Raised when comparing values from mismatched coordinate spaces (e.g. log-odds vs probability)."""
    pass


class DegradedExplanationError(Exception):
    """Raised when explanation additivity residual check fails, blocking narrative generation."""
    pass


class ExplanationService:
    """Computes mathematically verified local TreeSHAP explanations."""

    def __init__(
        self,
        model: xgb.Booster,
        feature_schema: FeatureSchema,
        background_data: pd.DataFrame,
        output_space: OutputSpace = "log_odds",
        tolerance: float = 1e-4,
        top_k: int = 5,
        contribution_cutoff: float = 0.05,
        background_id: str = "synthetic-bg-100",
    ):
        self.model = model
        self.feature_schema = feature_schema
        self.background_data = background_data
        self.output_space = output_space
        self.tolerance = tolerance
        self.top_k = top_k
        self.contribution_cutoff = contribution_cutoff
        self.background_id = background_id

        # Initialize TreeExplainer
        # In XGBoost, default tree output is margin (log-odds).
        self.explainer = shap.TreeExplainer(
            self.model,
            data=self.background_data,
            model_output="raw" if output_space == "log_odds" else "probability",
            feature_perturbation="interventional" if self.background_data is not None else "tree_path_dependent",
        )

    def verify_additivity(
        self,
        base_value: float,
        shap_values: np.ndarray,
        raw_margin: float,
        default_prob: float,
        space: OutputSpace,
    ) -> tuple[float, AdditivityStatus]:
        """Perform exact-by-construction sanity test verifying SHAP sum reconstructs target output.
        
        NOTE: TreeSHAP satisfies the local accuracy (additivity) property exactly by construction.
        This verification serves solely as a software sanity check to ensure no numerical overflow,
        link space corruption, or feature indexing drift occurred during attribution calculation.
        """
        shap_sum = float(np.sum(shap_values))
        reconstructed = base_value + shap_sum

        if space == "log_odds":
            # In log-odds space: base_value + sum(shap) should reconstruct raw_margin
            residual = abs(reconstructed - raw_margin)
            # Also check sigmoid consistency
            sigmoid_reconstructed = float(expit(reconstructed))
            prob_residual = abs(sigmoid_reconstructed - default_prob)
            max_residual = max(residual, prob_residual)
        elif space == "probability":
            # In probability space: base_value + sum(shap) should reconstruct default_prob
            max_residual = abs(reconstructed - default_prob)
        else:
            raise ValueError(f"Unknown output space: {space}")

        if max_residual <= self.tolerance:
            status: AdditivityStatus = "valid"
        elif max_residual <= self.tolerance * 50:
            status = "degraded"
        else:
            status = "invalid"

        return max_residual, status

    def assert_valid_cross_space_comparison(
        self,
        val_a: float,
        space_a: str,
        val_b: float,
        space_b: str,
    ) -> None:
        """Prevent comparing log-odds values directly to probabilities without link transformation."""
        if space_a != space_b:
            raise CrossSpaceComparisonError(
                f"Invalid cross-space comparison between '{space_a}' and '{space_b}'. "
                f"Log-odds contributions cannot be directly compared to probability metrics without transformation."
            )

    def explain(
        self,
        feature_df: pd.DataFrame,
        prediction: PredictionResult,
    ) -> ExplanationResult:
        """Generate TreeSHAP attributions and perform mathematical fidelity audit."""
        # Align DataFrame to canonical feature order
        ordered_df = feature_df[self.feature_schema.feature_order]

        # Compute SHAP values
        shap_explanation = self.explainer(ordered_df)
        raw_shap_values = shap_explanation.values[0]

        # Base value E[f(x)]
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            base_value = float(self.explainer.expected_value[0])
        else:
            base_value = float(self.explainer.expected_value)

        # Mathematical additivity check
        residual, additivity_status = self.verify_additivity(
            base_value=base_value,
            shap_values=raw_shap_values,
            raw_margin=prediction.raw_output,
            default_prob=prediction.default_probability,
            space=self.output_space,
        )

        # Collect raw feature contribution data
        raw_contributions = []
        for idx, col_name in enumerate(self.feature_schema.feature_order):
            feat_def = self.feature_schema.get_feature(col_name)
            val = float(ordered_df[col_name].iloc[0])
            phi = float(raw_shap_values[idx])

            if phi > 0:
                direction = "risk_increasing"
            elif phi < 0:
                direction = "risk_reducing"
            else:
                direction = "neutral"

            raw_contributions.append({
                "name": col_name,
                "label": feat_def.label if feat_def else col_name,
                "value": val,
                "unit": feat_def.unit if feat_def else "",
                "shap_value": round(phi, 6),
                "direction": direction,
            })

        # Sort all features by absolute contribution magnitude
        raw_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        feature_contributions: list[FeatureContributionSummary] = [
            FeatureContributionSummary(
                name=item["name"],
                label=item["label"],
                value=item["value"],
                unit=item["unit"],
                shap_value=item["shap_value"],
                direction=item["direction"],
                rank=rank_idx,
            )
            for rank_idx, item in enumerate(raw_contributions, start=1)
        ]

        # Filter top risk-increasing and top risk-reducing drivers
        top_risk_increasing = [
            f for f in feature_contributions
            if f.direction == "risk_increasing" and abs(f.shap_value) >= self.contribution_cutoff
        ][: self.top_k]

        top_risk_reducing = [
            f for f in feature_contributions
            if f.direction == "risk_reducing" and abs(f.shap_value) >= self.contribution_cutoff
        ][: self.top_k]

        explainer_config = {
            "explainer_class": "shap.TreeExplainer",
            "model_output": self.explainer.model_output,
            "feature_perturbation": self.explainer.feature_perturbation,
            "background_data_id": self.background_id,
            "shap_version": shap.__version__,
        }

        methodology_notes = (
            f"SHAP local attributions computed in '{self.output_space}' space relative to background reference "
            f"distribution '{self.background_id}'. Additivity equation verified as an exact-by-construction sanity "
            f"check with residual {residual:.2e} (status: {additivity_status}). Scientific Caveat: TreeSHAP satisfies "
            f"local accuracy by construction. SHAP measures model attribution relative to the reference background; "
            f"it does NOT establish real-world causality, institutional discrimination, or guarantee that isolated "
            f"modification of one variable will yield a specific outcome."
        )

        return ExplanationResult(
            schema_version="1.0.0",
            output_space=self.output_space,
            base_value=round(base_value, 6),
            shap_additivity_residual=round(residual, 8),
            tolerance=self.tolerance,
            additivity_status=additivity_status,
            features=feature_contributions,
            top_risk_increasing=top_risk_increasing,
            top_risk_reducing=top_risk_reducing,
            explainer_config=explainer_config,
            methodology_notes=methodology_notes,
        )

    def is_safe_for_narrative(self, result: ExplanationResult) -> bool:
        """Check if explanation passes mathematical verification to safely support narrative synthesis."""
        if result.additivity_status != "valid":
            raise DegradedExplanationError(
                f"Downstream narrative generation blocked: explanation additivity status is '{result.additivity_status}' "
                f"with residual {result.shap_additivity_residual:.2e} exceeding tolerance {result.tolerance:.2e}."
            )
        return True
