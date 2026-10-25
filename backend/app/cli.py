"""CLI runner executing Prediction -> SHAP -> Counterfactuals end-to-end on an applicant fixture."""

import argparse
import json
from pathlib import Path
from typing import Any
import pandas as pd

from backend.app.config import get_settings
from backend.app.schemas.feature_schema import FeatureSchema
from backend.app.schemas.prediction import ThresholdConfig
from backend.app.services.counterfactual import CounterfactualService
from backend.app.services.explanation import ExplanationService
from backend.app.services.prediction import ModelLoader, PredictionService


def run_pipeline(
    applicant_path: str = "data/demo/applicant_10492_synthetic.json",
    output_path: str | None = None,
) -> dict[str, Any]:
    """Run prediction -> SHAP -> counterfactuals pipeline and return verified result payload."""
    settings = get_settings()

    # Load feature schema
    schema_file = Path(settings.feature_schema_path)
    if not schema_file.is_file():
        raise FileNotFoundError(f"Feature schema not found at {schema_file}")
    with open(schema_file, "r", encoding="utf-8") as f:
        feature_schema = FeatureSchema.model_validate(json.load(f))

    # Load applicant data
    app_file = Path(applicant_path)
    if not app_file.is_file():
        raise FileNotFoundError(f"Applicant file not found at {app_file}")
    with open(app_file, "r", encoding="utf-8") as f:
        applicant_data = json.load(f)

    # Initialize Prediction Service
    loader = ModelLoader(
        model_path=settings.model_path,
        manifest_path=settings.model_manifest_path,
        feature_schema=feature_schema,
    )
    threshold_cfg = ThresholdConfig(
        value=settings.threshold_value,
        source=settings.threshold_source,
        owner=settings.threshold_owner,
    )
    prediction_service = PredictionService(
        loader=loader,
        feature_schema=feature_schema,
        threshold_config=threshold_cfg,
    )

    # 1. Prediction step
    prediction_result = prediction_service.predict(applicant_data)

    # Prepare DataFrame for Explainer
    query_df, _ = prediction_service.prepare_feature_vector(applicant_data)

    # Load background data for SHAP and DiCE
    bg_file = Path(settings.background_data_path)
    background_df = pd.read_json(bg_file)

    # 2. SHAP Explanation step
    explanation_service = ExplanationService(
        model=loader.model,
        feature_schema=feature_schema,
        background_data=background_df,
        output_space=settings.shap_output_space,
        tolerance=settings.shap_tolerance,
        top_k=settings.shap_top_k,
        contribution_cutoff=settings.shap_contribution_cutoff,
        background_id=loader.manifest.explainer_background_data_id,
    )
    explanation_result = explanation_service.explain(
        feature_df=query_df,
        prediction=prediction_result,
    )

    # Verify additivity safety before proceeding
    explanation_service.is_safe_for_narrative(explanation_result)

    # 3. Constrained Counterfactual step
    counterfactual_service = CounterfactualService(
        prediction_service=prediction_service,
        feature_schema=feature_schema,
        training_df=background_df,
        dice_method=settings.dice_method,
        target_threshold=settings.dice_target_probability_threshold,
    )
    counterfactual_result = counterfactual_service.generate_counterfactuals(
        applicant_data=applicant_data,
        total_cfs=settings.dice_total_cfs,
    )

    # Assemble verified execution envelope
    output_payload = {
        "execution_summary": {
            "mode": settings.mode,
            "synthetic_fixtures_active": settings.is_synthetic_model,
            "pipeline_stages_completed": ["Pillar_A_Prediction", "Pillar_A_SHAP", "Pillar_B_Counterfactuals"],
            "disclaimer": (
                "DEMO EXECUTION NOTICE: Applicant inputs and model weights are synthetic fixtures. "
                "Predicted probability, SHAP values, additivity residuals, and counterfactuals are "
                "computed deterministically by active software modules."
            ),
        },
        "applicant": {
            "applicant_ref": applicant_data.get("applicant_ref", "APP-DEMO"),
            "raw_inputs": applicant_data,
        },
        "prediction": prediction_result.model_dump(mode="json"),
        "explanation": explanation_result.model_dump(mode="json"),
        "counterfactuals": counterfactual_result.model_dump(mode="json"),
    }

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

    return output_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XAI-RAG Part 1 End-to-End Pipeline")
    parser.add_argument(
        "--applicant",
        default="data/demo/applicant_10492_synthetic.json",
        help="Path to applicant input JSON fixture",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write verified JSON results",
    )
    args = parser.parse_args()

    payload = run_pipeline(applicant_path=args.applicant, output_path=args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
