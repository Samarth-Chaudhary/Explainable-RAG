"""Shared test fixtures for XAI-RAG unit and integration tests."""

import json
from pathlib import Path
import pandas as pd
import pytest

from backend.app.config import Settings
from backend.app.schemas.feature_schema import FeatureSchema
from backend.app.schemas.prediction import ThresholdConfig
from backend.app.services.counterfactual import CounterfactualService
from backend.app.services.explanation import ExplanationService
from backend.app.services.prediction import ModelLoader, PredictionService


@pytest.fixture(scope="session")
def base_dir() -> Path:
    """Return repository root path."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def feature_schema(base_dir) -> FeatureSchema:
    """Load canonical feature schema."""
    schema_path = base_dir / "data" / "demo" / "feature_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        return FeatureSchema.model_validate(json.load(f))


@pytest.fixture(scope="session")
def threshold_config() -> ThresholdConfig:
    """Standard test decision threshold configuration."""
    return ThresholdConfig(
        value=0.35,
        source="Credit Committee Policy Manual v2.4 (Synthetic Fixture)",
        owner="Head of Credit Underwriting & Risk Governance",
    )


@pytest.fixture(scope="session")
def model_loader(base_dir, feature_schema) -> ModelLoader:
    """Load and validate synthetic demo model."""
    model_path = str(base_dir / "data" / "demo" / "model.json")
    manifest_path = str(base_dir / "data" / "demo" / "model_manifest.json")
    return ModelLoader(model_path=model_path, manifest_path=manifest_path, feature_schema=feature_schema)


@pytest.fixture(scope="session")
def prediction_service(model_loader, feature_schema, threshold_config) -> PredictionService:
    """Instantiate prediction service."""
    return PredictionService(
        loader=model_loader,
        feature_schema=feature_schema,
        threshold_config=threshold_config,
    )


@pytest.fixture(scope="session")
def background_df(base_dir) -> pd.DataFrame:
    """Load background reference dataset."""
    bg_path = base_dir / "data" / "demo" / "explainer_background.json"
    return pd.read_json(bg_path)


@pytest.fixture(scope="session")
def explanation_service(model_loader, feature_schema, background_df) -> ExplanationService:
    """Instantiate explanation service."""
    return ExplanationService(
        model=model_loader.model,
        feature_schema=feature_schema,
        background_data=background_df,
        output_space="log_odds",
        tolerance=1e-4,
        top_k=5,
        contribution_cutoff=0.05,
        background_id=model_loader.manifest.explainer_background_data_id,
    )


@pytest.fixture(scope="session")
def counterfactual_service(prediction_service, feature_schema, background_df) -> CounterfactualService:
    """Instantiate counterfactual service."""
    return CounterfactualService(
        prediction_service=prediction_service,
        feature_schema=feature_schema,
        training_df=background_df,
        dice_method="genetic",
        target_threshold=0.35,
    )


@pytest.fixture
def valid_applicant_dict() -> dict:
    """Standard valid Scenario B synthetic applicant dictionary."""
    return {
        "applicant_id": "APPLICANT-10492-SYNTHETIC",
        "applicant_ref": "APP-10492",
        "AMT_CREDIT": 1500000.0,
        "AMT_ANNUITY": 45000.0,
        "AMT_GOODS_PRICE": 1685393.0,
        "AMT_INCOME_TOTAL": 70000.0,
        "EXISTING_DEBT_OBLIGATIONS": 2600.0,
        "DAYS_EMPLOYED": -1200.0,
        "AGE": 38.0,
        "CREDIT_BUREAU_SCORE": 620.0,
        "PAST_DELINQUENCIES": 2,
    }
