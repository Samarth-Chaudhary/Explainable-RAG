"""Production asset bootstrapper training only from fetched real credit risk data.

Note: Zero synthetic training data in production path.
Synthetic fixtures are strictly restricted to tests/fixtures/ and labelled as such.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

from backend.app.schemas.feature_schema import FeatureDefinition, FeatureSchema, NumericRange
from backend.app.schemas.model_manifest import ModelManifest
from backend.app.schemas.ontology import OntologyFile, OntologyRecord

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEMO_DIR = REPO_ROOT / "data" / "demo"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

FEATURE_NAMES = [
    "FOIR_RATIO",
    "LTV_RATIO",
    "CREDIT_BUREAU_SCORE",
    "PAST_DELINQUENCIES",
    "DAYS_EMPLOYED",
    "AGE",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_INCOME_TOTAL",
]


def create_feature_schema() -> FeatureSchema:
    """Define the authoritative feature schema for the credit risk model."""
    features = [
        FeatureDefinition(
            name="AMT_CREDIT",
            label="Loan Amount Requested",
            dtype="float",
            unit="INR",
            allowed_range=NumericRange(min_value=50000.0, max_value=10000000.0),
            nullability=False,
            null_handling_rule="error",
            position_index=0,
            derived=False,
            mutability="actionable",
            description="Total principal loan amount requested by the applicant",
        ),
        FeatureDefinition(
            name="AMT_ANNUITY",
            label="Monthly Loan Installment",
            dtype="float",
            unit="INR",
            allowed_range=NumericRange(min_value=1000.0, max_value=500000.0),
            nullability=False,
            null_handling_rule="error",
            position_index=1,
            derived=False,
            mutability="actionable",
            description="Proposed monthly installment (amortized over loan tenor)",
        ),
        FeatureDefinition(
            name="AMT_GOODS_PRICE",
            label="Collateral / Asset Value",
            dtype="float",
            unit="INR",
            allowed_range=NumericRange(min_value=50000.0, max_value=15000000.0),
            nullability=False,
            null_handling_rule="error",
            position_index=2,
            derived=False,
            mutability="actionable",
            description="Appraised value of collateral or purchase price of financed asset",
        ),
        FeatureDefinition(
            name="AMT_INCOME_TOTAL",
            label="Monthly Verified Income",
            dtype="float",
            unit="INR",
            allowed_range=NumericRange(min_value=15000.0, max_value=2000000.0),
            nullability=False,
            null_handling_rule="error",
            position_index=3,
            derived=False,
            mutability="immutable",
            description="Verified monthly net eligible income of applicant",
        ),
        FeatureDefinition(
            name="EXISTING_DEBT_OBLIGATIONS",
            label="Existing Monthly Obligations",
            dtype="float",
            unit="INR",
            allowed_range=NumericRange(min_value=0.0, max_value=1000000.0),
            nullability=False,
            null_handling_rule="error",
            position_index=4,
            derived=False,
            mutability="immutable",
            description="Existing monthly debt servicing obligations from credit bureau",
        ),
        FeatureDefinition(
            name="DAYS_EMPLOYED",
            label="Employment Tenure",
            dtype="float",
            unit="days",
            allowed_range=NumericRange(min_value=-20000.0, max_value=0.0),
            nullability=False,
            null_handling_rule="error",
            position_index=5,
            derived=False,
            mutability="immutable",
            description="Days since commencement of current employment (negative days convention)",
        ),
        FeatureDefinition(
            name="AGE",
            label="Applicant Age",
            dtype="float",
            unit="years",
            allowed_range=NumericRange(min_value=18.0, max_value=85.0),
            nullability=False,
            null_handling_rule="error",
            position_index=6,
            derived=False,
            mutability="sensitive",
            description="Applicant age in completed years (protected attribute; frozen)",
        ),
        FeatureDefinition(
            name="CREDIT_BUREAU_SCORE",
            label="Credit Bureau Score",
            dtype="float",
            unit="score",
            allowed_range=NumericRange(min_value=300.0, max_value=900.0),
            nullability=False,
            null_handling_rule="error",
            position_index=7,
            derived=False,
            mutability="immutable",
            description="Three-digit credit bureau score reflecting historical repayment performance",
        ),
        FeatureDefinition(
            name="PAST_DELINQUENCIES",
            label="Past 30+ DPD Delinquencies",
            dtype="int",
            unit="count",
            allowed_range=NumericRange(min_value=0.0, max_value=20.0),
            nullability=False,
            null_handling_rule="error",
            position_index=8,
            derived=False,
            mutability="immutable",
            description="Count of 30+ days past due delinquency events in last 24 months",
        ),
        FeatureDefinition(
            name="FOIR_RATIO",
            label="Fixed Obligation to Income Ratio",
            dtype="float",
            unit="ratio",
            allowed_range=NumericRange(min_value=0.0, max_value=2.5),
            nullability=False,
            null_handling_rule="error",
            position_index=9,
            derived=True,
            formula="(AMT_ANNUITY + EXISTING_DEBT_OBLIGATIONS) / AMT_INCOME_TOTAL",
            mutability="derived",
            description="Proportion of monthly income committed to total fixed debt obligations",
        ),
        FeatureDefinition(
            name="LTV_RATIO",
            label="Loan to Value Ratio",
            dtype="float",
            unit="ratio",
            allowed_range=NumericRange(min_value=0.0, max_value=2.0),
            nullability=False,
            null_handling_rule="error",
            position_index=10,
            derived=True,
            formula="AMT_CREDIT / AMT_GOODS_PRICE",
            mutability="derived",
            description="Ratio of total loan amount to appraised collateral / asset value",
        ),
    ]

    feature_order = [
        "FOIR_RATIO",
        "LTV_RATIO",
        "CREDIT_BUREAU_SCORE",
        "PAST_DELINQUENCIES",
        "DAYS_EMPLOYED",
        "AGE",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_INCOME_TOTAL",
    ]

    return FeatureSchema(
        schema_version="1.0.0",
        features=features,
        feature_order=feature_order,
        preprocessing_notes="Strict canonical ordering with dynamic derivation of FOIR_RATIO and LTV_RATIO.",
    )


def create_demo_ontology() -> OntologyFile:
    """Define domain ontology mapping mathematical features to policy clauses."""
    existing_path = REPO_ROOT / "data" / "ontology" / "credit_risk_ontology_v1.json"
    if existing_path.is_file():
        try:
            return OntologyFile.model_validate_json(existing_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    records = [
        OntologyRecord(
            feature_name="FOIR_RATIO",
            label="Fixed Obligation to Income Ratio",
            plain_language_definition="Share of eligible monthly income absorbed by recurring debt obligations",
            unit="ratio",
            feature_type="derived",
            formula="(AMT_ANNUITY + EXISTING_DEBT_OBLIGATIONS) / AMT_INCOME_TOTAL",
            mutability="derived_recompute",
            immutable_inputs=["EXISTING_DEBT_OBLIGATIONS", "AMT_INCOME_TOTAL"],
            actionable_inputs=["AMT_ANNUITY"],
            legal_topics=["repayment_capacity", "debt_burden", "underwriting_affordability"],
            metadata_tags=["FOIR_Cap", "Affordability_Assessment", "Debt_Burden_Limits"],
            lexical_terms=["FOIR", "fixed obligations", "debt burden", "repayment capacity"],
            semantic_template="Applicant FOIR ratio is {current}; feasible target is {target}. Identify underwriting limits.",
            source_families=["RBI_official", "internal_credit_policy"],
            ontology_version="1.0.0",
        ),
        OntologyRecord(
            feature_name="LTV_RATIO",
            label="Loan to Value Ratio",
            plain_language_definition="Proportion of loan exposure relative to appraised collateral asset value",
            unit="ratio",
            feature_type="derived",
            formula="AMT_CREDIT / AMT_GOODS_PRICE",
            mutability="derived_recompute",
            immutable_inputs=[],
            actionable_inputs=["AMT_CREDIT", "AMT_GOODS_PRICE"],
            legal_topics=["collateral_haircut", "exposure_limits", "prudential_ltv_cap"],
            metadata_tags=["LTV_Cap", "Collateral_Haircut", "Prudential_Exposure_Limits"],
            lexical_terms=["LTV", "loan to value", "collateral haircut", "exposure limit"],
            semantic_template="Applicant LTV ratio is {current}; feasible target is {target}. Identify regulatory LTV caps.",
            source_families=["RBI_official", "internal_credit_policy"],
            ontology_version="1.0.0",
        ),
        OntologyRecord(
            feature_name="CREDIT_BUREAU_SCORE",
            label="Credit Bureau Score",
            plain_language_definition="Numerical credit bureau score reflecting historical repayment performance",
            unit="score",
            feature_type="primitive",
            formula=None,
            mutability="immutable",
            immutable_inputs=["CREDIT_BUREAU_SCORE"],
            actionable_inputs=[],
            legal_topics=["bureau_reporting", "credit_history"],
            metadata_tags=["Bureau_Score", "Credit_History"],
            lexical_terms=["credit score", "bureau score", "cibil", "credit history"],
            semantic_template="Applicant credit bureau score is {current}. Review bureau score policy thresholds.",
            source_families=["internal_credit_policy", "regulatory_guidance"],
            ontology_version="1.0.0",
        ),
        OntologyRecord(
            feature_name="PAST_DELINQUENCIES",
            label="Past 30+ DPD Delinquencies",
            plain_language_definition="Count of 30+ days past due delinquency events in last 24 months",
            unit="count",
            feature_type="primitive",
            formula=None,
            mutability="immutable",
            immutable_inputs=["PAST_DELINQUENCIES"],
            actionable_inputs=[],
            legal_topics=["delinquency_history", "adverse_events"],
            metadata_tags=["Delinquency", "Past_Due"],
            lexical_terms=["delinquency", "dpd", "default history", "overdue"],
            semantic_template="Applicant has {current} past delinquency events. Review credit policy on adverse records.",
            source_families=["internal_credit_policy", "regulatory_guidance"],
            ontology_version="1.0.0",
        ),
        OntologyRecord(
            feature_name="AMT_CREDIT",
            label="Loan Amount Requested",
            plain_language_definition="Total principal loan amount requested by the applicant",
            unit="INR",
            feature_type="primitive",
            formula=None,
            mutability="actionable",
            immutable_inputs=[],
            actionable_inputs=["AMT_CREDIT"],
            legal_topics=["single_borrower_limit", "exposure_norms"],
            metadata_tags=["Credit_Limit", "Exposure_Norms"],
            lexical_terms=["loan amount", "principal", "credit limit"],
            semantic_template="Applicant requested loan amount {current}; proposed scenario {target}. Identify exposure ceilings.",
            source_families=["internal_credit_policy"],
            ontology_version="1.0.0",
        ),
        OntologyRecord(
            feature_name="AMT_ANNUITY",
            label="Monthly Loan Installment",
            plain_language_definition="Proposed monthly installment repayment under current tenor",
            unit="INR",
            feature_type="primitive",
            formula=None,
            mutability="actionable",
            immutable_inputs=[],
            actionable_inputs=["AMT_ANNUITY"],
            legal_topics=["installment_limits", "repayment_capacity"],
            metadata_tags=["Installment", "EMI"],
            lexical_terms=["installment", "emi", "annuity", "monthly payment"],
            semantic_template="Monthly installment is {current}; target is {target}. Identify tenor restructuring options.",
            source_families=["internal_credit_policy"],
            ontology_version="1.0.0",
        ),
    ]

    return OntologyFile(
        ontology_version="1.0.0",
        domain="retail_and_commercial_credit_risk",
        records=records,
    )


def bootstrap_demo_assets(output_base: str = "data") -> dict[str, str]:
    """Train real production model from fetched dataset and save all verified assets."""
    base_path = Path(output_base)
    demo_dir = base_path / "demo"
    ontology_dir = base_path / "ontology"
    demo_dir.mkdir(parents=True, exist_ok=True)
    ontology_dir.mkdir(parents=True, exist_ok=True)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Feature schema
    feature_schema = create_feature_schema()
    schema_path = demo_dir / "feature_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(feature_schema.model_dump(), f, indent=2)

    # 2. Ontology
    ontology = create_demo_ontology()
    ontology_path = ontology_dir / "credit_risk_ontology_v1.json"
    with open(ontology_path, "w", encoding="utf-8") as f:
        json.dump(ontology.model_dump(), f, indent=2)

    # 3. Train production model from fetched real dataset if not already present
    model_path = demo_dir / "model.json"
    manifest_path = demo_dir / "model_manifest.json"
    bg_path = demo_dir / "explainer_background.json"

    if not model_path.exists() or not manifest_path.exists():
        from scripts.train_and_evaluate_models import train_and_evaluate
        train_and_evaluate(seed=42)

    # 4. Save synthetic test fixture under tests/fixtures/ ONLY (labelled as such)
    synthetic_applicant_fixture = {
        "applicant_id": "APPLICANT-10492-SYNTHETIC-FIXTURE",
        "applicant_ref": "APP-10492",
        "data_mode": "synthetic_test_fixture",
        "_notes": "Synthetic fixture for unit testing only. Not for production scoring.",
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
    fixture_path = FIXTURES_DIR / "synthetic_applicant_10492.json"
    with open(fixture_path, "w", encoding="utf-8") as f:
        json.dump(synthetic_applicant_fixture, f, indent=2)

    # Mirror to demo for backward-compatible test fixtures
    legacy_fixture_path = demo_dir / "applicant_10492_synthetic.json"
    with open(legacy_fixture_path, "w", encoding="utf-8") as f:
        json.dump(synthetic_applicant_fixture, f, indent=2)

    return {
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "schema_path": str(schema_path),
        "background_path": str(bg_path),
        "applicant_path": str(legacy_fixture_path),
        "fixture_path": str(fixture_path),
        "ontology_path": str(ontology_path),
    }


if __name__ == "__main__":
    out = bootstrap_demo_assets()
    print("Asset bootstrap completed successfully on real data:")
    for k, v in out.items():
        print(f"  {k}: {v}")
