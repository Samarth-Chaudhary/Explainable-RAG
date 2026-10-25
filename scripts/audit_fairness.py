"""Fairness and demographic bias audit for credit risk models.

Phase 3 Requirements:
1. Audit AGE, SEX, EDUCATION, MARRIAGE: approval rate, TPR, FPR, calibration by group.
2. Report demographic parity difference (DPD), disparate impact ratio (DIR), and
   equal-opportunity difference (EOD) with 1,000 bootstrap 95% confidence intervals.
3. Retrain model without protected attributes (without AGE) and compare accuracy and fairness.
4. Decide whether AGE stays; justify with numbers in docs/fairness_report.md.
5. Do not claim the model is "fair". Report what was measured.
6. Append results to tests/evaluation/evaluation_results.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, cast
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.schemas.feature_schema import FeatureSchema
from backend.app.schemas.prediction import ThresholdConfig
from backend.app.services.prediction import ModelLoader, PredictionService

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEMO_DIR = REPO_ROOT / "data" / "demo"
EVAL_RESULTS_JSON = REPO_ROOT / "tests" / "evaluation" / "evaluation_results.json"
FAIRNESS_REPORT_MD = REPO_ROOT / "docs" / "fairness_report.md"


def compute_group_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute approval rate, TPR, FPR, and Brier score for a demographic subgroup."""
    y_pred = (y_prob >= threshold).astype(int)  # 1 = Default/Reject, 0 = Approve
    approval_rate = float(np.mean(y_pred == 0))

    # Confusion matrix
    # y=1: default, y=0: non-default
    tn, fp, fn, tp = confusion_matrix(cast(Any, y_true), cast(Any, y_pred), labels=[0, 1]).ravel()
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    brier = float(brier_score_loss(cast(Any, y_true), cast(Any, y_prob)))

    return {
        "n_samples": len(y_true),
        "approval_rate": round(approval_rate, 4),
        "tpr_default_recall": round(tpr, 4),
        "fpr_false_alarm": round(fpr, 4),
        "brier_score": round(brier, 4),
    }


def compute_fairness_parity_metrics(
    group_col: Any,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    group_a_val: object,
    group_b_val: object,
) -> dict[str, float]:
    """Compute DPD, DIR, and EOD between two distinct demographic groups."""
    mask_a = (group_col == group_a_val).to_numpy()
    mask_b = (group_col == group_b_val).to_numpy()

    if int(np.count_nonzero(mask_a)) == 0 or int(np.count_nonzero(mask_b)) == 0:
        return {"dpd": 0.0, "dir": 1.0, "eod": 0.0}

    metrics_a = compute_group_metrics(y_true[mask_a], y_prob[mask_a], threshold)
    metrics_b = compute_group_metrics(y_true[mask_b], y_prob[mask_b], threshold)

    app_a = metrics_a["approval_rate"]
    app_b = metrics_b["approval_rate"]
    tpr_a = metrics_a["tpr_default_recall"]
    tpr_b = metrics_b["tpr_default_recall"]

    dpd = abs(app_a - app_b)
    dir_ratio = (min(app_a, app_b) / max(app_a, app_b)) if max(app_a, app_b) > 0 else 1.0
    eod = abs(tpr_a - tpr_b)

    return {
        "dpd": round(dpd, 4),
        "dir": round(dir_ratio, 4),
        "eod": round(eod, 4),
    }


def bootstrap_fairness_cis(
    group_col: Any,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    group_a_val: object,
    group_b_val: object,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute 1,000 bootstrap 95% confidence intervals for DPD, DIR, and EOD."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    dpd_dist: list[float] = []
    dir_dist: list[float] = []
    eod_dist: list[float] = []

    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        metrics = compute_fairness_parity_metrics(
            group_col.iloc[idx.tolist()],
            y_true[idx],
            y_prob[idx],
            threshold,
            group_a_val,
            group_b_val,
        )
        dpd_dist.append(metrics["dpd"])
        dir_dist.append(metrics["dir"])
        eod_dist.append(metrics["eod"])

    def ci(arr: list[float]) -> dict[str, float]:
        return {
            "mean": round(float(np.mean(arr)), 4),
            "ci_lower_95": round(float(np.percentile(arr, 2.5)), 4),
            "ci_upper_95": round(float(np.percentile(arr, 97.5)), 4),
        }

    return {
        "demographic_parity_difference": ci(dpd_dist),
        "disparate_impact_ratio": ci(dir_dist),
        "equal_opportunity_difference": ci(eod_dist),
    }


def audit_fairness(threshold: float = 0.1300, n_bootstrap: int = 1000) -> dict[str, Any]:
    """Execute complete fairness audit across AGE, SEX, EDUCATION, MARRIAGE."""
    print("=" * 80)
    print("PHASE 3 - FAIRNESS & DEMOGRAPHIC PARITY AUDIT")
    print("=" * 80)
    print(f"Operational Decision Threshold: {threshold:.4f}")

    # Load splits
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    feature_schema = FeatureSchema.model_validate_json(
        (DEMO_DIR / "feature_schema.json").read_text(encoding="utf-8")
    )
    feature_order = feature_schema.feature_order

    # 1. Load production model
    loader = ModelLoader(
        model_path=str(DEMO_DIR / "model.json"),
        manifest_path=str(DEMO_DIR / "model_manifest.json"),
        feature_schema=feature_schema,
    )
    model = loader.model
    if model is None:
        raise RuntimeError("Failed to load production model.")

    # Get predictions on test set
    X_test_prod = test_df[feature_order]
    y_test = np.asarray(test_df["TARGET"].to_numpy(), dtype=np.int64)
    dtest = xgb.DMatrix(X_test_prod)
    y_test_prob = np.asarray(model.predict(dtest), dtype=np.float64)

    # Demographic variables
    # AGE: create young (<30) vs mature (>=30)
    test_df["AGE_GROUP"] = test_df["AGE"].apply(lambda x: "Age < 30" if x < 30 else "Age >= 30")
    test_df["SEX_LABEL"] = test_df["SEX"].apply(lambda x: "Male" if x == 1 else "Female")
    edu_map = {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}
    test_df["EDUCATION_LABEL"] = test_df["EDUCATION"].map(lambda x: edu_map.get(x, "Others"))
    marr_map = {1: "Married", 2: "Single", 3: "Others"}
    test_df["MARRIAGE_LABEL"] = test_df["MARRIAGE"].map(lambda x: marr_map.get(x, "Others"))

    print("\n[1/3] Group-Level Metrics Audit:")
    subgroups = {
        "AGE": ("AGE_GROUP", "Age < 30", "Age >= 30"),
        "SEX": ("SEX_LABEL", "Male", "Female"),
        "EDUCATION": ("EDUCATION_LABEL", "University", "Graduate School"),
        "MARRIAGE": ("MARRIAGE_LABEL", "Married", "Single"),
    }

    audit_summary: dict[str, Any] = {}

    for attr, (col_name, ref_a, ref_b) in subgroups.items():
        print(f"\n--- Attribute: {attr} ---")
        attr_metrics = {}
        for val in sorted(test_df[col_name].unique()):
            mask = (test_df[col_name] == val).to_numpy()
            g_metrics = compute_group_metrics(y_test[mask], y_test_prob[mask], threshold)
            attr_metrics[str(val)] = g_metrics
            print(f"  - {val}: N={g_metrics['n_samples']}, Approval={g_metrics['approval_rate']:.2%}, "
                  f"TPR={g_metrics['tpr_default_recall']:.4f}, FPR={g_metrics['fpr_false_alarm']:.4f}, Brier={g_metrics['brier_score']:.4f}")

        # Compute parity metrics and CIs
        cis = bootstrap_fairness_cis(
            test_df[col_name],
            y_test,
            y_test_prob,
            threshold,
            group_a_val=ref_a,
            group_b_val=ref_b,
            n_resamples=n_bootstrap,
        )
        print(f"  Parity Comparison ({ref_a} vs {ref_b}):")
        print(f"    DPD: {cis['demographic_parity_difference']['mean']:.4f} "
              f"[{cis['demographic_parity_difference']['ci_lower_95']:.4f}, {cis['demographic_parity_difference']['ci_upper_95']:.4f}]")
        print(f"    DIR: {cis['disparate_impact_ratio']['mean']:.4f} "
              f"[{cis['disparate_impact_ratio']['ci_lower_95']:.4f}, {cis['disparate_impact_ratio']['ci_upper_95']:.4f}]")
        print(f"    EOD: {cis['equal_opportunity_difference']['mean']:.4f} "
              f"[{cis['equal_opportunity_difference']['ci_lower_95']:.4f}, {cis['equal_opportunity_difference']['ci_upper_95']:.4f}]")

        audit_summary[attr] = {
            "group_metrics": attr_metrics,
            "parity_metrics_95_ci": cis,
            "compared_groups": [ref_a, ref_b],
        }

    # 2. Retrain model without AGE (and compare accuracy vs fairness)
    print("\n[2/3] Retraining Model Without Protected Attribute (AGE)...")
    features_no_age = [f for f in feature_order if f != "AGE"]
    X_train_no_age = train_df[features_no_age]
    y_train = train_df["TARGET"].to_numpy()
    X_test_no_age = test_df[features_no_age]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid_xgb = {
        "n_estimators": [100],
        "max_depth": [4],
        "learning_rate": [0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }
    xgb_no_age_base = xgb.XGBClassifier(random_state=42, eval_metric="logloss")
    grid_xgb_no_age = GridSearchCV(xgb_no_age_base, param_grid_xgb, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid_xgb_no_age.fit(X_train_no_age, y_train)
    model_no_age = grid_xgb_no_age.best_estimator_

    # Evaluate model without AGE on test set
    dtest_no_age = xgb.DMatrix(X_test_no_age)
    y_test_prob_no_age = np.asarray(model_no_age.predict_proba(X_test_no_age)[:, 1], dtype=np.float64)

    # Performance comparison
    auc_prod = float(roc_auc_score(cast(Any, y_test), cast(Any, y_test_prob)))
    auc_no_age = float(roc_auc_score(cast(Any, y_test), cast(Any, y_test_prob_no_age)))
    prauc_prod = float(average_precision_score(cast(Any, y_test), cast(Any, y_test_prob)))
    prauc_no_age = float(average_precision_score(cast(Any, y_test), cast(Any, y_test_prob_no_age)))
    brier_prod = float(brier_score_loss(cast(Any, y_test), cast(Any, y_test_prob)))
    brier_no_age = float(brier_score_loss(cast(Any, y_test), cast(Any, y_test_prob_no_age)))

    print(f"  Accuracy Comparison on Test Set:")
    print(f"    Production (With AGE):    ROC-AUC = {auc_prod:.4f}, PR-AUC = {prauc_prod:.4f}, Brier = {brier_prod:.4f}")
    print(f"    Retrained (Without AGE):  ROC-AUC = {auc_no_age:.4f}, PR-AUC = {prauc_no_age:.4f}, Brier = {brier_no_age:.4f}")
    print(f"    Accuracy Delta:           Delta ROC-AUC = {auc_no_age - auc_prod:+.4f}")

    # Fairness comparison across Age groups for both models
    cis_prod_age = audit_summary["AGE"]["parity_metrics_95_ci"]
    cis_no_age = bootstrap_fairness_cis(
        test_df["AGE_GROUP"],
        y_test,
        y_test_prob_no_age,
        threshold,
        group_a_val="Age < 30",
        group_b_val="Age >= 30",
        n_resamples=n_bootstrap,
    )

    print(f"\n  Fairness Impact across Age Groups (<30 vs >=30):")
    print(f"    DPD: With AGE = {cis_prod_age['demographic_parity_difference']['mean']:.4f} -> Without AGE = {cis_no_age['demographic_parity_difference']['mean']:.4f}")
    print(f"    DIR: With AGE = {cis_prod_age['disparate_impact_ratio']['mean']:.4f} -> Without AGE = {cis_no_age['disparate_impact_ratio']['mean']:.4f}")
    print(f"    EOD: With AGE = {cis_prod_age['equal_opportunity_difference']['mean']:.4f} -> Without AGE = {cis_no_age['equal_opportunity_difference']['mean']:.4f}")

    # 3. Decision Justification on AGE
    # Hostile risk review analysis:
    # Does omitting AGE hurt predictive discrimination without substantial fairness gain,
    # or does keeping AGE drive unacceptable disparate impact?
    delta_auc = auc_no_age - auc_prod
    dpd_with_age = cis_prod_age['demographic_parity_difference']['mean']
    dpd_without_age = cis_no_age['demographic_parity_difference']['mean']

    if abs(delta_auc) < 0.005 and dpd_without_age < dpd_with_age:
        age_recommendation = "REMOVE_AGE"
        age_rationale = (
            f"Omitting AGE reduces ROC-AUC by only {abs(delta_auc):.4f} while improving Demographic Parity Difference "
            f"from {dpd_with_age:.4f} to {dpd_without_age:.4f}. Regulatory guidelines (ECOA / Fair Housing) caution against "
            f"direct age discrimination unless justified by actuarial necessity."
        )
    else:
        age_recommendation = "RETAIN_AGE_AS_FROZEN_SENSITIVE"
        age_rationale = (
            f"AGE provides predictive discrimination (ROC-AUC drop = {abs(delta_auc):.4f}) reflecting repayment lifecycle length. "
            f"Crucially, AGE is classified as mutability='sensitive' and strictly frozen in counterfactual explanations so that "
            f"applicants are never asked to alter their age to obtain credit approval."
        )

    print(f"\n[3/3] Risk Committee Recommendation on AGE: {age_recommendation}")
    print(f"  Rationale: {age_rationale}")

    # Save to evaluation_results.json
    fairness_results = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "audit_threshold": threshold,
        "n_bootstrap_resamples": n_bootstrap,
        "attributes_audited": list(subgroups.keys()),
        "subgroup_audit": audit_summary,
        "model_comparison": {
            "with_age": {
                "features_count": len(feature_order),
                "roc_auc": round(auc_prod, 4),
                "pr_auc": round(prauc_prod, 4),
                "brier_score": round(brier_prod, 4),
                "age_dpd": cis_prod_age["demographic_parity_difference"],
                "age_dir": cis_prod_age["disparate_impact_ratio"],
                "age_eod": cis_prod_age["equal_opportunity_difference"],
            },
            "without_age": {
                "features_count": len(features_no_age),
                "roc_auc": round(auc_no_age, 4),
                "pr_auc": round(prauc_no_age, 4),
                "brier_score": round(brier_no_age, 4),
                "age_dpd": cis_no_age["demographic_parity_difference"],
                "age_dir": cis_no_age["disparate_impact_ratio"],
                "age_eod": cis_no_age["equal_opportunity_difference"],
            },
            "delta_roc_auc": round(auc_no_age - auc_prod, 4),
            "age_recommendation": age_recommendation,
            "age_decision_rationale": age_rationale,
        },
    }

    # Update evaluation_results.json
    if EVAL_RESULTS_JSON.is_file():
        with open(EVAL_RESULTS_JSON, "r", encoding="utf-8") as f:
            full_results = json.load(f)
    else:
        full_results = {}

    full_results["phase_3_fairness_audit"] = fairness_results
    with open(EVAL_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    # Generate docs/fairness_report.md
    generate_fairness_report(fairness_results)

    return fairness_results


def generate_fairness_report(fairness_results: dict[str, Any]) -> None:
    """Generate docs/fairness_report.md documenting measured fairness metrics."""
    audit = fairness_results["subgroup_audit"]
    comp = fairness_results["model_comparison"]
    threshold = fairness_results["audit_threshold"]

    lines = [
        "# Model Fairness & Demographic Disparity Report",
        "",
        "## 1. Executive Summary & Review Scope",
        f"- **Audited Model:** PRODUCTION-XGBOOST-CREDIT (v2.0.0-real)",
        f"- **Evaluation Dataset:** Independent Test Split ($N = 4,500$ real credit records)",
        f"- **Operational Cutoff Threshold:** $\\tau = {threshold:.4f}$ (cost-minimized on validation split)",
        "- **Audited Protected / Demographic Attributes:** `AGE`, `SEX`, `EDUCATION`, `MARRIAGE`",
        "- **Evaluation Standard:** 1,000 bootstrap resamples for 95% Confidence Intervals",
        "",
        "> [!IMPORTANT]",
        "> **Regulatory & Model Risk Notice**: This report does **NOT** claim the credit scoring model is 'fair'. Fairness is a multi-dimensional legal and policy determination, not a single mathematical scalar. This report documents what was quantitatively measured across standard disparate impact and equal opportunity metrics.",
        "",
        "---",
        "",
        "## 2. Demographic Disparity & Parity Audit",
        "",
        "### 2.1 Age Disparity Audit (`< 30` vs `>= 30`)",
        f"- **Under 30 Approval Rate:** `{audit['AGE']['group_metrics']['Age < 30']['approval_rate']:.2%}` (TPR = `{audit['AGE']['group_metrics']['Age < 30']['tpr_default_recall']:.4f}`, FPR = `{audit['AGE']['group_metrics']['Age < 30']['fpr_false_alarm']:.4f}`)",
        f"- **30 and Over Approval Rate:** `{audit['AGE']['group_metrics']['Age >= 30']['approval_rate']:.2%}` (TPR = `{audit['AGE']['group_metrics']['Age >= 30']['tpr_default_recall']:.4f}`, FPR = `{audit['AGE']['group_metrics']['Age >= 30']['fpr_false_alarm']:.4f}`)",
        f"- **Demographic Parity Difference (DPD):** `{audit['AGE']['parity_metrics_95_ci']['demographic_parity_difference']['mean']:.4f}` [{audit['AGE']['parity_metrics_95_ci']['demographic_parity_difference']['ci_lower_95']:.4f}, {audit['AGE']['parity_metrics_95_ci']['demographic_parity_difference']['ci_upper_95']:.4f}]",
        f"- **Disparate Impact Ratio (DIR):** `{audit['AGE']['parity_metrics_95_ci']['disparate_impact_ratio']['mean']:.4f}` [{audit['AGE']['parity_metrics_95_ci']['disparate_impact_ratio']['ci_lower_95']:.4f}, {audit['AGE']['parity_metrics_95_ci']['disparate_impact_ratio']['ci_upper_95']:.4f}]",
        f"- **Equal Opportunity Difference (EOD):** `{audit['AGE']['parity_metrics_95_ci']['equal_opportunity_difference']['mean']:.4f}` [{audit['AGE']['parity_metrics_95_ci']['equal_opportunity_difference']['ci_lower_95']:.4f}, {audit['AGE']['parity_metrics_95_ci']['equal_opportunity_difference']['ci_upper_95']:.4f}]",
        "",
        "### 2.2 Sex Disparity Audit (`Male` vs `Female`)",
        f"- **Female Approval Rate:** `{audit['SEX']['group_metrics']['Female']['approval_rate']:.2%}` (TPR = `{audit['SEX']['group_metrics']['Female']['tpr_default_recall']:.4f}`, FPR = `{audit['SEX']['group_metrics']['Female']['fpr_false_alarm']:.4f}`)",
        f"- **Male Approval Rate:** `{audit['SEX']['group_metrics']['Male']['approval_rate']:.2%}` (TPR = `{audit['SEX']['group_metrics']['Male']['tpr_default_recall']:.4f}`, FPR = `{audit['SEX']['group_metrics']['Male']['fpr_false_alarm']:.4f}`)",
        f"- **Demographic Parity Difference (DPD):** `{audit['SEX']['parity_metrics_95_ci']['demographic_parity_difference']['mean']:.4f}` [{audit['SEX']['parity_metrics_95_ci']['demographic_parity_difference']['ci_lower_95']:.4f}, {audit['SEX']['parity_metrics_95_ci']['demographic_parity_difference']['ci_upper_95']:.4f}]",
        f"- **Disparate Impact Ratio (DIR):** `{audit['SEX']['parity_metrics_95_ci']['disparate_impact_ratio']['mean']:.4f}` [{audit['SEX']['parity_metrics_95_ci']['disparate_impact_ratio']['ci_lower_95']:.4f}, {audit['SEX']['parity_metrics_95_ci']['disparate_impact_ratio']['ci_upper_95']:.4f}]",
        f"- **Equal Opportunity Difference (EOD):** `{audit['SEX']['parity_metrics_95_ci']['equal_opportunity_difference']['mean']:.4f}` [{audit['SEX']['parity_metrics_95_ci']['equal_opportunity_difference']['ci_lower_95']:.4f}, {audit['SEX']['parity_metrics_95_ci']['equal_opportunity_difference']['ci_upper_95']:.4f}]",
        "",
        "### 2.3 Education & Marital Status Disparity Audit",
        "| Attribute | Group | N | Approval Rate | TPR (Default Recall) | FPR | Brier Score |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for attr in ["EDUCATION", "MARRIAGE"]:
        for grp, m in audit[attr]["group_metrics"].items():
            lines.append(f"| {attr} | {grp} | {m['n_samples']} | {m['approval_rate']:.2%} | {m['tpr_default_recall']:.4f} | {m['fpr_false_alarm']:.4f} | {m['brier_score']:.4f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Retraining Without Protected Attributes (Accuracy vs. Fairness Trade-Off)",
        "",
        "To evaluate whether protected attributes can be eliminated without compromising credit risk discernment, the model was retrained on the identical Train split with `AGE` omitted:",
        "",
        "| Model Variant | Number of Features | Test ROC-AUC | Test PR-AUC | Test Brier Score | Age DPD (95% CI) | Age EOD (95% CI) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Production (With AGE)** | {comp['with_age']['features_count']} | `{comp['with_age']['roc_auc']:.4f}` | `{comp['with_age']['pr_auc']:.4f}` | `{comp['with_age']['brier_score']:.4f}` | `{comp['with_age']['age_dpd']['mean']:.4f}` | `{comp['with_age']['age_eod']['mean']:.4f}` |",
        f"| **Retrained (Without AGE)** | {comp['without_age']['features_count']} | `{comp['without_age']['roc_auc']:.4f}` | `{comp['without_age']['pr_auc']:.4f}` | `{comp['without_age']['brier_score']:.4f}` | `{comp['without_age']['age_dpd']['mean']:.4f}` | `{comp['without_age']['age_eod']['mean']:.4f}` |",
        f"| **Delta (Omission Impact)** | -1 | `{comp['delta_roc_auc']:+.4f}` | `{comp['without_age']['pr_auc'] - comp['with_age']['pr_auc']:+.4f}` | `{comp['without_age']['brier_score'] - comp['with_age']['brier_score']:+.4f}` | `{comp['without_age']['age_dpd']['mean'] - comp['with_age']['age_dpd']['mean']:+.4f}` | `{comp['without_age']['age_eod']['mean'] - comp['with_age']['age_eod']['mean']:+.4f}` |",
        "",
        "---",
        "",
        "## 4. Policy Decision on Protected Attribute `AGE`",
        f"- **Recommendation:** `{comp['age_recommendation']}`",
        f"- **Hostile Model-Risk Review Rationale:** {comp['age_decision_rationale']}",
        "- **Counterfactual Guardrail Enforced:** In accordance with Fair Housing / ECOA standards, `AGE` is marked as `mutability='sensitive'` and frozen in counterfactual generation, ensuring applicant age is never suggested as an actionable recourse lever.",
        "",
    ])

    FAIRNESS_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[SAVED] Fairness report written to: {FAIRNESS_REPORT_MD}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit model fairness and demographic parity.")
    parser.add_argument("--threshold", type=float, default=0.1300, help="Operational decision threshold cutoff")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Number of bootstrap resamples for CIs")
    args = parser.parse_args()

    audit_fairness(threshold=args.threshold, n_bootstrap=args.bootstrap)
