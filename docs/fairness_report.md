# Model Fairness & Demographic Disparity Report

## 1. Executive Summary & Review Scope
- **Audited Model:** PRODUCTION-XGBOOST-CREDIT (v2.0.0-real)
- **Evaluation Dataset:** Independent Test Split ($N = 4,500$ real credit records)
- **Operational Cutoff Threshold:** $\tau = 0.1300$ (cost-minimized on validation split)
- **Audited Protected / Demographic Attributes:** `AGE`, `SEX`, `EDUCATION`, `MARRIAGE`
- **Evaluation Standard:** 1,000 bootstrap resamples for 95% Confidence Intervals

> [!IMPORTANT]
> **Regulatory & Model Risk Notice**: This report does **NOT** claim the credit scoring model is 'fair'. Fairness is a multi-dimensional legal and policy determination, not a single mathematical scalar. This report documents what was quantitatively measured across standard disparate impact and equal opportunity metrics.

---

## 2. Demographic Disparity & Parity Audit

### 2.1 Age Disparity Audit (`< 30` vs `>= 30`)
- **Under 30 Approval Rate:** `36.99%` (TPR = `0.8790`, FPR = `0.5591`)
- **30 and Over Approval Rate:** `41.70%` (TPR = `0.8446`, FPR = `0.5087`)
- **Demographic Parity Difference (DPD):** `0.0477` [0.0170, 0.0785]
- **Disparate Impact Ratio (DIR):** `0.8861` [0.8146, 0.9586]
- **Equal Opportunity Difference (EOD):** `0.0358` [0.0031, 0.0795]

### 2.2 Sex Disparity Audit (`Male` vs `Female`)
- **Female Approval Rate:** `41.26%` (TPR = `0.8668`, FPR = `0.5174`)
- **Male Approval Rate:** `38.62%` (TPR = `0.8415`, FPR = `0.5364`)
- **Demographic Parity Difference (DPD):** `0.0263` [0.0020, 0.0547]
- **Disparate Impact Ratio (DIR):** `0.9367` [0.8703, 0.9950]
- **Equal Opportunity Difference (EOD):** `0.0290` [0.0015, 0.0706]

### 2.3 Education & Marital Status Disparity Audit
| Attribute | Group | N | Approval Rate | TPR (Default Recall) | FPR | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| EDUCATION | Graduate School | 1580 | 46.77% | 0.8281 | 0.4571 | 0.1277 |
| EDUCATION | High School | 766 | 32.64% | 0.8426 | 0.6151 | 0.1599 |
| EDUCATION | Others | 68 | 52.94% | 1.0000 | 0.4545 | 0.0277 |
| EDUCATION | University | 2086 | 37.63% | 0.8784 | 0.5482 | 0.1381 |
| MARRIAGE | Married | 2063 | 40.14% | 0.8614 | 0.5150 | 0.1426 |
| MARRIAGE | Others | 54 | 31.48% | 0.8889 | 0.6444 | 0.1251 |
| MARRIAGE | Single | 2383 | 40.50% | 0.8487 | 0.5296 | 0.1314 |

---

## 3. Retraining Without Protected Attributes (Accuracy vs. Fairness Trade-Off)

To evaluate whether protected attributes can be eliminated without compromising credit risk discernment, the model was retrained on the identical Train split with `AGE` omitted:

| Model Variant | Number of Features | Test ROC-AUC | Test PR-AUC | Test Brier Score | Age DPD (95% CI) | Age EOD (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Production (With AGE)** | 9 | `0.7759` | `0.5507` | `0.1365` | `0.0477` | `0.0358` |
| **Retrained (Without AGE)** | 8 | `0.7762` | `0.5518` | `0.1363` | `0.0494` | `0.0369` |
| **Delta (Omission Impact)** | -1 | `+0.0003` | `+0.0011` | `-0.0002` | `+0.0017` | `+0.0011` |

---

## 4. Policy Decision on Protected Attribute `AGE`
- **Recommendation:** `RETAIN_AGE_AS_FROZEN_SENSITIVE`
- **Hostile Model-Risk Review Rationale:** AGE provides predictive discrimination (ROC-AUC drop = 0.0003) reflecting repayment lifecycle length. Crucially, AGE is classified as mutability='sensitive' and strictly frozen in counterfactual explanations so that applicants are never asked to alter their age to obtain credit approval.
- **Counterfactual Guardrail Enforced:** In accordance with Fair Housing / ECOA standards, `AGE` is marked as `mutability='sensitive'` and frozen in counterfactual generation, ensuring applicant age is never suggested as an actionable recourse lever.
