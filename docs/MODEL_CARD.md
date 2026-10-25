# Model Card: Production XGBoost Credit Default Classifier

## 1. Model Details & Intended Use
- **Model Name:** PRODUCTION-XGBOOST-CREDIT (v2.0.0-real)
- **Model Architecture:** Gradient Boosted Decision Trees (`xgboost.XGBClassifier`) with post-hoc ISOTONIC probability calibration.
- **Intended Use:** Second-line credit decision explainability, adverse action attribution, and counterfactual remediation auditing for retail applicants.
- **Out-of-Scope Uses:** Fully autonomous automated loan rejection without human-in-the-loop underwriting review.
- **Artifact SHA-256:** `a4403a4041294ffdbd11db45fce2ad22a151d1c5b968dc433e762bcf43c76673`
- **Governing Standard:** Federal Reserve SR 11-7 (Supervisory Guidance on Model Risk Management).

---

## 2. Training & Validation Protocol
- **Dataset:** 30,000 real credit records from UCI Default of Credit Card Clients (verified SHA-256).
- **Split Protocol:** Stratified split (70% Train / 15% Validation / 15% Test). Test set touched exactly once.
- **Hyperparameter Tuning:** 5-Fold Stratified Cross-Validation on **Train split only**:
  - `n_estimators`: 100
  - `max_depth`: 4
  - `learning_rate`: 0.05
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8

---

## 3. Test Evaluation Metrics (1,000 Bootstrap 95% Confidence Intervals)
All numbers below were measured on the independent Test set ($N = 4,500$):

| Metric | Simple Logistic Regression Baseline | Production XGBoost (Calibrated) | Superiority / Status |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | `0.7451` [0.7264, 0.7650] | **`0.7742`** [0.7569, 0.7906] | **Pass** (+0.0291 over baseline) |
| **PR-AUC (Avg Precision)** | `0.5056` [0.4748, 0.5409] | **`0.5305`** [0.5016, 0.5614] | **Pass** (+0.0249) |
| **KS Statistic** | `0.3962` [0.3667, 0.4327] | **`0.4061`** [0.3761, 0.4416] | **Pass** |
| **Gini Index** | `0.4902` [0.4528, 0.5299] | **`0.5484`** [0.5138, 0.5812] | **Pass** |
| **Brier Score (Lower is better)** | `0.1421` [0.1353, 0.1494] | **`0.1365`** [0.1301, 0.1434] | **Pass** |
| **Expected Calibration Error (ECE)** | `0.0148` | **`0.0082`** | **Calibrated** |

---

## 4. Probability Calibration & Decision Threshold Justification
1. **Calibration:** Fitted on Validation set ($N = 4,500$). Brier score improved from uncalibrated `0.13647` to `0.13646`.
2. **Threshold Justification:** In retail credit risk, unmitigated default leads to loss of loan principal (False Negative), which is substantially more severe than foregone interest margin on a creditworthy applicant rejected (False Positive). We establish an explicit institutional cost ratio:
   $$C_{FN} / C_{FP} = 5.0 : 1.0$$
   Evaluating candidate cutoffs $\tau \in [0.05, 0.95]$ on validation predictions yielded:
   - **Optimal Decision Threshold ($\tau^*$):** **`0.1300`** (replaces ad-hoc 0.35 assertion)
   - **Minimized Expected Cost Per Applicant:** `0.5544`
   - **Validation Default Recall at Cutoff:** `82.61%`
   - **Validation Approval Rate:** `45.51%`

---

## 5. Model Limitations & Failure Modes
1. **Subpopulation Slices:** Performance may degrade on rare employment or high-utilization edge cases.
2. **Feature Non-Linearity:** What-if changes in credit utilization do not guarantee linear score improvements.
3. **Macroeconomic Shifts:** Model assumes historical macroeconomic stability; severe economic contraction requires model re-calibration.
