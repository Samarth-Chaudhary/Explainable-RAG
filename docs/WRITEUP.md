# Explainable Credit-Risk Auditing: Technical Architecture & Model Risk Assessment

**Executive Summary & Governance Review**  
*Model Risk Management & Internal Audit Committee* | *October 2026*

---

## 1. Problem Statement & Scope

Credit underwriting models increasingly employ non-linear ensembles (e.g., gradient boosted trees) to maximize default discrimination. However, regulatory frameworks (e.g., Federal Reserve SR 11-7, Basel BCBS 75, and RBI Digital Lending Directions) impose strict governance requirements: adverse action notices must state concrete, verifiable reasons for rejection; restructuring paths must be legally feasible and operationally actionable; and model risk auditors must have tamper-evident lineage tracing.

Standard implementations suffer from four critical deficiencies:
1. **Explainability Disconnect**: SHAP values explain model margins, but fail to establish institutional policy provenance.
2. **Unconstrained Optimization**: Counterfactual algorithms suggest mathematically valid perturbations on immutable applicant attributes (e.g., changing `AGE` or `PAST_DELINQUENCIES`).
3. **Retrieval Fabrication Risk**: Uncalibrated RAG pipelines hallucinate statutory justifications when authoritative policy evidence is missing.
4. **Security Vulnerability**: Relying on unauthenticated client-supplied role headers or keyword blocklists leaves credit decisions open to authorization bypass and prompt injection.

This project delivers a defensible, real-data decision-auditing platform evaluated against real-world credit risk data and official sovereign policy corpora.

---

## 2. Real Data & Pipeline Methodology

The pipeline transitions entirely away from synthetic generation to empirical real-world credit benchmarks:

- **Primary & Fallback Data Pipeline**: Automated fetchers (`scripts/fetch_data.py`) download and verify datasets via cryptographic SHA-256 checksums:
  - UCI *Default of Credit Card Clients* (30,000 applicants, 24 features; target default prevalence: 22.12%).
  - OpenML German Credit (1,000 applicants; target default prevalence: 30.00%).
  - Home Credit Default Risk API integration with non-zero exit validation when Kaggle credentials are missing.
- **Data Preprocessing & Derivations**: Fixed stratified split (Train: 21,000; Validation: 4,500; Test: 4,500; Seed: 42). Feature schema derives real underwriting ratios (`FOIR_RATIO`, `LTV_RATIO`), converts Home Credit annual income to monthly primitives, and maps sentinel values (`DAYS_EMPLOYED=365243 -> 0`) into explicit tenure indicators.
- **Model Training & Tuning**: 5-fold cross-validation on the train split only:
  - Baseline: L2-penalized Logistic Regression ($C=10.0$).
  - Production: XGBoost Classifier (`max_depth=4`, `learning_rate=0.05`, `n_estimators=100`, `subsample=0.8`).
- **Probability Calibration & Cost-Optimal Threshold**: Isotonic calibration fitted on the validation set. Decision cutoff was optimized against a stated credit loss asymmetry ($C_{FN}/C_{FP} = 5.0$), yielding $\tau^* = 0.1300$ (cost per applicant = 0.5544, recall = 82.61%, approval rate = 45.51%), strictly replacing arbitrary default thresholds.

---

## 3. Empirical Results (Test Set Performance with 95% CIs)

All metrics are measured on the unseen Test set ($N=4,500$) with 1,000 bootstrap resamples and logged in `tests/evaluation/evaluation_results.json`:

| Metric Category | Baseline (Logistic Regression) | Production (XGBoost) | Model Gate Outcome |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | 0.7451 [0.7264, 0.7650] | **0.7742 [0.7569, 0.7906]** | **PASS** (+0.0291 over baseline) |
| **PR-AUC** | 0.5056 [0.4727, 0.5401] | **0.5305 [0.4998, 0.5658]** | **PASS** (+0.0249 over baseline) |
| **Brier Score** | 0.1421 [0.1368, 0.1477] | **0.1365 [0.1314, 0.1420]** | **PASS** (Lower loss) |
| **Expected Calibration Error** | 0.0184 | **0.0082** | Post-calibration isotonic scaling |

---

## 4. Fairness, Parity & Protected Attribute Audit

Auditing was conducted across `AGE` (threshold 30), `SEX`, `EDUCATION`, and `MARRIAGE` under 1,000 bootstrap resamples:

- **Age Parity (<30 vs $\ge 30$)**:
  - Demographic Parity Difference (DPD): **0.0477 [0.0170, 0.0785]**
  - Disparate Impact Ratio (DIR): **0.8861 [0.8146, 0.9586]**
  - Equal Opportunity Difference (EOD): **0.0358 [0.0031, 0.0795]**
- **Sex Parity (Male vs Female)**:
  - DPD: **0.0263 [0.0020, 0.0547]**; DIR: **0.9367 [0.8703, 0.9950]**; EOD: **0.0290 [0.0015, 0.0706]**
- **Model Retraining Without `AGE`**:
  - Test ROC-AUC changed from 0.7759 to **0.7762** (+0.0003).
  - Age DPD worsened slightly from 0.0477 to **0.0494** (+0.0017) due to proxy correlation via financial accumulation.
- **Governance Decision**: `AGE` is retained as a frozen input for repayment capacity evaluation, but strictly marked immutable in counterfactual search.

---

## 5. Explanation & Counterfactual Validation

- **SHAP vs Permutation Importance**: Spearman rank correlation $\rho = \mathbf{0.9833}$ ($p = 1.94 \times 10^{-6}$).
- **Top-5 Feature Stability**:
  - Bootstrap resampling ($B=100$): Mean Jaccard = **1.0000**.
  - Input perturbation ($\epsilon=0.01$): Mean Jaccard = **0.8538 [0.7778, 1.0000]**, Mean Spearman $\rho = \mathbf{0.7895}$.
- **Faithfulness**:
  - Area Under Deletion Curve (AUDC): SHAP **0.1313** vs Random **0.0803** (attribution removes true risk drivers first).
  - Area Under Insertion Curve (AUIC): SHAP **0.1340** vs Random **0.0797**.
- **Counterfactuals on 200 Real Rejected Applicants**:
  - Real Validity Rate: **17.67% [12.83%, 22.50%]** (DiCE optimization struggled with severe credit distress).
  - Feasible Applicant Rate: **20.50% [15.50%, 26.00%]**.
  - DiCE Failure Rate: **38.00%** (76 / 200 applicants triggered domain fallback search).
  - Fallback Search Rate: **38.50%** (77 / 200 applicants required deterministic heuristic optimization).
  - Immutability Enforcement: 100% enforced; attempting to mutate `AGE`, `CREDIT_BUREAU_SCORE`, or `PAST_DELINQUENCIES` fails validation.

---

## 6. Sovereign Policy Retrieval & Security Controls

- **Authoritative Corpus Ingested**: Fed SR 11-7, Basel BCBS 75, RBI Digital Lending Directions (2025), and RBI Fair Practices Code (total 80 chunks).
- **Retrieval Benchmark Status**: 64 candidate queries (320 query-clause pairs) drafted to `tests/evaluation/to_label.csv`. Marked **BLOCKED PENDING HUMAN LABELS** per model risk rules; synthetic metrics purged.
- **Citation Faithfulness**:
  - Real reports audit: 150 / 150 citations verified via exact Unicode NFKC match (**100.0% pass rate**).
  - Adversarial tampering suite: 20 / 20 adversarial modifications caught (**100.0% catch rate**).
- **Authentication**: Client-supplied `X-User-Role` headers eliminated; replaced with PBKDF2-HMAC-SHA256 password hashing and HS256 JWTs with signed server-side role claims.
- **Prompt Injection Defense**: Multi-layered defense (Unicode NFKC, homoglyph mapping, zero-width stripping, nonced passive XML wrapping) tested against 25 adversarial injection variants across 7 languages: **100.0% caught (25/25)**, 0 false positives.

---

## 7. Model Risk Reviewer Limitations & Failures

1. **Low Counterfactual Feasibility**: On real borrowers with severe credit distress, counterfactual validity is only 17.67%. In a production banking environment, DiCE must not be presented as a turnkey automated restructuring tool.
2. **Retrieval Evaluation Uncertified**: Without human gold labels for the 64 sovereign queries, retrieval Recall@k cannot be certified.
3. **Disparate Impact Exists**: Neither the baseline nor XGBoost achieves perfect demographic parity (Age DPD = 0.0477, Sex DPD = 0.0263). The model cannot be certified as "fair".
