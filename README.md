# XAI-RAG: Defensible Real-Data Credit-Risk Explainability & Decision-Auditing Platform

[![Model Risk CI Pipeline](https://github.com/Samarth-Chaudhary/Explainable-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/Samarth-Chaudhary/Explainable-RAG/actions/workflows/ci.yml)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-blue?logo=github)](https://samarth-chaudhary.github.io/Explainable-RAG/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Data Mode](https://img.shields.io/badge/data%20mode-real--data%20verified-brightgreen.svg)](data/DATA_SOURCES.md)

**XAI-RAG** is a defensible, real-data credit-risk decision-auditing and remediation platform. Designed under hostile model-risk review standards (SR 11-7, Basel BCBS 75, and RBI Digital Lending Directions), it bridges the institutional gap between machine learning default discrimination, local feature attribution, constrained restructuring feasibility, authoritative regulatory retrieval, and cryptographically verified audit trails.

Zero synthetic data is permitted in production pathways. All reported metrics derive directly from verifiable script executions logged in [`tests/evaluation/evaluation_results.json`](tests/evaluation/evaluation_results.json).

---

## Measured Performance & Empirical Benchmarks

Every number below was computed on the unseen test partition ($N=4,500$) or evaluated on 200 real rejected test applicants. None are estimated, rounded up, or asserted without code execution.

### 1. Model Discrimination & Calibration (Phase 2)
Evaluated on the unseen test split ($N=4,500$, seed 42) with 1,000 bootstrap resamples:

| Metric | Logistic Regression Baseline | Production XGBoost Classifier | Gate Outcome |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | 0.7451 [0.7264, 0.7650] | **0.7742 [0.7569, 0.7906]** | **PASS** (+0.0291 over baseline) |
| **PR-AUC** | 0.5056 [0.4727, 0.5401] | **0.5305 [0.4998, 0.5658]** | **PASS** (+0.0249 over baseline) |
| **Brier Score** | 0.1421 [0.1368, 0.1477] | **0.1365 [0.1314, 0.1420]** | **PASS** (Lower expected loss) |
| **Expected Calibration Error** | 0.0184 | **0.0082** | Isotonic calibration fit on validation split |
| **Decision Cutoff** | $\tau = 0.5000$ (Standard) | **$\tau^* = 0.1300$** | Cost-optimal ($C_{FN}/C_{FP} = 5.0$ on validation split) |

### 2. Demographic Fairness & Parity Audit (Phase 3)
Audited across `AGE`, `SEX`, `EDUCATION`, and `MARRIAGE` with 1,000 bootstrap resamples:

| Protected Group | Comparison | Demographic Parity Diff (DPD) | Disparate Impact Ratio (DIR) | Equal Opportunity Diff (EOD) |
| :--- | :--- | :--- | :--- | :--- |
| **Age** | $<30$ vs $\ge 30$ | **0.0477 [0.0170, 0.0785]** | **0.8861 [0.8146, 0.9586]** | **0.0358 [0.0031, 0.0795]** |
| **Sex** | Male vs Female | **0.0263 [0.0020, 0.0547]** | **0.9367 [0.8703, 0.9950]** | **0.0290 [0.0015, 0.0706]** |

*Retraining without `AGE` yielded test ROC-AUC = 0.7762 vs 0.7759 (+0.0003), but DPD worsened slightly to 0.0494 (+0.0017) due to proxy correlations. See [`docs/fairness_report.md`](docs/fairness_report.md).*

### 3. Explanation Validity & Faithfulness (Phase 4)
- **SHAP vs Permutation Importance Rank Correlation**: Spearman $\rho = \mathbf{0.9833}$ ($p = 1.94 \times 10^{-6}$).
- **Top-5 Feature Stability under Bootstrap** ($B=100$): Mean Jaccard = **1.0000**.
- **Top-5 Feature Stability under Input Perturbation** ($\epsilon=0.01$, $N=200$): Mean Jaccard = **0.8538 [0.7778, 1.0000]**, Mean Spearman $\rho = \mathbf{0.7895}$.
- **Faithfulness Curve (Area Under Deletion Curve)**: SHAP **0.1313** vs Random **0.0803** (+0.0510 gain).
- **Faithfulness Curve (Area Under Insertion Curve)**: SHAP **0.1340** vs Random **0.0797** (+0.0543 gain).
- **TreeSHAP Additivity Sanity Residual**: $|f(x) - \sum \phi_i| \le \mathbf{1.80 \times 10^{-6}}$ (exact by construction sanity test).

### 4. Counterfactual Feasibility on 200 Real Test Rejections (Phase 5)
Evaluated across 200 real rejected test applicants under strict immutability (`AGE`, `CREDIT_BUREAU_SCORE`, `PAST_DELINQUENCIES` frozen):

- **Validity Rate**: **17.67% [12.83%, 22.50%]** (102 / 600 candidate perturbations reached approval target).
- **Actionability Rate**: **17.67% [12.83%, 22.50%]** (100% of valid candidates respected immutability and domain bounds).
- **Feasible Applicant Rate**: **20.50% [15.50%, 26.00%]** (41 / 200 applicants found at least one viable restructuring path).
- **DiCE Optimization Failure Rate**: **38.00%** (76 / 200 applicants failed gradient/genetic optimization).
- **Domain Fallback Search Rate**: **38.50%** (77 / 200 applicants required deterministic heuristic fallback).
- **Mean Generation Latency**: **2153.05 ms/applicant**.

### 5. Sovereign Policy Grounding & Citation Faithfulness (Phase 6)
- **Ingested Sovereign Documents**: Federal Reserve SR 11-7, Basel Committee BCBS 75, RBI Digital Lending Directions (2025), and RBI Master Circular on Fair Practices Code (total 80 chunks).
- **Retrieval Evaluation Status**: 64 candidate queries (320 query-clause pairs) drafted to [`tests/evaluation/to_label.csv`](tests/evaluation/to_label.csv). Marked **BLOCKED PENDING HUMAN GROUND-TRUTH LABELS**; synthetic retrieval metrics purged.
- **Citation Faithfulness on Real Reports**: 150 / 150 citations verified via exact Unicode NFKC match (**100.0% pass rate**).
- **Adversarial Citation Tampering Suite**: 20 / 20 tampered/fabricated quotes caught (**100.0% catch rate**).

### 6. Authentication & Prompt Injection Defense (Phase 7)
- **Authentication**: PBKDF2-HMAC-SHA256 password hashing (100,000 iterations), HS256 JWTs with server-side role claims, 60-min expiry.
- **Header Tampering Defense**: Bare `X-User-Role` headers get HTTP 401; unsigned tokens (`alg=none`) get HTTP 401; role escalation gets HTTP 403.
- **Prompt Injection Benchmark**: Evaluated against 25 adversarial variants across 7 languages (English, Spanish, French, German, Russian, Chinese, Hindi, Arabic), leetspeak, zero-width characters, and Base64/Hex encoding:
  - **Catch Rate**: **100.00% (25 / 25 caught)**.
  - **Benign Controls Preserved**: **5 / 5 (100.00% specificity, 0 false positives)**.

---

## What This Does NOT Prove

As a senior model-risk review standard, the following boundaries must be recognized by institutional audit committees:

1. **Does NOT Prove Demographic Fairness**: Audited disparities exist across age (DIR = 0.8861) and sex (DIR = 0.9367). Removing `AGE` does not eliminate age disparity due to financial proxy variables. This model is not certified as bias-free.
2. **Does NOT Guarantee Restructuring Approval**: A counterfactual scenario is an algorithmic what-if exploration, not a commercial commitment. On real borrowers with severe credit distress, counterfactual validity is only 17.67%.
3. **Does NOT Prove Retrieval Precision on Arbitrary Law**: Because retrieval evaluation is strictly blocked pending human ground-truth annotation for `tests/evaluation/to_label.csv`, retrieval Recall@k is not certified on the sovereign corpus.
4. **Does NOT Prove Human Causal Reality**: TreeSHAP additivity is an algorithmic sanity test of additive feature decomposition; it does not prove that the underlying tree correlations represent real-world causal drivers of default.
5. **Does NOT Guarantee Absolute Immunity to Semantic Injection**: While 100% of tested heuristic and obfuscated variants were caught, heuristic pattern matching is not mathematically complete against arbitrary novel semantic perturbations. Structural nonced XML isolation remains the primary defense.

---

## Clean Start Instructions

### 1. Environment Setup & Data Acquisition

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies (runtime hand-curated, pinned)
pip install -r backend/requirements.txt
pip install -r requirements-dev.txt

# 3. Fetch real datasets and verify cryptographic SHA-256 hashes
python scripts/fetch_data.py

# 4. Fetch real sovereign policy documents (Fed SR 11-7, Basel BCBS 75, RBI)
python scripts/fetch_policy_documents.py

# 5. Partition real data (Train: 21,000, Val: 4,500, Test: 4,500; Seed: 42)
python scripts/prepare_data.py
```

### 2. Model Training & Evaluation

```powershell
# Train baseline & XGBoost, tune on train, calibrate, optimize threshold, and evaluate on test
python scripts/train_and_evaluate_models.py

# Audit fairness across protected attributes (AGE, SEX, EDUCATION, MARRIAGE)
python scripts/audit_fairness.py

# Validate explanation stability, rank correlation, and faithfulness
python scripts/validate_explanations.py

# Evaluate counterfactuals on 200 real rejected test applicants
python scripts/evaluate_counterfactuals_real.py

# Ingest real policy documents, draft retrieval eval set, and audit citation faithfulness
python scripts/evaluate_retrieval_and_citations.py

# Run security benchmark against 25 injection variants and JWT authentication
python scripts/evaluate_security.py
```

### 3. Run Test Suite with Strict Coverage Gate

```powershell
pytest --cov=backend/app/services --cov-report=term-missing --cov-fail-under=80
```

---

## API Authentication & Role Gating

All protected endpoints require an `Authorization: Bearer <token>` header containing a valid HS256 JWT issued by `/api/auth/token`.

| Role | Permitted Actions |
| :--- | :--- |
| `credit_officer` | Execute decision audits, inspect applicant financial features, view counterfactual scenarios. |
| `reviewer` | Record underwriter review dispositions, approve restructuring paths. |
| `auditor` | Verify cryptographic SHA-256 hash-chain audit ledger continuity, inspect full pipeline traces. |
| `compliance_reviewer` | Upload and verify policy documents, trigger index rebuilding. |
| `admin` | Full system governance, document lifecycle management, and index updates. |

### Obtain Authentication Token

```bash
curl -X POST http://localhost:8000/api/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "credit_officer", "password": "OfficerSecure2026!"}'
```

---

## Repository Documentation

- [`docs/WRITEUP.md`](docs/WRITEUP.md): Comprehensive two-page model risk evaluation summary.
- [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md): 90-second executive walkthrough script.
- [`docs/DATA_CARD.md`](docs/DATA_CARD.md): Data provenance, licensing, missingness, and schema mappings.
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md): Intended use, calibration, test performance, and governance threshold.
- [`docs/fairness_report.md`](docs/fairness_report.md): Parity metrics, disparate impact ratios, and sensitive attribute decisions.
- [`docs/explanation_validation_report.md`](docs/explanation_validation_report.md): Permutation importance, stability, and faithfulness curves.
- [`docs/counterfactual_report.md`](docs/counterfactual_report.md): 200-applicant test evaluation, actionability, and DiCE failure analysis.
- [`docs/retrieval_and_citations_report.md`](docs/retrieval_and_citations_report.md): Sovereign corpus ingestion and citation verifier audits.
- [`docs/security_report.md`](docs/security_report.md): Password hashing, JWT role gating, and prompt injection benchmark.
