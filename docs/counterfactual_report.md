# Counterfactual Evaluation on Real Rejected Test Applicants

## 1. Executive Summary & Review Scope
- **Audited Model:** PRODUCTION-XGBOOST-CREDIT (v2.0.0-real)
- **Evaluation Cohort:** $N = 200$ real rejected applicants sampled randomly from the independent Test split (`data/processed/test.csv`)
- **Operational Target Threshold:** $\tau = 0.3500$
- **Candidates Evaluated per Applicant:** 3 candidate profiles ($N_{total} = 600$)
- **Confidence Intervals:** 1,000 bootstrap resamples for 95% Confidence Intervals

> [!IMPORTANT]
> **Hostile Model-Risk Review Notice on Algorithmic Recourse**: Generating counterfactuals on real credit data is constrained by fundamental repayment physics: historical bureau delinquencies and credit scores cannot be altered by restructuring loan terms. When credit bureau scores are severely impaired, actionable modifications (loan amount, tenor, collateral) alone cannot always reduce default risk below the approval cutoff. Below is the honest, unvarnished performance of the counterfactual engine across the real test cohort.

---

## 2. Quantitative Performance Metrics (with 95% Bootstrap CIs)

| Evaluation Metric | Measured Mean | 95% Bootstrap CI | Definition / Calculation Basis |
| :--- | :---: | :---: | :--- |
| **Candidate Validity Rate** | `17.67%` | [12.83%, 22.50%] | Candidates meeting target risk ($P(Y=1) < 0.35$) / all candidates evaluated |
| **Candidate Actionability Rate** | `17.67%` | [12.83%, 22.50%] | Candidates meeting target risk **and** satisfying all immutability, range, and derivation constraints |
| **Feasible Applicant Rate** | `20.50%` | [15.50%, 26.00%] | Proportion of applicants for whom at least one fully feasible remediation path exists |
| **Proximity (Normalized L1)** | `0.0745` | [0.0624, 0.0881] | Mean normalized absolute distance across actionable levers |
| **Sparsity (Features Altered)** | `1.48` | [1.38, 1.59] | Mean number of actionable features modified per candidate (out of 3 possible levers) |
| **Generation Latency** | `2153.05 ms` | [792.20 ms, 3860.11 ms] | Total optimization and validation runtime per applicant |

---

## 3. DiCE Optimization Failures & Fallback Search Disclosure

In real-world credit risk pipelines, gradient-based or genetic counterfactual algorithms frequently encounter non-convex barriers where actionable perturbations cannot overcome fixed bureau scores. We disclose the exact failure and fallback frequencies:

- **DiCE Optimization Failure Rate:** `38.00%` (76 / 200 applicants)
  - DiCE either threw an exception, timed out, or returned zero candidate vectors satisfying the target class within its search bounds.
- **Constrained Domain Fallback Search Rate:** `38.50%` (77 / 200 applicants)
  - When DiCE failed to discover feasible candidates, the constrained domain restructuring search (tenor extension, collateral pledging, credit downpayment) was automatically invoked.
- **Model-Risk Review Finding:** Without the domain fallback search, recourse generation would have completely failed for 38.00% of rejected applicants. The hybrid architecture is necessary to provide practical applicant feedback.

---

## 4. Immutability & Constraint Enforcement Audit

The post-generation validation layer strictly enforces feature mutability classifications. Any candidate altering a frozen feature is immediately rejected:

| Feature Category | Features Enforced | Mutability Status | Enforcement Mechanism |
| :--- | :--- | :---: | :--- |
| **Protected Demographic** | `AGE`, `SEX`, `EDUCATION`, `MARRIAGE` | `sensitive` / `immutable` | Post-generation delta check ($|x_{cand} - x_{orig}| < 10^{-4}$); rejects candidate if altered |
| **Credit Bureau History** | `CREDIT_BUREAU_SCORE`, `PAST_DELINQUENCIES`, `DAYS_EMPLOYED` | `immutable` | Hard constraint check; prevents applicants from 'erasing' past delinquencies |
| **Financial Capacity** | `AMT_INCOME_TOTAL`, `EXISTING_DEBT_OBLIGATIONS` | `immutable` | Verified income and external debts cannot be inflated or hidden during restructuring |
| **Actionable Levers** | `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE` | `actionable` | Bounded range checks; loan amount, installment, and collateral value eligible for change |
| **Derived Ratios** | `FOIR_RATIO`, `LTV_RATIO` | `derived` | **Never** perturbed directly; automatically recalculated from updated primitives prior to re-scoring |
