# Dataset Card: Credit Decisioning & Risk Explainability Benchmark

## 1. Dataset Summary & Governance
- **Canonical Dataset:** UCI Default of Credit Card Clients
- **Source Identifier:** `ucimlrepo:id=350`
- **License / Terms of Use:** CC BY 4.0
- **Primary Domain:** Retail & consumer credit repayment and default risk auditing.
- **Verification Integrity:** Verified by live fetching script `scripts/fetch_data.py` and SHA-256 recorded in `data/DATA_SOURCES.md`.

---

## 2. Dataset Splitting & Preprocessing Order
Following strict model-risk management guidelines (Fed SR 11-7 / Basel III), data splitting occurs **prior to any featurization, imputation, or scaling**.

- **Total Sample Size:** 30,000 observations
- **Split Ratios:** 70% Train (21,000), 15% Validation (4,500), 15% Test (4,500)
- **Split Strategy:** Stratified by default target (`TARGET`) with fixed random seed (`42`).
- **Target Prevalence (Default Rate):**
  - Full Dataset: `22.12%`
  - Train Split: `22.12%`
  - Validation Split: `22.11%`
  - Test Split (Touched exactly once): `22.13%`

---

## 3. Explicit Feature Engineering & Unit Harmonization
| Feature | Raw Field Origin | Unit / Coordinate Space | Preprocessing Decision & Transformation |
| :--- | :--- | :--- | :--- |
| `AMT_CREDIT` | Credit line granted / limit | INR / Currency | Cleaned and bounded to schema bounds [50k, 10M] |
| `AMT_ANNUITY` | Monthly repayment installment | INR / Month | Verified monthly amortization installment [1k, 500k] |
| `AMT_GOODS_PRICE`| Collateral / asset value | INR | Appraised asset value ensuring baseline LTV calculation |
| `AMT_INCOME_TOTAL`| Verified monthly net income | INR / Month | Annual income divided by 12 (Home Credit) or underwriting multiple (UCI) |
| `EXISTING_DEBT_OBLIGATIONS`| Debt obligations | INR / Month | Current monthly debt service from credit bureau / billing |
| `FOIR_RATIO` | Derived: Debt-to-income | Ratio [0.0, 2.5] | Dynamically recomputed: `(AMT_ANNUITY + EXISTING_DEBT_OBLIGATIONS) / AMT_INCOME_TOTAL` |
| `LTV_RATIO` | Derived: Loan-to-value | Ratio [0.0, 2.0] | Dynamically recomputed: `AMT_CREDIT / AMT_GOODS_PRICE` |
| `CREDIT_BUREAU_SCORE`| External repayment score | Score [300, 900] | Canonical credit bureau score scale reflecting historical repayment |
| `PAST_DELINQUENCIES`| Delinquent periods | Count [0, 20] | Verified count of 30+ DPD delinquency events |
| `DAYS_EMPLOYED` | Employment tenure | Negative days | Sentinel values (e.g. 365243) replaced with 0.0 non-employment |
| `AGE` | Applicant age | Years [18, 85] | Extracted continuous age in completed years |

---

## 4. Missingness Table (Train Split)
| Feature | Missing Values Count (%) | Resolution Strategy |
| :--- | :--- | :--- |
| `FOIR_RATIO` | 0 (0.00%) | None / Valid range verified |
| `LTV_RATIO` | 0 (0.00%) | None / Valid range verified |
| `CREDIT_BUREAU_SCORE` | 0 (0.00%) | None / Valid range verified |
| `PAST_DELINQUENCIES` | 0 (0.00%) | None / Valid range verified |
| `DAYS_EMPLOYED` | 0 (0.00%) | None / Valid range verified |
| `AGE` | 0 (0.00%) | None / Valid range verified |
| `AMT_CREDIT` | 0 (0.00%) | None / Valid range verified |
| `AMT_ANNUITY` | 0 (0.00%) | None / Valid range verified |
| `AMT_INCOME_TOTAL` | 0 (0.00%) | None / Valid range verified |
| `AMT_GOODS_PRICE` | 0 (0.00%) | None / Valid range verified |
| `EXISTING_DEBT_OBLIGATIONS` | 0 (0.00%) | None / Valid range verified |

---

## 5. Known Biases & Distributional Limitations
1. **Historical Repayment Bias:** Credit card and micro-lending datasets reflect institutional granting criteria at time of collection. Individuals denied credit historically do not appear in default outcome observations (survivorship bias).
2. **Demographic Imbalance:**
   - Sex: Male: ~39.6%, Female: ~60.4%
   - Marital Status: Single (~53.2%), Married (~45.5%)
3. **Macroeconomic Conditions:** Default rates reflect specific interest rate regimes and liquidity environments. Models calibrated on this data must undergo periodic drift monitoring before production deployment.
4. **Proxy Risk:** Bureau scores and delinquency counts can act as institutional proxies for systemic demographic disparities; Phase 3 fairness audits quantify these disparities directly.
