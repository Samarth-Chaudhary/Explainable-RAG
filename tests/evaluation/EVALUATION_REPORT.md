# XAI-RAG Quantitative Evaluation Report

> **DISCLAIMER:** Demonstrative/synthetic benchmark dataset. All numbers below are computed by `tests/evaluation/eval_cli.py` on 2026-10-20 04:38:10 UTC. No targets or hypothetical aspirations are reported as achieved performance without verification.

---

## 1. Counterfactual Metrics
- **Validity Rate:** `100.00%` (2/2 evaluated candidates re-scored $\le 0.35$)
- **Actionability Rate:** `100.00%` (2/2 evaluated candidates satisfied target AND passed all immutable/domain/range/derivation constraints)
- **Mean Generation Latency:** `0.022s`
- **Mean Target Miss Magnitude:** `0.0`
- **Sample Size:** `3` applicant scenarios evaluated

---

## 2. Attribution-to-Retrieval Quality Benchmark
*Benchmark Claim Control: Research hypotheses (~52% naive baseline, 91%+ target Recall@3) are aspirational hypotheses only.*

| Metric | Ontology-Bridged Translator | Naive Feature-String Baseline | Difference |
| :--- | :---: | :---: | :---: |
| **Recall@3** | **`83.33%`** | `50.00%` | `+33.33%` |
| **Precision@3** | **`44.44%`** | `22.22%` | `+22.22%` |
| **MRR** | **`0.8333`** | `0.2222` | `+0.6111` |
| **nDCG@3** | **`0.7689`** | `0.2689` | `+0.5000` |

- **Sample Size:** `3` labelled test scenarios
- **Averaging Method:** `macro_average`

---

## 3. Citation Faithfulness
- **Citation Faithfulness Score:** `80.00%` (4/5 verified against corpus)
- **Fabricated Quote Detection:** `PASSED (Fabrication detected)`
- **Normalization Policy:** `Unicode NFKC normalization + ASCII/Unicode quote/dash unification + whitespace collapsing (\s+ -> ' ') + trimming. Exact substring matching only. Verification establishes textual provenance only; it does not establish legal applicability, currency, or statutory validity.`

---

## 4. Explanation Fidelity
- **Output Space:** `log_odds`
- **Base Value:** `-5.717`
- **SHAP Additivity Residual:** `0.000000`
- **Residual Within Tolerance ($|\epsilon| \le 10^{-4}$):** `True`

---
*Report generated automatically by `tests/evaluation/eval_cli.py`.*
