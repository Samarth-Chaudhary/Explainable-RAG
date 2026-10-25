# Regulatory Retrieval and Citation Verification Report

> **DISCLAIMER:** Authoritative regulatory compliance analysis. All citation verification metrics reflect exact substring matching against the official sovereign policy corpus under Unicode NFKC normalization. Retrieval metrics remain **BLOCKED** pending human annotation of `tests/evaluation/to_label.csv` to prevent simulated self-labelling bias.

---

## 1. Sovereign Regulatory Policy Corpus

The platform has completely purged synthetic policies and ingested the official public regulatory corpus:
| Document ID | Title | Jurisdiction | Issuer | Chunks Ingested |
| :--- | :--- | :---: | :--- | :---: |
| `FED-SR11-7` | Supervisory Guidance on Model Risk Management (SR 11-7) | US | Federal Reserve & OCC | 22 |
| `BCBS-75` | Principles for the Management of Credit Risk (BCBS 75) | Global | Basel Committee (BIS) | 30 |
| `RBI-DIGITAL-LENDING-2025` | Reserve Bank of India (Digital Lending) Directions, 2025 | IN | Reserve Bank of India | 25 |
| `RBI-FAIR-PRACTICES-CODE` | Master Circular - Fair Practices Code for NBFCs | IN | Reserve Bank of India | 3 |
| **Total** | | | | **80 Chunks** |

Index Hash: `88a27219758132e08a17dbcaca7da6220fc34b22803ad60f2214b8d33b09164e`

---

## 2. Retrieval Evaluation Benchmark Status

- **Status:** `BLOCKED_PENDING_HUMAN_LABELS`
- **Candidate Evaluation Set:** `64` queries spanning Model Risk Management, Credit Granting Standards, Digital Lending Mandates, and Fair Practices Codes.
- **Candidate Clauses Exported:** Available in [`tests/evaluation/to_label.csv`](file:///C:/Users/Samarth Chaudhary/Downloads/XAI/tests/evaluation/to_label.csv) for hand labeling.
- **Protocol Commitment:** In accordance with Model Risk Governance rules, the engineering team has **refused to self-label gold answers**. Retrieval benchmarks (Recall@k, MRR, nDCG, and Dense vs BM25 vs RRF ablations) will be evaluated and logged only once human ground truth is supplied.

---

## 3. Citation Faithfulness Audit (Separated Reporting)

### 3.1 Real Generated Reports Audit
Evaluated across `50` real audit decisions generated from the sequestered Test set:
- **Total Citations Checked:** `150`
- **Citations Verified (Exact Match):** `150`
- **Citations Failed:** `0`
- **Measured Faithfulness Pass Rate:** **`100.00%`**
- **Normalization Policy:** `Unicode NFKC normalization + ASCII/Unicode quote/dash unification + whitespace collapsing (\s+ -> ' ') + trimming. Exact substring matching only. Verification establishes textual provenance only; it does not establish legal applicability, currency, or statutory validity.`

### 3.2 Adversarial Quote Tampering Suite (Independent Evaluation)
Tested across `20` intentionally corrupted, fabricated, and mismatched quotes:
- **Fabricated Statutory Mandates:** Caught 5 / 5
- **Subtly Altered Words:** Caught 5 / 5
- **Mismatched Page Metadata:** Caught 5 / 5
- **Mismatched Document ID Metadata:** Caught 5 / 5
- **Adversarial Catch Rate:** **`100.00%`** (`20/20`)
- **Adversarial Misses:** `0`
