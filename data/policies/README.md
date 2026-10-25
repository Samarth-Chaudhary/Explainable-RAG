# Sovereign Regulatory Policy Corpus

## Authoritative Public Policy Documents Only

This directory (`data/policies/`) is reserved strictly for authentic, in-force regulatory and supervisory policy documents fetched directly from official sovereign authorities:
1. **Federal Reserve Board & OCC**: *Supervisory Guidance on Model Risk Management (SR 11-7)* (`fed_sr11_7.pdf`)
2. **Basel Committee on Banking Supervision (BIS)**: *Principles for the Management of Credit Risk (BCBS 75)* (`basel_credit_risk.pdf`)
3. **Reserve Bank of India (RBI)**: *Reserve Bank of India (Digital Lending) Directions, 2025* (`rbi_digital_lending_2025.pdf`)
4. **Reserve Bank of India (RBI)**: *Master Circular - Fair Practices Code for NBFCs* (`rbi_fair_practices_code.pdf`)

Every document is downloaded by `scripts/fetch_policy_documents.py` and cryptographically validated via SHA-256 against entries recorded in `data/DATA_SOURCES.md` and `data/POLICY_SOURCES.md`.

### Strict Non-Fabrication Rules
1. **Zero Synthetic Documents**: No synthetic, generated, or mock policy text may reside in this directory. Any test fixtures are restricted exclusively to `tests/fixtures/synthetic_policies/`.
2. **Unbroken Provenance**: Ingested policy chunks maintain immutable source IDs, section titles, page numbers, and SHA-256 checksums.
3. **No Unverified Legal Claims**: Audit reports cite regulatory clauses only when retrieved from this authentic corpus with high-confidence semantic and lexical relevance.
