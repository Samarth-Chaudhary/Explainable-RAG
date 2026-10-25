# Data Sources & Cryptographic Provenance Registry

All real datasets used for model training and risk evaluation must be verified by SHA-256.

---

## UCI Default of Credit Card Clients (Zero-Login Fallback)
- **Source Identifier:** `ucimlrepo:id=350`
- **Local Path:** `data\raw\uci_credit_card.csv`
- **Retrieval Timestamp:** 2026-10-25 09:28:39 UTC
- **File Size:** 2,728,183 bytes
- **SHA-256 Digest:** `20a2540061fab3984f208fbe55745b0d60e3a86207c8b02e408fd23f1a3a4b94`
- **Dimensions:** 30,000 rows × 24 columns
- **License / Terms:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Provenance & Verification Notes:** Live verified via official ucimlrepo package. Real financial default benchmark.

---

## OpenML German Credit (Benchmark Control)
- **Source Identifier:** `openml:data_id=31`
- **Local Path:** `data\raw\german_credit.csv`
- **Retrieval Timestamp:** 2026-10-25 09:28:44 UTC
- **File Size:** 140,017 bytes
- **SHA-256 Digest:** `9888fb06e7bf42963f4db2c8fd57666d5426c0a3692d10b51571de49518a105a`
- **Dimensions:** 1,000 rows × 21 columns
- **License / Terms:** Public Domain / CC0
- **Provenance & Verification Notes:** Live verified via OpenML API (credit-g).

---


## Official Regulatory Policy Documents

| Document ID | Title | Issuer | Jurisdiction | File Size | SHA-256 Checksum | Official URL | Retrieval Date (UTC) |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- | :---: |
| `FED-SR11-7` | Supervisory Guidance on Model Risk Management (SR 11-7) | Federal Reserve Board & OCC | US | 300,659 B | `0046c0e4eb705830d43312da000271655000beb2a349e1cc3802815638c62cc0` | [Official Source](https://www.federalreserve.gov/boarddocs/srletters/2011/sr1107a1.pdf) | `2026-10-25` |
| `BCBS-75` | Principles for the Management of Credit Risk (BCBS 75) | Basel Committee on Banking Supervision (BIS) | Global / International | 133,982 B | `08fa2bbe0930007e2fd6c637190febdd288b8d3d59df4d36251e57d94d97ae26` | [Official Source](https://www.bis.org/publications/200009-guidelines-principles-management-credit-risk.pdf) | `2026-10-25` |
| `RBI-DIGITAL-LENDING-2025` | Reserve Bank of India (Digital Lending) Directions, 2025 (RBI/2025-26/36) | Reserve Bank of India (RBI) | IN | 1,120,752 B | `2dd838cf9bcb58f27bec0c0f5485cf1c764ad238454dea8c9041516375d2321b` | [Official Source](https://rbidocs.rbi.org.in/rdocs/notification/PDFs/36NT8C402BE7C2A349E0BFFF3C526668CD7A.PDF) | `2026-10-25` |
| `RBI-FAIR-PRACTICES-CODE` | Master Circular - Fair Practices Code for Non-Banking Financial Companies | Reserve Bank of India (RBI) | IN | 584,065 B | `1a479e68d24e9fd89251a7bcdfef14bd9db80686fa4cf06b8b8f47a792f01dfc` | [Official Source](https://rbidocs.rbi.org.in/rdocs/notification/PDFs/AI141P060614F.PDF) | `2026-10-25` |
