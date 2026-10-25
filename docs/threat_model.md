# Threat Model & Security Posture

## 1. Assets and Boundaries
- **Model Artifacts**: XGBoost model weights and feature schema. Protection: Cryptographic SHA-256 verification at load time to prevent tampering or backdoor injection.
- **Applicant Data**: Financial records, debt obligations, income, and personal attributes. Protection: Pseudonymous applicant references (`applicant_ref`), absence of unnecessary PII, strict schema validation.
- **Policy Corpus & Citations**: Official credit guidelines and regulations. Protection: Immutable source registry with SHA-256 file hashes.
- **Audit Ledger**: Complete record of predictions, explanations, and counterfactuals. Protection: Append-only ledger with cryptographic chaining (implemented in Part 2).

## 2. Threat Scenarios & Mitigations
| Threat | Description | Mitigation Strategy |
| :--- | :--- | :--- |
| **Adversarial Input Manipulation** | Malicious inputs with extra fields, NaN injection, or type confusion | Strict Pydantic v2 schemas (`extra='forbid'`), field-level type and range checks |
| **Model Inversion / Feature Order Attack** | Silent re-ordering of features leading to inverted risk scoring | Explicit feature order hash computed and checked before scoring |
| **Counterfactual Gamification** | Altering immutable fields (e.g. reducing applicant age or erasing delinquency history) | Hard immutability constraints enforced post-generation; immediate candidate rejection |
| **Prompt Injection / Citation Hallucination** | Malicious content in policy PDFs attempting to hijack LLM synthesis | Retrieval strictly verified by exact character/token matching against authoritative chunks |
| **Mathematical Deception** | Mixing log-odds margin contributions with probability metrics | Explicit output-space typing; strict additivity residual gating |
