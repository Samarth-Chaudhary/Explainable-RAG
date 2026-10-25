# 90-Second Walkthrough: Explainable Credit-Risk Auditing Platform

**Target Audience**: Senior Underwriters, Model Risk Reviewers, and Internal Audit Officers  
**Duration**: Exactly 90 Seconds  
**Objective**: Demonstrate end-to-end defensibility: real-data scoring, local TreeSHAP attribution, constrained counterfactual feasibility, sovereign policy citation verification, and tamper-evident audit logging.

---

### Phase 1: Authentication & Applicant Rejection (0:00 - 0:25)
1. **Login & Role Assertion**:
   - Access the dashboard or API endpoint `/api/auth/token` with institutional credentials (`credit_officer` / `OfficerSecure2026!`).
   - Highlight that client-supplied `X-User-Role` headers are cryptographically rejected; authorization uses verified HS256 JWT server-side claims.
2. **Execute Audit on Applicant**:
   - Select real test applicant (`APP-10492` from test set, debt-to-income distress, high utilization).
   - Trigger synchronous audit via `POST /api/audits`.
   - Show the 10-stage execution pipeline completing deterministically:
     - Calibrated XGBoost evaluates default probability $P(\text{Default}) = 0.8100$.
     - Operational decision threshold $\tau^* = 0.1300$ (cost-optimized on validation split, $C_{FN}/C_{FP} = 5.0$) mandates **REJECT_HIGH_RISK**.

---

### Phase 2: TreeSHAP Attribution & Constrained Counterfactuals (0:25 - 0:50)
1. **Explainability & Sanity Verification**:
   - Inspect the TreeSHAP waterfall in the Dossier report.
   - Primary risk-increasing drivers: `FOIR_RATIO` (+1.42 log-odds), `CREDIT_BUREAU_SCORE` (+0.95), and `LTV_RATIO` (+0.61).
   - Point out the algorithmic sanity check: Additivity residual $|f(x) - \sum \phi_i| = 2.1 \times 10^{-7} \le 10^{-4}$.
2. **Constrained What-If Counterfactuals**:
   - Review proposed remediation paths.
   - Prove **immutability constraints**: `AGE`, `CREDIT_BUREAU_SCORE`, and `PAST_DELINQUENCIES` are strictly frozen.
   - Candidate Path: Increasing liquid collateral or extending loan tenor to lower `FOIR_RATIO` from 0.72 to 0.48 brings default probability to $0.1150 < \tau^*$.
   - Point to the model risk card: On 200 real test rejections, counterfactual validity is **17.67%**, proving the system does not promise unrealistic approval.

---

### Phase 3: Sovereign Policy Grounding & Audit Hash Ledger (0:50 - 1:15)
1. **Authoritative Policy Retrieval**:
   - Show retrieved regulatory clauses grounding the rejection.
   - Passages link directly to official public guidance: **RBI Digital Lending Directions 2025**, **Basel BCBS 75**, and **Fed SR 11-7**.
   - Show citation verifier output: Exact Unicode NFKC match passed with **100.0% faithfulness** (150/150 citations verified).
2. **Prompt Injection Containment**:
   - Demonstrate that untrusted policy passages are wrapped in cryptographically nonced XML tags (`<untrusted_policy_data nonce="...">`), preventing instruction override attacks.

---

### Phase 4: Tamper-Evident Ledger & Reviewer Sign-Off (1:15 - 1:30)
1. **Cryptographic Lineage**:
   - Inspect the audit trail at `GET /api/audits/{id}`.
   - Demonstrate the SHA-256 block chain (`previous_event_hash`, `payload_hash`, `event_hash`).
   - Hash chain integrity verifies: `chain_integrity_valid: true`.
2. **Human Reviewer Sign-Off**:
   - Submit underwriter review action via `POST /api/audits/{id}/disposition` (`"disposition": "approve_with_counterfactual_restructuring"`).
   - Export official compliance PDF dossier via `/api/audits/{id}/export.pdf` complete with institutional audit hash and human review watermark.
