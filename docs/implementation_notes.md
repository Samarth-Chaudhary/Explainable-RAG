# XAI-RAG Implementation Notes (Part 1 Build)

## Inspection and Runtime Environment
- **Date**: October 2026
- **Operating System**: Windows (PowerShell + cmd runtime)
- **Python Runtime**: Python 3.11.6
- **Workspace State at Kickoff**: Clean, uninitialized workspace directory.
- **Dependency Environment**: Initialized `.venv` with Python 3.11 containing:
  - `xgboost` 3.2.0
  - `shap` 0.51.0
  - `dice-ml` 0.12
  - `scikit-learn` 1.9.1 / 1.4+
  - `pydantic` 2.14.1
  - `pydantic-settings` 2.16.0
  - `fastapi` 0.149.0
  - `pytest` 9.1.1, `pytest-cov` 7.1.0

## Core Capabilities Delivered
1. **Repository Scaffold**: Complete backend, schema, data, docs, and test layout.
2. **Contracts & Strict Schemas**:
   - Feature Schema specification with mutability classes (`immutable`, `actionable`, `derived`, `sensitive`), bounds, null handling, and deterministic order hashing.
   - Model Manifest schema with SHA-256 integrity and calibration metadata.
   - Ontology Schema and Source Registry Schema.
   - Strict `AuditReport` schema with `extra="forbid"`.
   - Handoff contracts: `PredictionResult`, `ExplanationResult`, `CounterfactualResult`.
   - Typed stubs for future modules: `HybridQueryPayload`, `Chunk`, `RetrievalTrace`, `RetrievedEvidence`, `VerifierResult`, `AuditEvent`, `ErrorEnvelope`, `JobStatus`.
3. **Pillar A (Prediction)**:
   - Model loader with artifact SHA-256 verification and feature order hash validation.
   - Strict input validation rejecting missing, extra, non-numeric, out-of-range, or misordered features.
   - Derived feature recalculation (`FOIR_RATIO`, `LTV_RATIO`).
   - Decision threshold comparison edge cases (documented `>=` rule).
4. **Pillar A (SHAP Explanation)**:
   - `TreeExplainer` with explicit output space (`log_odds` margin vs `probability`).
   - Mathematical additivity verification: $base\_value + \sum \phi_i = raw\_margin$ in log-odds, and sigmoid transformation check.
   - Degraded / invalid state handling when tolerance is breached (blocking narrative generation).
   - Detection and failure on invalid cross-space comparisons.
   - Full ranking and extraction of top risk-increasing and risk-reducing drivers.
5. **Pillar B (DiCE Constrained Counterfactuals)**:
   - Constrained counterfactual optimizer wrapping DiCE / genetic search.
   - Strict immutability post-validation (freezing immutable attributes such as age, credit score, delinquencies, tenure).
   - Dynamic recomputation of derived ratios (`FOIR_RATIO`, `LTV_RATIO`) from perturbed primitives prior to re-scoring.
   - Direct production re-scoring of each candidate through `services/prediction.py`.
   - Exact message on failure: `"No feasible counterfactual found under configured constraints"`.
   - Comprehensive metrics: validity, actionability, proximity, diversity, target-miss magnitude, generation time, with explicit denominators.
6. **Synthetic Demo Fixtures**:
   - `SYNTHETIC-DEMO-XGB`: small XGBoost binary classifier trained with fixed seed (42).
   - `APPLICANT-10492-SYNTHETIC`: Scenario B synthetic applicant fixture.

## Known Limitations & Boundaries
- Part 1 does NOT implement live document ingestion, vector retrieval, LLM synthesis, or UI. Those services raise `NotImplementedError("Implemented in Part 2/3")` per specification.
- Model is a synthetic demonstration model and must never be represented as validated credit-risk performance.
