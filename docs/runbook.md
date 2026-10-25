# XAI-RAG Operations Runbook

## 1. Quickstart Commands

### Windows PowerShell
```powershell
# 1. Activate environment
cd /path/to/XAI
.\.venv\Scripts\Activate.ps1

# 2. Run synthetic demo asset bootstrap
.\.venv\Scripts\python -m backend.app.demo_bootstrap

# 3. Start backend REST API (port 8000)
.\.venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# 4. In a separate terminal, start React frontend dashboard (port 5173)
cd frontend
npm install
npm run dev
```

### Unix / macOS Bash
```bash
# 1. Activate environment
cd /path/to/XAI
source .venv/bin/activate

# 2. Run bootstrap
python -m backend.app.demo_bootstrap

# 3. Start backend API
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# 4. Start frontend
cd frontend && npm install && npm run dev
```

---

## 2. Test Suites Execution

```powershell
# Run all unit and integration tests (84 tests)
pytest -v

# Run with test coverage report
pytest --cov=backend.app --cov-report=term-missing tests/
```

---

## 3. Quantitative Evaluation Execution

```powershell
# Run quantitative benchmark evaluation harness
python tests/evaluation/eval_cli.py
```
This executes:
1. Counterfactual validity and actionability rate measurement over test applicants.
2. Attribution-to-retrieval Recall@3, Precision@3, MRR, nDCG@3 comparing the semantic ontology bridge vs naive baseline.
3. Citation faithfulness score calculation with deterministic exact-text substring verification.
4. SHAP explanation fidelity and additivity checks.
Outputs are saved to `tests/evaluation/evaluation_results.json` and `tests/evaluation/EVALUATION_REPORT.md`.

---

## 4. Environment Variables Reference

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `XAI_MODE` | `demo` | `demo` uses synthetic models & data; `configured` enables production checks |
| `XAI_MODEL_PATH` | `data/demo/model.json` | Path to XGBoost JSON model artifact |
| `XAI_MODEL_MANIFEST_PATH`| `data/demo/model_manifest.json` | Path to model metadata manifest |
| `XAI_FEATURE_SCHEMA_PATH`| `data/demo/feature_schema.json` | Path to feature schema definition |
| `XAI_ONTOLOGY_PATH` | `data/ontology/credit_risk_ontology.json` | Domain feature-to-policy ontology |
| `XAI_DATABASE_URL` | `sqlite:///./data/xai_audit.db` | Persistence database URI |
| `XAI_THRESHOLD_VALUE` | `0.35` | Default probability decision cutoff |
| `XAI_THRESHOLD_OWNER` | `Credit Committee` | Accountable executive or governance body |
| `XAI_SHAP_OUTPUT_SPACE` | `log_odds` | Primary coordinate space for SHAP (`log_odds` or `probability`) |
| `XAI_LLM_PROVIDER` | `local_deterministic` | LLM synthesis provider (`local_deterministic`, `gemini`, `openai`) |

---

## 5. Security & Maintenance Operations

- **Database Inspection**: SQLite database is located at `data/xai_audit.db`. You can query tables `audits`, `audit_events`, `documents`, `chunks`, `index_builds`, `reviewer_actions`, and `retrieval_traces`.
- **Index Rebuild**: Authorized administrators can trigger `POST /api/documents/{document_id}/reindex` with `X-User-Role: admin`.
- **Reviewer Dispositions**: Human underwriters record dispositions via `POST /api/audits/{audit_id}/disposition` with `X-User-Role: reviewer`.
