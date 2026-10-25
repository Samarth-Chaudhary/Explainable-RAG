# Synthetic Policy Test Fixtures

**CRITICAL COMPLIANCE NOTICE**:
The PDF files in this directory are synthetic, miniature test fixtures designed strictly for fast unit-testing of edge cases in PDF chunking, metadata parsing, and boundary handling.

**DO NOT USE IN PRODUCTION PATHS**:
Under no circumstances should any code in `backend/app/` or production pipelines load or ingest files from this directory. All production policy retrieval and explainability pipelines must exclusively ingest the official, sovereign regulatory documents downloaded by `scripts/fetch_policy_documents.py` into `data/policies/`.
