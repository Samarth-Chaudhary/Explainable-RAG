"""Comprehensive unit and route integration tests for all FastAPI endpoints.

Requires valid JWT Bearer tokens with server-side role claims.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.security import create_access_token

client = TestClient(app)

ADMIN_HEADERS = {"Authorization": f"Bearer {create_access_token('admin_user', 'admin')}"}
OFFICER_HEADERS = {"Authorization": f"Bearer {create_access_token('officer_user', 'credit_officer')}"}
REVIEWER_HEADERS = {"Authorization": f"Bearer {create_access_token('reviewer_user', 'reviewer')}"}
ANALYST_HEADERS = {"Authorization": f"Bearer {create_access_token('analyst_user', 'analyst')}"}


def test_get_models_endpoint():
    """Verify GET /api/models returns model catalog and active schema with valid JWT."""
    response = client.get("/api/models", headers=ANALYST_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "active_model" in data
    assert len(data["models"]) >= 1
    model = data["models"][0]
    assert "XGB" in model["name"]
    assert model["features_count"] == 11
    assert "default_threshold" in model
    assert model["default_threshold"]["value"] == 0.35


def test_get_applicants_endpoint():
    """Verify GET /api/applicants returns minimized applicant fixtures with valid JWT."""
    response = client.get("/api/applicants", headers=OFFICER_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    assert any("10492" in a["applicant_ref"] for a in data["applicants"])

    # Test single applicant retrieval
    app_res = client.get("/api/applicants/APP-10492", headers=OFFICER_HEADERS)
    assert app_res.status_code == 200
    app_data = app_res.json()
    assert "features" in app_data
    assert "AMT_CREDIT" in app_data["features"]
    assert app_data["features"]["AMT_CREDIT"] > 0


def test_audit_execution_and_retrieval_lifecycle():
    """Verify complete audit execution, polling, report fetching, trace, and disposition."""
    # Load applicant fixture
    with open("data/demo/applicant_10492_synthetic.json", "r", encoding="utf-8") as f:
        applicant = json.load(f)

    # 1. Trigger Run Audit (synchronous)
    run_payload = {
        "applicant_data": applicant,
        "applicant_ref": "APP-10492-SYNTHETIC",
        "idempotency_key": "IDEMP-TEST-KEY-001",
        "user_role": "credit_officer",
        "run_async": False,
    }
    response = client.post("/api/audits", json=run_payload, headers=OFFICER_HEADERS)
    assert response.status_code == 201
    job_data = response.json()
    assert job_data["status"] == "completed"
    assert job_data["progress_percentage"] == 100
    audit_id = job_data["audit_id"]
    assert audit_id is not None
    assert job_data["report"] is not None

    # 2. Test Idempotency with same key
    idemp_res = client.post("/api/audits", json=run_payload, headers=OFFICER_HEADERS)
    assert idemp_res.status_code == 201
    assert idemp_res.json()["audit_id"] == audit_id

    # 3. Read Audit Summary
    summary_res = client.get(f"/api/audits/{audit_id}", headers=OFFICER_HEADERS)
    assert summary_res.status_code == 200
    summary = summary_res.json()
    assert summary["audit_id"] == audit_id
    assert summary["chain_integrity_valid"] is True
    assert summary["events_count"] >= 5

    # 4. Read Strict Dossier JSON Report
    report_res = client.get(f"/api/audits/{audit_id}/report", headers=OFFICER_HEADERS)
    assert report_res.status_code == 200
    report = report_res.json()
    assert report["audit_id"] == audit_id
    assert report["human_review_required"] is True
    assert report["prediction"]["label"] == "REJECT_HIGH_RISK"
    assert len(report["evidence"]) > 0

    # 5. Read Retrieval Trace
    trace_res = client.get(f"/api/audits/{audit_id}/trace", headers=OFFICER_HEADERS)
    assert trace_res.status_code == 200
    trace = trace_res.json()
    assert trace["audit_id"] == audit_id
    assert "dense_query" in trace
    assert "bm25_terms" in trace

    # 6. Export PDF Dossier
    pdf_res = client.get(f"/api/audits/{audit_id}/export.pdf", headers=OFFICER_HEADERS)
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert len(pdf_res.content) > 1000

    # 7. Record Reviewer Disposition (Role: reviewer)
    disp_payload = {
        "disposition": "approve",
        "notes": "Reviewed counterfactual candidate CF-1; acceptable under tenor extension policy.",
        "reviewer_id": "REV-OFFICER-789",
    }
    disp_res = client.post(
        f"/api/audits/{audit_id}/disposition",
        json=disp_payload,
        headers=REVIEWER_HEADERS,
    )
    assert disp_res.status_code == 200
    assert disp_res.json()["status"] == "recorded"


def test_documents_lifecycle_and_reindex():
    """Verify document listing, status update, and reindexing."""
    # List documents
    docs_res = client.get("/api/documents", headers=ADMIN_HEADERS)
    assert docs_res.status_code == 200
    data = docs_res.json()
    assert "documents" in data

    # Reindex (requires admin or compliance_reviewer)
    reindex_res = client.post(
        "/api/documents/SYN-POL-FOIR-001/reindex",
        headers=ADMIN_HEADERS,
    )
    assert reindex_res.status_code == 200
    assert reindex_res.json()["status"] == "reindexed"
    assert "index_version" in reindex_res.json()


def test_role_based_access_rejections():
    """Verify that unauthorized roles are forbidden from sensitive actions."""
    # Analyst cannot record reviewer disposition (requires reviewer, compliance_reviewer, or admin)
    disp_res = client.post(
        "/api/audits/AUDIT-NONEXISTENT/disposition",
        json={"disposition": "approve", "notes": "test", "reviewer_id": "1"},
        headers=ANALYST_HEADERS,
    )
    assert disp_res.status_code == 403
    assert "Access denied" in disp_res.json()["message"]


def test_evaluation_metrics_endpoint():
    """Verify GET /api/evaluation/metrics returns benchmark metrics or safe unmeasured disclaimers."""
    response = client.get("/api/evaluation/metrics", headers=ANALYST_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "benchmark_claim_control" in data or "retrieval_metrics" in data
