"""Applicant search and inspection endpoints with PII minimization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.app.api.deps import get_current_user_role

router = APIRouter(prefix="/api/applicants", tags=["Applicants"])


@router.get("")
def list_applicants(
    user_role: str = Depends(get_current_user_role),
    query: str | None = Query(default=None, description="Search term for applicant reference"),
) -> dict[str, Any]:
    """Search and list available applicant records with PII minimization."""
    demo_dir = Path("data/demo")
    applicants = []

    # Applicant #10492 primary fixture
    app_10492_file = demo_dir / "applicant_10492_synthetic.json"
    if app_10492_file.is_file():
        with open(app_10492_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            metadata_keys = {"applicant_id", "applicant_ref", "_notes", "data_mode"}
            features = data.get("features") or {k: v for k, v in data.items() if k not in metadata_keys}
            ref = data.get("applicant_ref", "APP-10492")
            if not query or query.lower() in ref.lower() or query.lower() in "10492":
                applicants.append({
                    "applicant_ref": ref,
                    "label": "Applicant #10492 (Synthetic Commercial Loan Demo)",
                    "requested_amount_inr": features.get("AMT_CREDIT", 1500000.0),
                    "income_inr": features.get("AMT_INCOME_TOTAL", 70000.0),
                    "data_mode": "demo_synthetic",
                    "file_path": str(app_10492_file),
                })

    return {
        "total": len(applicants),
        "applicants": applicants,
        "pii_disclosure": "Records are pseudonymous synthetic fixtures. No actual customer PII is stored or exposed.",
    }


@router.get("/{applicant_ref}")
def get_applicant_details(
    applicant_ref: str,
    user_role: str = Depends(get_current_user_role),
) -> dict[str, Any]:
    """Retrieve minimized feature payload for an applicant by reference ID."""
    demo_dir = Path("data/demo")
    app_10492_file = demo_dir / "applicant_10492_synthetic.json"

    if app_10492_file.is_file():
        with open(app_10492_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            stored_ref = data.get("applicant_ref", "APP-10492")
            if applicant_ref in (stored_ref, "APP-10492", "APP-10492-SYNTHETIC", "10492"):
                metadata_keys = {"applicant_id", "applicant_ref", "_notes", "data_mode"}
                features = data.get("features") or {k: v for k, v in data.items() if k not in metadata_keys}
                return {
                    "applicant_ref": stored_ref,
                    "data_mode": "demo_synthetic",
                    "features": features,
                    "notes": data.get("_notes", "Synthetic applicant fixture for credit risk auditing demo."),
                }

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Applicant record '{applicant_ref}' not found in registered corpus.",
    )
