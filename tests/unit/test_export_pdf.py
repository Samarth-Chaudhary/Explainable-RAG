"""Unit tests for PDF export service."""

from datetime import datetime, timezone
from pathlib import Path
import pytest
import pymupdf

from backend.app.schemas.audit_report import (
    AuditReport,
    ReportCounterfactualCandidate,
    ReportDossierSummary,
    ReportEvidenceItem,
    ReportExplanationSummary,
    ReportFeatureContribution,
    ReportModelSummary,
    ReportPredictionSummary,
    ReportSearchTraceSummary,
    ReportVerifierSummary,
)
from backend.app.services.export_pdf import PDFExportService


@pytest.fixture
def sample_audit_report() -> AuditReport:
    """Construct a full valid AuditReport for testing export."""
    now = datetime.now(timezone.utc)
    return AuditReport(
        audit_id="AUDIT-TEST-PDF-001",
        applicant_ref="APP-10492-SYNTHETIC",
        generated_at=now,
        mode="demo",
        status="complete",
        model=ReportModelSummary(
            name="CreditRiskXGBoostDemo",
            version="1.0.0",
            artifact_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            calibration_status="calibrated",
        ),
        prediction=ReportPredictionSummary(
            raw_output=3.3045,
            default_probability=0.9646,
            threshold=0.35,
            label="REJECT_HIGH_RISK",
            output_notes="Predicted default probability exceeds operational threshold 0.35.",
        ),
        explanation=ReportExplanationSummary(
            output_space="log_odds",
            base_value=-5.7170,
            shap_additivity_residual=2.1e-7,
            features=[
                ReportFeatureContribution(
                    name="FOIR_RATIO",
                    label="Fixed Obligation to Income Ratio",
                    value=0.68,
                    unit="ratio",
                    shap_value=5.834,
                    direction="risk_increasing",
                ),
                ReportFeatureContribution(
                    name="LTV_RATIO",
                    label="Loan to Value Ratio",
                    value=0.89,
                    unit="ratio",
                    shap_value=2.352,
                    direction="risk_increasing",
                ),
                ReportFeatureContribution(
                    name="AMT_INCOME_TOTAL",
                    label="Monthly Verified Income",
                    value=70000.0,
                    unit="INR",
                    shap_value=-0.45,
                    direction="risk_reducing",
                ),
            ],
        ),
        counterfactuals=[
            ReportCounterfactualCandidate(
                id="CF-1",
                changes=[
                    {"feature": "AMT_ANNUITY", "current": 45000.0, "proposed": 31500.0, "delta": -13500.0},
                    {"feature": "AMT_GOODS_PRICE", "current": 1685393.0, "proposed": 2022471.6, "delta": 337078.6},
                ],
                derived_recalculations=[
                    {"feature": "FOIR_RATIO", "formula": "AMT_ANNUITY / AMT_INCOME_TOTAL", "previous_value": 0.68, "new_value": 0.49},
                    {"feature": "LTV_RATIO", "formula": "AMT_CREDIT / AMT_GOODS_PRICE", "previous_value": 0.89, "new_value": 0.74},
                ],
                predicted_probability=0.0288,
                target_met=True,
                constraints_passed=["AMT_INCOME_TOTAL: immutable", "AMT_ANNUITY in [5000, 100000]"],
                feasibility_status="feasible",
                caveats=["Model what-if scenario for human review; not an approval commitment."],
            )
        ],
        search_trace=ReportSearchTraceSummary(
            ontology_version="1.0.0",
            features_used=["FOIR_RATIO", "LTV_RATIO"],
            dense_query="FOIR ratio repayment capacity thresholds",
            bm25_terms=["FOIR", "LTV", "repayment"],
            metadata_filters={"jurisdiction": "IN"},
            retrieved_chunk_ids=["SYN-POL-FOIR-001-P1-C1"],
            index_version="INDEX-TEST-V1",
        ),
        evidence=[
            ReportEvidenceItem(
                source_id="SYN-POL-FOIR-001",
                title="SYNTHETIC DEMO POLICY - NOT LAW: FOIR Thresholds",
                issuer="Synthetic Internal Credit Committee",
                jurisdiction="IN",
                source_type="internal",
                version="v1.0",
                effective_date="2026-02-01",
                page=1,
                section="General Provisions",
                quote="The maximum permissible Fixed Obligation to Income Ratio shall not exceed 50 percent.",
                chunk_id="SYN-POL-FOIR-001-P1-C1",
                url="https://demo.internal/policy/foir",
                verification_status="verified",
            )
        ],
        dossier=ReportDossierSummary(
            adverse_action_explanation="Draft decision explanation (for human review): Application risk exceeds policy limits.",
            candidate_review_paths=["Tenor extension to reduce monthly obligation."],
            regulatory_justification="Adverse decision driven by elevated FOIR and LTV.",
            evidence_gaps=[],
            reviewer_notes="Candidate remediation paths require credit committee validation.",
        ),
        verifier=ReportVerifierSummary(
            quote_checks_total=1,
            quote_checks_passed=1,
            quote_checks_failed=0,
            faithfulness_score=1.0,
            normalized_match_policy="Unicode NFKC + ASCII quote/dash unification",
        ),
        human_review_required=True,
    )


def test_pdf_export_generation(tmp_path: Path, sample_audit_report: AuditReport):
    """Verify that PDFExportService generates a multi-page, valid PDF document."""
    service = PDFExportService(output_dir=tmp_path)
    output_pdf = service.export_report_pdf(sample_audit_report)

    assert output_pdf.exists()
    assert output_pdf.stat().st_size > 1000

    # Inspect generated PDF with PyMuPDF
    doc = pymupdf.open(str(output_pdf))
    assert doc.page_count == 3

    # Check text content on page 1
    page1_text = doc[0].get_text()
    assert "CREDIT RISK DECISION AUDIT DOSSIER" in page1_text
    assert "AUDIT-TEST-PDF-001" in page1_text
    assert "APP-10492-SYNTHETIC" in page1_text
    assert "DEMO / SYNTHETIC DATA" in page1_text
    assert "Draft decision explanation" in page1_text
    assert "TreeSHAP" in page1_text

    # Check text content on page 2 (Counterfactuals)
    page2_text = doc[1].get_text()
    assert "CONSTRAINED COUNTERFACTUAL REMEDIATION PATHS" in page2_text
    assert "Candidate Scenario #1" in page2_text

    # Check text content on page 3 (Evidence & Verification)
    page3_text = doc[2].get_text()
    assert "RETRIEVED REGULATORY & POLICY EVIDENCE" in page3_text
    assert "SYN-POL-FOIR-001" in page3_text
    assert "CITATIONS_VERIFIED" in page3_text or "CITATIONS" in page3_text or "FAITHFULNESS" in page3_text
    assert "HUMAN REVIEWER DISPOSITION" in page3_text

    doc.close()
