"""Integration test running Prediction -> SHAP -> Counterfactuals end-to-end and validating AuditReport contract."""

from datetime import datetime, timezone
from backend.app.cli import run_pipeline
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
from backend.app.services.audit_ledger import AuditLedgerService
from backend.app.services.citation_verifier import CitationVerifierService
from backend.app.services.document_ingestion import DocumentIngestionService
from backend.app.services.export_pdf import PDFExportService
from backend.app.services.ontology_translator import OntologyTranslatorService
from backend.app.services.report_generator import ReportGeneratorService
from backend.app.services.retrieval import HybridRetrievalService


def test_end_to_end_pipeline_execution():
    """Verify end-to-end pipeline execution on synthetic applicant fixture."""
    payload = run_pipeline("data/demo/applicant_10492_synthetic.json")

    assert payload is not None
    assert "prediction" in payload
    assert "explanation" in payload
    assert "counterfactuals" in payload

    # P0 Gate: Mathematical Fidelity
    pred = payload["prediction"]
    exp = payload["explanation"]
    assert exp["output_space"] == "log_odds"
    assert exp["additivity_status"] == "valid"
    assert exp["shap_additivity_residual"] <= exp["tolerance"]

    # P0 Gate: Counterfactual Integrity
    cfs = payload["counterfactuals"]
    assert cfs["no_feasible_found"] is False
    assert len(cfs["candidates"]) >= 1

    cand0 = cfs["candidates"][0]
    assert cand0["target_met"] is True
    assert cand0["predicted_probability"] < 0.35
    assert cand0["feasibility_status"] == "feasible"

    # Verify that FOIR and LTV were recomputed
    recalcs = {r["feature"]: r["new_value"] for r in cand0["derived_recalculations"]}
    assert "FOIR_RATIO" in recalcs
    assert "LTV_RATIO" in recalcs


def test_audit_report_contract_validation():
    """Verify that an AuditReport matching the contract validates strictly."""
    payload = run_pipeline("data/demo/applicant_10492_synthetic.json")
    pred = payload["prediction"]
    exp = payload["explanation"]
    cfs = payload["counterfactuals"]

    # Construct complete AuditReport according to contract schema
    report = AuditReport(
        audit_id="AUDIT-2026-TEST-001",
        applicant_ref="APP-10492",
        generated_at=datetime.now(timezone.utc),
        mode="demo",
        status="complete",
        model=ReportModelSummary(
            name=pred["model"]["name"],
            version=pred["model"]["version"],
            artifact_sha256=pred["model"]["artifact_sha256"],
            calibration_status=pred["model"]["calibration_status"],
        ),
        prediction=ReportPredictionSummary(
            raw_output=pred["raw_output"],
            default_probability=pred["default_probability"],
            threshold=pred["threshold"]["value"],
            label=pred["decision_label"],
            output_notes=pred["output_notes"],
        ),
        explanation=ReportExplanationSummary(
            output_space=exp["output_space"],
            base_value=exp["base_value"],
            shap_additivity_residual=exp["shap_additivity_residual"],
            features=[
                ReportFeatureContribution(
                    name=f["name"],
                    label=f["label"],
                    value=f["value"],
                    unit=f["unit"],
                    shap_value=f["shap_value"],
                    direction=f["direction"],
                )
                for f in exp["features"]
            ],
        ),
        counterfactuals=[
            ReportCounterfactualCandidate(
                id=c["id"],
                changes=c["changes"],
                derived_recalculations=c["derived_recalculations"],
                predicted_probability=c["predicted_probability"],
                target_met=c["target_met"],
                constraints_passed=c["constraints_passed"],
                feasibility_status=c["feasibility_status"],
                caveats=c["caveats"],
            )
            for c in cfs["candidates"]
        ],
        search_trace=ReportSearchTraceSummary(
            ontology_version="1.0.0",
            features_used=["FOIR_RATIO", "LTV_RATIO"],
            dense_query="Applicant FOIR ratio is 0.68; feasible target is 0.49.",
            bm25_terms=["FOIR", "LTV", "collateral", "tenor extension"],
            metadata_filters={"jurisdiction": "IN"},
            retrieved_chunk_ids=["DOC-001-P4-C1", "DOC-002-P12-C3"],
            index_version="1.0.0-synthetic",
        ),
        evidence=[
            ReportEvidenceItem(
                source_id="SRC-POLICY-01",
                title="Illustrative Credit Policy Fixture",
                issuer="Credit Committee",
                jurisdiction="IN",
                source_type="internal_credit_policy",
                version="2.4",
                effective_date="2026-01-01",
                page=12,
                section="Sec 12.4",
                quote="Tenor extension up to 60 months permitted upon committee review.",
                chunk_id="DOC-002-P12-C3",
                url="https://internal.bank.example/credit_policy_v2.4.pdf",
                verification_status="verified",
            )
        ],
        dossier=ReportDossierSummary(
            adverse_action_explanation="Adverse decision driven primarily by elevated debt burden (FOIR) and collateral exposure (LTV).",
            candidate_review_paths=[
                "Path 1: Extend loan tenor to reduce monthly installment, lowering FOIR to 0.49.",
                "Path 2: Pledge additional collateral valued at INR 3.37L, lowering LTV to 0.74.",
            ],
            regulatory_justification="Adverse notice conforms to fair lending disclosure requirements.",
            evidence_gaps=[],
            reviewer_notes="Candidate remediation paths require standard underwriter verification of collateral title.",
        ),
        verifier=ReportVerifierSummary(
            quote_checks_total=1,
            quote_checks_passed=1,
            quote_checks_failed=0,
            faithfulness_score=1.0,
            normalized_match_policy="exact_normalized_tokens",
        ),
        human_review_required=True,
    )

    # Validate report serialization and deserialization
    dumped = report.model_dump()
    reloaded = AuditReport.model_validate(dumped)
    assert reloaded.audit_id == "AUDIT-2026-TEST-001"
    assert reloaded.human_review_required is True


def test_pdf_export_service_renders_valid_dossier(tmp_path):
    """Verify that Part 3 PDFExportService generates a valid compliance PDF."""
    from backend.app.services.export_pdf import PDFExportService
    import fitz

    # Build report from test_audit_report_contract_validation fixture
    payload = run_pipeline("data/demo/applicant_10492_synthetic.json")
    pred = payload["prediction"]
    exp = payload["explanation"]
    cfs = payload["counterfactuals"]

    report = AuditReport(
        audit_id="AUDIT-2026-TEST-PDF",
        applicant_ref="APP-10492",
        generated_at=datetime.now(timezone.utc),
        mode="demo",
        status="complete",
        model=ReportModelSummary(
            name=pred["model"]["name"],
            version=pred["model"]["version"],
            artifact_sha256=pred["model"]["artifact_sha256"],
            calibration_status=pred["model"]["calibration_status"],
        ),
        prediction=ReportPredictionSummary(
            raw_output=pred["raw_output"],
            default_probability=pred["default_probability"],
            threshold=pred["threshold"]["value"],
            label=pred["decision_label"],
            output_notes=pred["output_notes"],
        ),
        explanation=ReportExplanationSummary(
            output_space=exp["output_space"],
            base_value=exp["base_value"],
            shap_additivity_residual=exp["shap_additivity_residual"],
            features=[
                ReportFeatureContribution(
                    name=f["name"],
                    label=f["label"],
                    value=f["value"],
                    unit=f["unit"],
                    shap_value=f["shap_value"],
                    direction=f["direction"],
                )
                for f in exp["features"]
            ],
        ),
        counterfactuals=[
            ReportCounterfactualCandidate(
                id=c["id"],
                changes=c["changes"],
                derived_recalculations=c["derived_recalculations"],
                predicted_probability=c["predicted_probability"],
                target_met=c["target_met"],
                constraints_passed=c["constraints_passed"],
                feasibility_status=c["feasibility_status"],
                caveats=c["caveats"],
            )
            for c in cfs["candidates"]
        ],
        search_trace=ReportSearchTraceSummary(
            ontology_version="1.0.0",
            features_used=["FOIR_RATIO", "LTV_RATIO"],
            dense_query="Applicant FOIR ratio is 0.68; feasible target is 0.49.",
            bm25_terms=["FOIR", "LTV"],
            metadata_filters={"jurisdiction": "IN"},
            retrieved_chunk_ids=["DOC-001"],
            index_version="1.0.0-synthetic",
        ),
        evidence=[],
        dossier=ReportDossierSummary(
            adverse_action_explanation="Adverse decision driven primarily by elevated debt burden (FOIR).",
            candidate_review_paths=["Path 1: Extend loan tenor."],
            regulatory_justification="Adverse notice conforms to requirements.",
            evidence_gaps=[],
            reviewer_notes="Candidate remediation paths require standard underwriter verification.",
        ),
        verifier=ReportVerifierSummary(
            quote_checks_total=0,
            quote_checks_passed=0,
            quote_checks_failed=0,
            faithfulness_score=1.0,
            normalized_match_policy="exact_normalized_tokens",
        ),
        human_review_required=True,
    )

    pdf_service = PDFExportService(output_dir=tmp_path)
    pdf_path = pdf_service.export_report_pdf(report)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000

    doc = fitz.open(pdf_path)
    assert doc.page_count >= 1
    text = doc[0].get_text()
    assert "DEMO / SYNTHETIC DATA" in text
    assert "AUDIT-2026-TEST-PDF" in text

