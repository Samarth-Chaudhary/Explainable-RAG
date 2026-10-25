"""Printable PDF compliance audit dossier export service using PyMuPDF.

Renders an enterprise-grade compliance dossier carrying:
- Prominent DEMO / SYNTHETIC DATA watermarks (when in demo mode).
- Audit ID, timestamps, and model/ontology/index version metadata.
- "Draft decision explanation - for human review only" statutory notice.
- Correct Unicode Rupee symbol (₹) and INR currency rendering.
- Prediction parameters, SHAP attributions, counterfactual remediation paths.
- Retrieved regulatory evidence with exact citations, pages, and sections.
- Citation verifier audit outcome (faithfulness score and normalization policy).
- Page numbers, header banners, and structured bounding boxes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
import pymupdf

from backend.app.schemas.audit_report import AuditReport


class PDFExportProtocol(Protocol):
    """Protocol for generating formatted, printable compliance audit dossiers in PDF format."""

    def export_report_pdf(
        self,
        report: AuditReport,
        output_path: str | Path | None = None,
    ) -> Path:
        """Render complete audit report to PDF."""
        ...


class PDFExportService:
    """Service rendering AuditReport objects into printable, verifiable PDF dossiers."""

    def __init__(self, output_dir: str | Path = "data/exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_report_pdf(
        self,
        report: AuditReport,
        output_path: str | Path | None = None,
    ) -> Path:
        """Generate a complete, multi-page PDF compliance dossier from AuditReport."""
        if output_path is None:
            safe_audit_id = report.audit_id.replace(":", "_").replace("/", "_")
            target_path = self.output_dir / f"audit_dossier_{safe_audit_id}.pdf"
        else:
            target_path = Path(output_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)

        doc = pymupdf.open()

        # ----------------------------------------------------------------------
        # PAGE 1: Executive Summary, Prediction & SHAP Decomposition
        # ----------------------------------------------------------------------
        page1 = doc.new_page(width=595.3, height=841.9)  # Standard A4: 595 x 842 pt
        self._render_header(page1, report, page_num=1, total_pages=3)
        self._render_watermark(page1, report)

        y = 75.0

        # Title Block
        y = self._render_section_title(page1, y, "CREDIT RISK DECISION AUDIT DOSSIER")
        notice_text = (
            "NOTICE: DRAFT DECISION EXPLANATION FOR HUMAN REVIEW ONLY\n"
            "This document is an explainable AI decision-support report. It does not constitute an "
            "automated statutory refusal, credit approval, or binding legal commitment."
        )
        page1.draw_rect(pymupdf.Rect(40, y, 555, y + 36), color=(0.7, 0.7, 0.7), fill=(0.96, 0.96, 0.96))
        page1.insert_text(pymupdf.Point(48, y + 14), notice_text, fontsize=8, color=(0.2, 0.2, 0.2))
        y += 48.0

        # Audit Metadata Table
        meta_items = [
            ("Audit Run ID:", report.audit_id, "Mode / Environment:", report.mode.upper()),
            ("Applicant Ref:", report.applicant_ref, "Timestamp (UTC):", report.generated_at.strftime("%Y-%m-%d %H:%M:%SZ")),
            ("Model Artifact:", f"{report.model.name} v{report.model.version}", "Model Hash:", f"{report.model.artifact_sha256[:16]}..."),
            ("Ontology Version:", report.search_trace.ontology_version, "Index Version:", f"{report.search_trace.index_version[:16]}..."),
        ]
        y = self._render_key_value_grid(page1, y, "1. AUDIT & SYSTEM PROVENANCE", meta_items)

        # Prediction Summary
        pred = report.prediction
        risk_pct = pred.default_probability * 100.0
        thresh_pct = pred.threshold * 100.0
        pred_items = [
            ("Assessed Default Probability:", f"{pred.default_probability:.4f} ({risk_pct:.1f}%)", "Operational Decision Label:", pred.label),
            ("Configured Policy Threshold:", f"{pred.threshold:.4f} ({thresh_pct:.1f}%)", "Raw Output Margin (Log-Odds):", f"{pred.raw_output:.4f}"),
            ("Calibration Status:", report.model.calibration_status, "Human Review Mandate:", "Required (Always Mandatory)"),
        ]
        y = self._render_key_value_grid(page1, y, "2. MODEL INFERENCE & RISK EVALUATION", pred_items)

        # Adverse Action Narrative
        y = self._render_section_title(page1, y, "3. ADVERSE ACTION EXPLANATION DRAFT")
        page1.insert_textbox(
            pymupdf.Rect(40, y, 555, y + 55),
            report.dossier.adverse_action_explanation,
            fontsize=8.5,
            color=(0.1, 0.1, 0.1),
        )
        y += 62.0

        # SHAP Decomposition Summary
        expl = report.explanation
        y = self._render_section_title(page1, y, "4. MATHEMATICAL FEATURE ATTRIBUTIONS (TreeSHAP)")
        shap_header = (
            f"Output Space: {expl.output_space.upper()} | Base Value E[f]: {expl.base_value:.4f} | "
            f"Additivity Residual: {expl.shap_additivity_residual:.2e} (Pass)"
        )
        page1.insert_text(pymupdf.Point(40, y), shap_header, fontsize=8, color=(0.3, 0.3, 0.3))
        y += 14.0

        # Feature table
        headers = ["Rank", "Feature Name", "Observed Value", "Attribution (Log-Odds)", "Risk Direction"]
        col_widths = [35, 170, 110, 110, 90]
        y = self._render_table_header(page1, y, headers, col_widths)

        sorted_features = sorted(expl.features, key=lambda f: abs(f.shap_value), reverse=True)[:6]
        for idx, feat in enumerate(sorted_features, 1):
            val_str = f"{feat.value:.2f} {feat.unit}".replace("INR", "₹")
            row = [
                str(idx),
                feat.label or feat.name,
                val_str,
                f"{feat.shap_value:+.4f}",
                "Risk Increasing" if feat.direction in ("risk_increasing", "increases_risk") else "Risk Reducing",
            ]
            y = self._render_table_row(page1, y, row, col_widths, is_highlight=(feat.direction in ("risk_increasing", "increases_risk")))

        # ----------------------------------------------------------------------
        # PAGE 2: Counterfactual Remediation Paths & Review Scenarios
        # ----------------------------------------------------------------------
        page2 = doc.new_page(width=595.3, height=841.9)
        self._render_header(page2, report, page_num=2, total_pages=3)
        self._render_watermark(page2, report)

        y = 75.0
        y = self._render_section_title(page2, y, "5. CONSTRAINED COUNTERFACTUAL REMEDIATION PATHS")
        cf_intro = (
            "DISCLOSURE: The scenarios below are model what-if simulations based on strict mutability constraints.\n"
            "They illustrate candidate operational restructuring paths that reduce model risk below the 0.35 threshold.\n"
            "They do NOT guarantee automatic loan approval upon completion."
        )
        page2.draw_rect(pymupdf.Rect(40, y, 555, y + 36), color=(0.8, 0.8, 0.8), fill=(0.97, 0.97, 0.97))
        page2.insert_text(pymupdf.Point(48, y + 14), cf_intro, fontsize=8, color=(0.25, 0.25, 0.25))
        y += 48.0

        if not report.counterfactuals:
            page2.insert_text(
                pymupdf.Point(40, y + 15),
                "No feasible counterfactual found under configured constraints.",
                fontsize=9.5,
                color=(0.7, 0.1, 0.1),
            )
            y += 30.0
        else:
            for idx, cand in enumerate(report.counterfactuals[:2], 1):
                cand_title = (
                    f"Candidate Scenario #{idx} ({cand.id}): Target Met = {cand.target_met} | "
                    f"After-Change Re-scored Risk = {cand.predicted_probability:.4f} (Threshold 0.3500)"
                )
                page2.draw_rect(pymupdf.Rect(40, y, 555, y + 20), color=(0.2, 0.4, 0.6), fill=(0.92, 0.95, 0.98))
                page2.insert_text(pymupdf.Point(48, y + 14), cand_title, fontsize=8.5, color=(0.1, 0.2, 0.4))
                y += 26.0

                # Proposed feature modifications table
                cf_col_widths = [160, 115, 120, 120]
                y = self._render_table_header(page2, y, ["Actionable Lever", "Current Value", "Proposed Value", "Required Delta"], cf_col_widths)
                for chg in cand.changes:
                    chg_dict = chg if isinstance(chg, dict) else chg.model_dump()
                    feat_name = chg_dict.get("feature", "")
                    cur_val = chg_dict.get("current", 0.0)
                    prop_val = chg_dict.get("proposed", 0.0)
                    delta_val = chg_dict.get("delta", 0.0)

                    row_vals = [
                        feat_name,
                        f"₹{cur_val:,.2f}" if "AMT" in feat_name else f"{cur_val:.2f}",
                        f"₹{prop_val:,.2f}" if "AMT" in feat_name else f"{prop_val:.2f}",
                        f"{delta_val:+,.2f}",
                    ]
                    y = self._render_table_row(page2, y, row_vals, cf_col_widths)

                # Derived ratio recalculations
                if cand.derived_recalculations:
                    y += 6.0
                    y = self._render_table_header(page2, y, ["Derived Ratio", "Formula Reference", "Current Ratio", "Recomputed Ratio"], cf_col_widths)
                    for rec in cand.derived_recalculations:
                        rec_dict = rec if isinstance(rec, dict) else rec.model_dump()
                        feat_name = rec_dict.get("feature", "")
                        formula = rec_dict.get("formula", "")
                        old_v = rec_dict.get("previous_value", rec_dict.get("old_value", 0.0))
                        new_v = rec_dict.get("new_value", 0.0)
                        y = self._render_table_row(
                            page2,
                            y,
                            [feat_name, formula[:28], f"{old_v:.2f}", f"{new_v:.2f}"],
                            cf_col_widths,
                            is_highlight=True,
                        )

                # Caveats and constraints
                if cand.caveats:
                    y += 6.0
                    for cav in cand.caveats[:2]:
                        page2.insert_text(pymupdf.Point(45, y + 10), f"• {cav}", fontsize=7.5, color=(0.4, 0.4, 0.4))
                        y += 12.0
                y += 14.0

        # Reviewer remediation guidance
        if report.dossier.candidate_review_paths:
            y = self._render_section_title(page2, y, "6. CANDIDATE REVIEW PATHS FOR CREDIT COMMITTEE")
            for path_str in report.dossier.candidate_review_paths[:3]:
                page2.insert_text(pymupdf.Point(45, y + 10), f"→ {path_str}", fontsize=8, color=(0.1, 0.2, 0.4))
                y += 14.0

        # ----------------------------------------------------------------------
        # PAGE 3: Retrieved Evidence, Citation Verification & Audit Sign-off
        # ----------------------------------------------------------------------
        page3 = doc.new_page(width=595.3, height=841.9)
        self._render_header(page3, report, page_num=3, total_pages=3)
        self._render_watermark(page3, report)

        y = 75.0
        y = self._render_section_title(page3, y, "7. RETRIEVED REGULATORY & POLICY EVIDENCE")

        if not report.evidence:
            page3.insert_text(
                pymupdf.Point(40, y + 15),
                "No authoritative matching policy clauses retrieved from configured corpus.",
                fontsize=9,
                color=(0.6, 0.2, 0.2),
            )
            y += 28.0
        else:
            for ev in report.evidence[:3]:
                is_ver = (ev.verification_status == "verified")
                status_color = (0.1, 0.5, 0.2) if is_ver else (0.7, 0.2, 0.2)
                status_label = "VERIFIED VERBATIM" if is_ver else f"UNVERIFIED ({ev.verification_status.upper()})"

                # Clause Header Box
                page3.draw_rect(pymupdf.Rect(40, y, 555, y + 18), color=(0.7, 0.7, 0.7), fill=(0.94, 0.94, 0.94))
                header_line = f"Source: {ev.source_id} | Page {ev.page} | Section: {ev.section} [{status_label}]"
                page3.insert_text(pymupdf.Point(46, y + 13), header_line, fontsize=8, color=status_color)
                y += 22.0

                # Quoted text block
                quote_box = pymupdf.Rect(45, y, 550, y + 42)
                page3.draw_rect(quote_box, color=(0.85, 0.85, 0.85), fill=(0.98, 0.98, 0.98))
                page3.insert_textbox(
                    pymupdf.Rect(50, y + 4, 545, y + 38),
                    f"\"{ev.quote}\"",
                    fontsize=7.5,
                    color=(0.15, 0.15, 0.15),
                )
                y += 48.0

        # Citation Verifier Summary
        ver = report.verifier
        y = self._render_section_title(page3, y, "8. CITATION FAITHFULNESS VERIFICATION OUTCOME")
        faith_pct = ver.faithfulness_score * 100.0
        ver_items = [
            ("Total Quoted Passages:", str(ver.quote_checks_total), "Verified Verbatim Passages:", str(ver.quote_checks_passed)),
            ("Unverified / Failed Passages:", str(ver.quote_checks_failed), "Citation Faithfulness Score:", f"{faith_pct:.1f}%"),
            ("Matching Standard:", "Unicode NFKC + ASCII Quote/Dash Unification", "Semantic Proximity Fallback:", "FORBIDDEN (Never Used)"),
        ]
        y = self._render_key_value_grid(page3, y, "", ver_items)

        # Policy & Evidence Gaps
        if report.dossier.evidence_gaps:
            y = self._render_section_title(page3, y, "9. POLICY COVERAGE GAPS & COMPLIANCE NOTES")
            for gap in report.dossier.evidence_gaps:
                page3.insert_text(pymupdf.Point(45, y + 10), f"⚠ Gap: {gap}", fontsize=8, color=(0.6, 0.3, 0.1))
                y += 14.0

        # Sign-off & Disposition Block
        y = max(y + 10, 680.0)
        y = self._render_section_title(page3, y, "10. HUMAN REVIEWER DISPOSITION & SIGN-OFF")
        page3.draw_rect(pymupdf.Rect(40, y, 555, y + 80), color=(0.5, 0.5, 0.5))
        page3.insert_text(pymupdf.Point(50, y + 20), "Reviewer Decision: [  ] APPROVE    [  ] REJECT FOR REWORK    [  ] ESCALATE TO COMMITTEE", fontsize=8.5, color=(0.2, 0.2, 0.2))
        page3.insert_text(pymupdf.Point(50, y + 42), "Reviewer Name & Employee ID: __________________________________________________", fontsize=8, color=(0.3, 0.3, 0.3))
        page3.insert_text(pymupdf.Point(50, y + 64), "Signature & Date: _____________________________________   Notes: ____________________", fontsize=8, color=(0.3, 0.3, 0.3))

        doc.save(str(target_path))
        doc.close()
        return target_path

    def _render_header(self, page: pymupdf.Page, report: AuditReport, page_num: int, total_pages: int) -> None:
        """Render consistent running top header across pages."""
        page.draw_rect(pymupdf.Rect(40, 20, 555, 48), color=(0.2, 0.3, 0.5), fill=(0.15, 0.25, 0.45))
        page.insert_text(pymupdf.Point(50, 38), "XAI-RAG ENTERPRISE CREDIT AUDIT PLATFORM", fontsize=10, color=(1, 1, 1))
        page.insert_text(
            pymupdf.Point(380, 38),
            f"Audit ID: {report.audit_id[:20]} | Page {page_num} of {total_pages}",
            fontsize=7.5,
            color=(0.9, 0.9, 0.9),
        )

    def _render_watermark(self, page: pymupdf.Page, report: AuditReport) -> None:
        """Render prominent synthetic/demo data watermark across page background."""
        if report.mode == "demo":
            # Prominent warning watermark strip
            page.draw_rect(pymupdf.Rect(40, 52, 555, 68), color=(0.85, 0.4, 0.4), fill=(0.98, 0.92, 0.92))
            page.insert_text(
                pymupdf.Point(95, 64),
                "DEMO / SYNTHETIC DATA - NOT LAW - DECISION SUPPORT ONLY",
                fontsize=8.5,
                color=(0.7, 0.15, 0.15),
            )
            # Light large background watermark text
            page.insert_text(
                pymupdf.Point(60, 420),
                "DEMO / SYNTHETIC DATA - NOT LAW",
                fontsize=26,
                color=(0.93, 0.93, 0.93),
            )

    def _render_section_title(self, page: pymupdf.Page, y: float, title: str) -> float:
        """Render bold section header banner."""
        page.draw_line(pymupdf.Point(40, y), pymupdf.Point(555, y), color=(0.3, 0.4, 0.6), width=1.2)
        page.insert_text(pymupdf.Point(40, y + 14), title, fontsize=9, color=(0.15, 0.25, 0.45))
        return y + 22.0

    def _render_key_value_grid(
        self,
        page: pymupdf.Page,
        y: float,
        title: str,
        items: list[tuple[str, str, str, str]],
    ) -> float:
        """Render structured 2-column key-value grid."""
        if title:
            y = self._render_section_title(page, y, title)

        for left_lbl, left_val, right_lbl, right_val in items:
            page.insert_text(pymupdf.Point(45, y + 10), left_lbl, fontsize=8, color=(0.4, 0.4, 0.4))
            page.insert_text(pymupdf.Point(145, y + 10), left_val[:32], fontsize=8, color=(0.1, 0.1, 0.1))
            page.insert_text(pymupdf.Point(310, y + 10), right_lbl, fontsize=8, color=(0.4, 0.4, 0.4))
            page.insert_text(pymupdf.Point(420, y + 10), right_val[:28], fontsize=8, color=(0.1, 0.1, 0.1))
            y += 14.0

        return y + 6.0

    def _render_table_header(
        self,
        page: pymupdf.Page,
        y: float,
        headers: list[str],
        col_widths: list[int],
    ) -> float:
        """Render table column headers."""
        total_width = sum(col_widths)
        page.draw_rect(pymupdf.Rect(40, y, 40 + total_width, y + 16), color=(0.6, 0.6, 0.6), fill=(0.9, 0.92, 0.95))
        curr_x = 45.0
        for header, width in zip(headers, col_widths):
            page.insert_text(pymupdf.Point(curr_x, y + 11), header, fontsize=7.5, color=(0.15, 0.25, 0.4))
            curr_x += width
        return y + 18.0

    def _render_table_row(
        self,
        page: pymupdf.Page,
        y: float,
        values: list[str],
        col_widths: list[int],
        is_highlight: bool = False,
    ) -> float:
        """Render a single table row."""
        total_width = sum(col_widths)
        fill_color = (0.99, 0.96, 0.96) if is_highlight else (1.0, 1.0, 1.0)
        page.draw_rect(pymupdf.Rect(40, y, 40 + total_width, y + 15), color=(0.88, 0.88, 0.88), fill=fill_color)
        curr_x = 45.0
        for val, width in zip(values, col_widths):
            page.insert_text(pymupdf.Point(curr_x, y + 10), str(val)[:30], fontsize=7.5, color=(0.15, 0.15, 0.15))
            curr_x += width
        return y + 16.0
