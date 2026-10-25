import React from "react";
import { FileText, Download, CheckSquare, AlertTriangle, FileCode } from "lucide-react";
import { AuditReport, UserRole } from "../types/api";
import { api } from "../api/client";

interface DossierViewProps {
  report: AuditReport;
  activeRole: UserRole;
  onOpenDispositionModal: () => void;
}

export const DossierView: React.FC<DossierViewProps> = ({
  report,
  activeRole,
  onOpenDispositionModal,
}) => {
  const { dossier, audit_id, applicant_ref, generated_at, mode, status } = report;

  const handleDownloadJson = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(report, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `audit_dossier_${audit_id}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const handleDownloadPdf = async () => {
    try {
      const blob = await api.downloadAuditPdf(audit_id);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `audit_dossier_${audit_id}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      alert(`Failed to download PDF: ${err.message}`);
    }
  };

  const canReview = activeRole === "reviewer" || activeRole === "admin";

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <span className="card-title">
            <FileText size={18} color="#eab308" />
            Decision Audit Dossier & Compliance Synthesis
          </span>
          <span className="card-subtitle">
            Draft Decision Explanation & Candidate Remediation Paths (for human review)
          </span>
        </div>
        <div style={{ display: "flex", gap: "10px" }}>
          <button className="btn btn-secondary" onClick={handleDownloadJson} id="btn-export-json">
            <FileCode size={14} />
            Export JSON
          </button>
          <button className="btn btn-primary" onClick={handleDownloadPdf} id="btn-export-pdf">
            <Download size={14} />
            Export PDF Dossier
          </button>
          {canReview && (
            <button
              className="btn btn-success"
              onClick={onOpenDispositionModal}
              id="btn-record-disposition"
            >
              <CheckSquare size={14} />
              Reviewer Disposition
            </button>
          )}
        </div>
      </div>

      <div className="card-body">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "18px", padding: "10px 14px", backgroundColor: "rgba(255, 255, 255, 0.02)", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-subtle)" }}>
          <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
            Audit ID: <strong className="font-mono" style={{ color: "var(--text-primary)" }}>{audit_id}</strong> | Applicant: <strong className="font-mono">{applicant_ref}</strong>
          </div>
          <div style={{ display: "flex", gap: "8px" }}>
            <span className="badge badge-warning font-mono">Mode: {mode.toUpperCase()}</span>
            <span className="badge badge-info font-mono">Status: {status.toUpperCase()}</span>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {/* Adverse Action Explanation */}
          <div>
            <h4 style={{ fontSize: "0.8125rem", color: "#fca5a5", textTransform: "uppercase", marginBottom: "6px" }}>
              Primary Adverse Action Drivers & Mathematical Attribution
            </h4>
            <div style={{ fontSize: "0.8125rem", lineHeight: 1.6, color: "var(--text-primary)", backgroundColor: "rgba(239, 68, 68, 0.04)", border: "1px solid rgba(239, 68, 68, 0.15)", borderRadius: "var(--radius-sm)", padding: "12px 14px" }}>
              {dossier.adverse_action_explanation}
            </div>
          </div>

          {/* Candidate Remediation Review Paths */}
          <div>
            <h4 style={{ fontSize: "0.8125rem", color: "#86efac", textTransform: "uppercase", marginBottom: "6px" }}>
              Candidate Remediation Paths (Underwriting Review)
            </h4>
            <div style={{ fontSize: "0.8125rem", lineHeight: 1.6, color: "var(--text-primary)", backgroundColor: "rgba(16, 185, 129, 0.04)", border: "1px solid rgba(16, 185, 129, 0.15)", borderRadius: "var(--radius-sm)", padding: "12px 14px" }}>
              <ul style={{ paddingLeft: "20px" }}>
                {dossier.candidate_review_paths.map((p, i) => (
                  <li key={i} style={{ marginBottom: "4px" }}>{p}</li>
                ))}
              </ul>
            </div>
          </div>

          {/* Regulatory Justification */}
          <div>
            <h4 style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", textTransform: "uppercase", marginBottom: "6px" }}>
              Policy & Institutional Grounding Summary
            </h4>
            <div style={{ fontSize: "0.8125rem", lineHeight: 1.6, color: "var(--text-primary)", backgroundColor: "var(--bg-card-subtle)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-sm)", padding: "12px 14px" }}>
              {dossier.regulatory_justification}
            </div>
          </div>

          {/* Evidence Gaps */}
          {dossier.evidence_gaps && dossier.evidence_gaps.length > 0 && (
            <div>
              <h4 style={{ fontSize: "0.8125rem", color: "#fbbf24", textTransform: "uppercase", marginBottom: "6px" }}>
                Evidence Gaps & Unmodelled Factors
              </h4>
              <div style={{ fontSize: "0.75rem", lineHeight: 1.6, color: "#fde68a", backgroundColor: "rgba(245, 158, 11, 0.04)", border: "1px solid rgba(245, 158, 11, 0.15)", borderRadius: "var(--radius-sm)", padding: "10px 14px" }}>
                <ul style={{ paddingLeft: "18px" }}>
                  {dossier.evidence_gaps.map((g, i) => (
                    <li key={i}>{g}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Reviewer Notes */}
          <div>
            <h4 style={{ fontSize: "0.8125rem", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "6px" }}>
              Compliance & Audit Trail Notes
            </h4>
            <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontStyle: "italic" }}>
              {dossier.reviewer_notes}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
