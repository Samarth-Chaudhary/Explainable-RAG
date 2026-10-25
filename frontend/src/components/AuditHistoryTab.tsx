import React, { useState, useEffect } from "react";
import { History, Eye, Download, RefreshCw, AlertCircle } from "lucide-react";
import { AuditRecordSummary } from "../types/api";
import { api } from "../api/client";

interface AuditHistoryTabProps {
  onSelectAudit: (auditId: string) => void;
}

export const AuditHistoryTab: React.FC<AuditHistoryTabProps> = ({
  onSelectAudit,
}) => {
  const [audits, setAudits] = useState<AuditRecordSummary[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadAudits();
  }, []);

  const loadAudits = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.listAudits(50, 0);
      setAudits(res.audits);
    } catch (err: any) {
      setError(err.message || "Failed to load audit history.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPdf = async (auditId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      const blob = await api.downloadAuditPdf(auditId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `audit_${auditId}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      alert(`Download failed: ${err.message}`);
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <span className="card-title">
            <History size={18} color="#38bdf8" />
            Immutable Audit Ledger & Historical Records
          </span>
          <span className="card-subtitle">
            Cryptographically chained and persisted credit decision audit trail
          </span>
        </div>
        <button className="btn btn-secondary" onClick={loadAudits} disabled={loading}>
          <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          Refresh
        </button>
      </div>

      <div className="card-body">
        {error && (
          <div style={{ padding: "10px 14px", backgroundColor: "var(--risk-high-bg)", border: "1px solid var(--risk-high-border)", borderRadius: "var(--radius-sm)", color: "#fca5a5", marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        {audits.length === 0 && !loading ? (
          <div style={{ padding: "30px", textAlign: "center", color: "var(--text-muted)" }}>
            No prior audits found in persistent ledger. Run an audit in the Workspace tab to record the first event.
          </div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Audit ID</th>
                  <th>Applicant</th>
                  <th>Status</th>
                  <th>Model / Version</th>
                  <th>Risk Probability</th>
                  <th>Decision Label</th>
                  <th>Generated Timestamp</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {audits.map((a) => (
                  <tr
                    key={a.audit_id}
                    style={{ cursor: "pointer" }}
                    onClick={() => onSelectAudit(a.audit_id)}
                  >
                    <td className="font-mono" style={{ fontWeight: 600, color: "#60a5fa" }}>
                      {a.audit_id}
                    </td>
                    <td className="font-mono">{a.applicant_ref}</td>
                    <td>
                      <span className="badge badge-info font-mono">{a.status}</span>
                    </td>
                    <td style={{ fontSize: "0.75rem" }}>
                      {a.model_name} <span className="font-mono text-muted">v{a.model_version}</span>
                    </td>
                    <td className="font-mono" style={{ fontWeight: 600 }}>
                      {a.default_probability != null
                        ? `${(a.default_probability * 100).toFixed(2)}%`
                        : "N/A"}
                    </td>
                    <td>
                      {a.decision_label ? (
                        <span
                          className={`badge ${
                            a.decision_label.includes("REJECT")
                              ? "badge-risk-high"
                              : "badge-risk-low"
                          }`}
                        >
                          {a.decision_label}
                        </span>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                      {a.generated_at ? new Date(a.generated_at).toLocaleString() : "N/A"}
                    </td>
                    <td>
                      <div style={{ display: "flex", gap: "6px" }}>
                        <button
                          className="btn btn-secondary"
                          style={{ padding: "3px 8px", fontSize: "0.6875rem" }}
                          onClick={() => onSelectAudit(a.audit_id)}
                          title="Reopen in Workspace"
                        >
                          <Eye size={12} />
                          Inspect
                        </button>
                        <button
                          className="btn btn-secondary"
                          style={{ padding: "3px 8px", fontSize: "0.6875rem" }}
                          onClick={(e) => handleDownloadPdf(a.audit_id, e)}
                          title="Export PDF"
                        >
                          <Download size={12} />
                          PDF
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
