import React, { useState, useEffect } from "react";
import { User, Play, RefreshCw, AlertCircle, CheckCircle2 } from "lucide-react";
import { api } from "../api/client";
import { ApplicantSummary } from "../types/api";

interface ApplicantWorkspaceProps {
  onRunAudit: (applicantData: Record<string, any>, applicantRef: string) => void;
  isRunning: boolean;
}

export const ApplicantWorkspace: React.FC<ApplicantWorkspaceProps> = ({
  onRunAudit,
  isRunning,
}) => {
  const [applicants, setApplicants] = useState<ApplicantSummary[]>([]);
  const [selectedRef, setSelectedRef] = useState<string>("APP-10492");
  const [features, setFeatures] = useState<Record<string, number>>({
    AMT_CREDIT: 1500000.0,
    AMT_ANNUITY: 45000.0,
    AMT_GOODS_PRICE: 1685393.0,
    AMT_INCOME_TOTAL: 70000.0,
    EXISTING_DEBT_OBLIGATIONS: 2600.0,
    DAYS_EMPLOYED: -1200.0,
    AGE: 38.0,
    CREDIT_BUREAU_SCORE: 620.0,
    PAST_DELINQUENCIES: 2,
  });
  const [dataMode, setDataMode] = useState<string>("demo_synthetic");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadApplicantList();
  }, []);

  const loadApplicantList = async () => {
    try {
      const res = await api.getApplicants();
      setApplicants(res.applicants);
      if (res.applicants.length > 0) {
        setSelectedRef(res.applicants[0].applicant_ref);
        loadApplicantFeatures(res.applicants[0].applicant_ref);
      }
    } catch (err: any) {
      console.error("Failed to load applicants", err);
    }
  };

  const loadApplicantFeatures = async (ref: string) => {
    setLoading(true);
    setError(null);
    try {
      const detail = await api.getApplicant(ref);
      setFeatures(detail.features);
      setDataMode(detail.data_mode);
      setSelectedRef(detail.applicant_ref);
    } catch (err: any) {
      setError(err.message || "Failed to load applicant record");
    } finally {
      setLoading(false);
    }
  };

  const handleFeatureChange = (key: string, val: string) => {
    const num = parseFloat(val);
    setFeatures((prev) => ({
      ...prev,
      [key]: isNaN(num) ? 0 : num,
    }));
  };

  const handleRun = () => {
    onRunAudit(features, selectedRef);
  };

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <span className="card-title">
            <User size={18} color="#38bdf8" />
            Applicant Risk Assessment Workspace
          </span>
          <span className="card-subtitle">
            Configure applicant features or inspect calibrated synthetic fixtures
          </span>
        </div>
        <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
          <span className="badge badge-warning font-mono">
            Mode: {dataMode === "demo_synthetic" ? "Synthetic Fixture" : "Configured Record"}
          </span>
          <button
            className="btn btn-primary"
            onClick={handleRun}
            disabled={isRunning || loading}
            id="btn-run-audit"
          >
            {isRunning ? (
              <>
                <RefreshCw size={14} className="animate-spin" />
                Executing Pipeline...
              </>
            ) : (
              <>
                <Play size={14} />
                Run Full Audit
              </>
            )}
          </button>
        </div>
      </div>

      <div className="card-body">
        {error && (
          <div style={{ padding: "10px 14px", backgroundColor: "var(--risk-high-bg)", border: "1px solid var(--risk-high-border)", borderRadius: "var(--radius-sm)", color: "#fca5a5", marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        <div style={{ marginBottom: "18px", display: "flex", alignItems: "center", gap: "12px" }}>
          <label style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", fontWeight: 600 }}>
            Select Applicant Record:
          </label>
          <select
            style={{
              backgroundColor: "var(--bg-surface)",
              color: "var(--text-primary)",
              border: "1px solid var(--border-medium)",
              padding: "6px 12px",
              borderRadius: "var(--radius-sm)",
              fontFamily: "var(--font-mono)",
              fontSize: "0.8125rem",
            }}
            value={selectedRef}
            onChange={(e) => loadApplicantFeatures(e.target.value)}
          >
            {applicants.map((a) => (
              <option key={a.applicant_ref} value={a.applicant_ref}>
                {a.applicant_ref} — {a.label}
              </option>
            ))}
            <option value="APP-10492">APP-10492 (Scenario B Demo Fixture)</option>
          </select>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: "14px" }}>
          {Object.entries(features).map(([key, val]) => (
            <div key={key} style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              <label style={{ fontSize: "0.6875rem", color: "var(--text-secondary)", fontWeight: 600, textTransform: "uppercase" }}>
                {key}
              </label>
              <input
                type="number"
                style={{
                  backgroundColor: "var(--bg-surface)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "var(--radius-sm)",
                  color: "var(--text-primary)",
                  padding: "6px 10px",
                  fontSize: "0.8125rem",
                  fontFamily: "var(--font-mono)",
                }}
                value={val}
                onChange={(e) => handleFeatureChange(key, e.target.value)}
                disabled={isRunning}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
