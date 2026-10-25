import React from "react";
import { AlertOctagon, CheckCircle2, Info, Activity } from "lucide-react";
import { ModelSummary, PredictionSummary } from "../types/api";

interface PredictionSummaryCardProps {
  prediction: PredictionSummary;
  model: ModelSummary;
  generatedAt: string;
}

export const PredictionSummaryCard: React.FC<PredictionSummaryCardProps> = ({
  prediction,
  model,
  generatedAt,
}) => {
  const isHighRisk = prediction.default_probability >= prediction.threshold;
  const probPercent = (prediction.default_probability * 100).toFixed(2);
  const thresholdPercent = (prediction.threshold * 100).toFixed(1);

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <span className="card-title">
            <Activity size={18} color="#f59e0b" />
            Deterministic Model Inference
          </span>
          <span className="card-subtitle">
            Source of Truth: XGBoost production inference pipeline (verbatim output)
          </span>
        </div>
        <div
          className={`badge ${isHighRisk ? "badge-risk-high" : "badge-risk-low"}`}
          style={{ padding: "6px 12px", fontSize: "0.8125rem" }}
        >
          {isHighRisk ? <AlertOctagon size={14} /> : <CheckCircle2 size={14} />}
          {prediction.label}
        </div>
      </div>

      <div className="card-body">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "20px" }}>
          <div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
              Probability of Default
            </div>
            <div
              className="font-mono"
              style={{
                fontSize: "2rem",
                fontWeight: 700,
                color: isHighRisk ? "#f87171" : "#34d399",
                lineHeight: 1.2,
                marginTop: "4px",
              }}
            >
              {probPercent}%
            </div>
            <div style={{ fontSize: "0.6875rem", color: "var(--text-secondary)", marginTop: "4px" }}>
              Raw Margin (Log-Odds): <span className="font-mono">{prediction.raw_output.toFixed(4)}</span>
            </div>
          </div>

          <div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
              Operational Threshold
            </div>
            <div
              className="font-mono"
              style={{
                fontSize: "2rem",
                fontWeight: 700,
                color: "#94a3b8",
                lineHeight: 1.2,
                marginTop: "4px",
              }}
            >
              {thresholdPercent}%
            </div>
            <div style={{ fontSize: "0.6875rem", color: "var(--text-secondary)", marginTop: "4px" }}>
              Institutional risk boundary: {prediction.threshold}
            </div>
          </div>

          <div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
              Model Governance
            </div>
            <div style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--text-primary)", marginTop: "6px" }}>
              {model.name} <span className="font-mono" style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>v{model.version}</span>
            </div>
            <div style={{ fontSize: "0.6875rem", color: "var(--text-secondary)", marginTop: "4px" }}>
              Calibration: <span className="font-mono badge badge-neutral" style={{ padding: "1px 6px" }}>{model.calibration_status}</span>
            </div>
            <div style={{ fontSize: "0.625rem", color: "var(--text-muted)", marginTop: "4px", fontFamily: "var(--font-mono)" }}>
              Digest: {model.artifact_sha256.slice(0, 16)}...
            </div>
          </div>
        </div>

        <div style={{ marginTop: "16px", padding: "10px 14px", backgroundColor: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-sm)", fontSize: "0.75rem", color: "var(--text-secondary)", display: "flex", gap: "8px" }}>
          <Info size={16} color="#38bdf8" style={{ flexShrink: 0, marginTop: "2px" }} />
          <div>
            <strong>Operational Policy Disclaimer:</strong> Decision-policy thresholds are configurable risk parameters established by institutional risk management and <em>do not constitute universal statutory or regulatory mandates</em>.
          </div>
        </div>
      </div>
    </div>
  );
};
