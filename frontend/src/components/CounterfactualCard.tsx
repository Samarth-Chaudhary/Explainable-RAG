import React, { useState } from "react";
import { Sliders, CheckCircle2, XCircle, AlertCircle, ArrowRight } from "lucide-react";
import { CounterfactualCandidate } from "../types/api";

interface CounterfactualCardProps {
  counterfactuals: CounterfactualCandidate[];
}

export const CounterfactualCard: React.FC<CounterfactualCardProps> = ({
  counterfactuals,
}) => {
  const [activeTab, setActiveTab] = useState<number>(0);

  if (!counterfactuals || counterfactuals.length === 0) {
    return (
      <div className="card">
        <div className="card-header">
          <span className="card-title">
            <Sliders size={18} color="#a855f7" />
            Constrained Counterfactual Remediation Paths
          </span>
        </div>
        <div className="card-body">
          <div style={{ padding: "20px", textAlign: "center", color: "#fca5a5", backgroundColor: "var(--risk-high-bg)", borderRadius: "var(--radius-sm)" }}>
            <AlertCircle size={24} style={{ margin: "0 auto 8px" }} />
            <p style={{ fontWeight: 600 }}>No feasible counterfactual found under configured constraints</p>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "4px" }}>
              All evaluated perturbation points violated actionable domain constraints, immutability rules, or failed to achieve the target risk threshold.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const candidate = counterfactuals[activeTab] || counterfactuals[0];
  const targetMet = candidate.target_met;
  const isFeasible = candidate.feasibility_status === "feasible";

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <span className="card-title">
            <Sliders size={18} color="#a855f7" />
            Constrained Counterfactual Remediation Paths
          </span>
          <span className="card-subtitle">
            Feasible what-if candidate adjustments satisfying business and underwriting constraints
          </span>
        </div>
        <div style={{ display: "flex", gap: "8px" }}>
          {counterfactuals.map((cf, idx) => (
            <button
              key={cf.id}
              className={`btn ${activeTab === idx ? "btn-primary" : "btn-secondary"}`}
              style={{ padding: "4px 10px", fontSize: "0.75rem" }}
              onClick={() => setActiveTab(idx)}
            >
              Candidate {cf.id}
            </button>
          ))}
        </div>
      </div>

      <div className="card-body">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px", padding: "12px 16px", backgroundColor: "rgba(255, 255, 255, 0.02)", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-subtle)" }}>
          <div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
              Re-Scored Default Probability
            </div>
            <div className="font-mono" style={{ fontSize: "1.5rem", fontWeight: 700, color: targetMet ? "#34d399" : "#f87171" }}>
              {(candidate.predicted_probability * 100).toFixed(2)}%
            </div>
          </div>

          <div style={{ display: "flex", gap: "10px" }}>
            <span className={`badge ${targetMet ? "badge-risk-low" : "badge-risk-high"}`}>
              {targetMet ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
              Target Threshold Met: {targetMet ? "YES" : "NO"}
            </span>
            <span className={`badge ${isFeasible ? "badge-risk-low" : "badge-warning"}`}>
              Status: {candidate.feasibility_status}
            </span>
          </div>
        </div>

        {/* Changes Table */}
        <div style={{ marginBottom: "18px" }}>
          <h4 style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", marginBottom: "8px", textTransform: "uppercase" }}>
            Actionable Primitive Modifications
          </h4>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Actionable Lever</th>
                  <th>Observed Value</th>
                  <th>Proposed Target</th>
                  <th>Net Delta</th>
                </tr>
              </thead>
              <tbody>
                {candidate.changes.map((ch) => (
                  <tr key={ch.feature}>
                    <td style={{ fontWeight: 600 }}>{ch.feature}</td>
                    <td className="font-mono">{ch.current.toLocaleString()}</td>
                    <td className="font-mono" style={{ color: "#38bdf8", fontWeight: 600 }}>
                      {ch.proposed.toLocaleString()}
                    </td>
                    <td className="font-mono" style={{ color: ch.delta < 0 ? "#34d399" : "#f87171" }}>
                      {ch.delta > 0 ? `+${ch.delta.toLocaleString()}` : ch.delta.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Derived recalculations */}
        {candidate.derived_recalculations && candidate.derived_recalculations.length > 0 && (
          <div style={{ marginBottom: "18px" }}>
            <h4 style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", marginBottom: "8px", textTransform: "uppercase" }}>
              Recomputed Derived Financial Ratios (Mathematical Integrity)
            </h4>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Derived Ratio</th>
                    <th>Previous Value</th>
                    <th>Recomputed Value</th>
                    <th>Formula Applied</th>
                  </tr>
                </thead>
                <tbody>
                  {candidate.derived_recalculations.map((dr) => (
                    <tr key={dr.feature}>
                      <td style={{ fontWeight: 600 }}>{dr.feature}</td>
                      <td className="font-mono">{(dr.previous_value * 100).toFixed(2)}%</td>
                      <td className="font-mono" style={{ color: "#34d399", fontWeight: 600 }}>
                        {(dr.new_value * 100).toFixed(2)}%
                      </td>
                      <td style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                        <code className="font-mono">{dr.formula}</code>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Constraints passed and caveats */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px", marginTop: "14px" }}>
          <div style={{ padding: "10px 14px", backgroundColor: "rgba(16, 185, 129, 0.05)", border: "1px solid rgba(16, 185, 129, 0.2)", borderRadius: "var(--radius-sm)", fontSize: "0.75rem" }}>
            <strong style={{ color: "#34d399" }}>Constraints Satisfied:</strong>
            <ul style={{ paddingLeft: "18px", marginTop: "4px", color: "var(--text-secondary)" }}>
              {candidate.constraints_passed.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          </div>

          <div style={{ padding: "10px 14px", backgroundColor: "rgba(245, 158, 11, 0.05)", border: "1px solid rgba(245, 158, 11, 0.2)", borderRadius: "var(--radius-sm)", fontSize: "0.75rem" }}>
            <strong style={{ color: "#fbbf24" }}>Mandatory Disclosures & Caveats:</strong>
            <ul style={{ paddingLeft: "18px", marginTop: "4px", color: "var(--text-secondary)" }}>
              {candidate.caveats.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
