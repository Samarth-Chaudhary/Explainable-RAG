import React from "react";
import { BarChart2, CheckCircle2, AlertTriangle } from "lucide-react";
import { ExplanationSummary } from "../types/api";

interface ShapAttributionCardProps {
  explanation: ExplanationSummary;
}

export const ShapAttributionCard: React.FC<ShapAttributionCardProps> = ({
  explanation,
}) => {
  const { output_space, base_value, shap_additivity_residual, features } = explanation;

  // Max absolute SHAP value for scaling bars
  const maxAbsShap = Math.max(...features.map((f) => Math.abs(f.shap_value)), 0.1);
  const isAdditivityValid = Math.abs(shap_additivity_residual) <= 1e-4;

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <span className="card-title">
            <BarChart2 size={18} color="#60a5fa" />
            TreeSHAP Local Feature Attribution
          </span>
          <span className="card-subtitle">
            Local feature contributions in declared output space: <strong className="font-mono">{output_space}</strong> (never mixed with probabilities)
          </span>
        </div>
        <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
          <span className="badge badge-neutral font-mono">
            Base E[f(x)]: {base_value.toFixed(4)}
          </span>
          <span
            className={`badge ${isAdditivityValid ? "badge-risk-low" : "badge-risk-high"} font-mono`}
          >
            {isAdditivityValid ? <CheckCircle2 size={12} /> : <AlertTriangle size={12} />}
            Residual: {shap_additivity_residual.toExponential(2)}
          </span>
        </div>
      </div>

      <div className="card-body">
        <div style={{ marginBottom: "12px", display: "flex", justifyContent: "space-between", fontSize: "0.6875rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
          <span>Feature & Current Value</span>
          <span>&larr; Risk Reducing | Risk Increasing &rarr;</span>
          <span>Attribution</span>
        </div>

        <div>
          {features.map((f) => {
            const isPositive = f.shap_value > 0;
            const barWidthPercent = Math.min((Math.abs(f.shap_value) / maxAbsShap) * 48, 48);

            return (
              <div key={f.name} className="shap-row">
                <div className="shap-name" title={f.label || f.name}>
                  {f.label || f.name}
                </div>
                <div className="shap-value">
                  {typeof f.value === "number" ? f.value.toLocaleString() : f.value}{" "}
                  <span style={{ color: "var(--text-muted)", fontSize: "0.6875rem" }}>{f.unit}</span>
                </div>

                <div className="shap-bar-container">
                  <div className="shap-bar-center" />
                  <div
                    className={`shap-bar ${isPositive ? "positive" : "negative"}`}
                    style={{
                      width: `${barWidthPercent}%`,
                    }}
                  />
                </div>

                <div
                  className="shap-contrib-val"
                  style={{ color: isPositive ? "#f87171" : "#34d399" }}
                >
                  {isPositive ? `+${f.shap_value.toFixed(3)}` : f.shap_value.toFixed(3)}
                </div>
              </div>
            );
          })}
        </div>

        <div style={{ marginTop: "14px", fontSize: "0.75rem", color: "var(--text-muted)", borderTop: "1px solid var(--border-subtle)", paddingTop: "10px" }}>
          <strong>Mathematical Additivity Check:</strong> Prediction margin reconstructed via{" "}
          <code className="font-mono">f(x) = Base + &sum; SHAP_i</code> within tolerance &plusmn;10<sup>-4</sup>. Probability is derived monotonically via logistic link function and never decomposed directly.
        </div>
      </div>
    </div>
  );
};
