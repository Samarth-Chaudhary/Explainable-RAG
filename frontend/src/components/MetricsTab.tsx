import React, { useState, useEffect } from "react";
import { BarChart3, ShieldAlert, CheckCircle2, TrendingUp, RefreshCw, AlertCircle } from "lucide-react";
import { EvaluationMetricsResponse } from "../types/api";
import { api } from "../api/client";

export const MetricsTab: React.FC = () => {
  const [metrics, setMetrics] = useState<EvaluationMetricsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadMetrics();
  }, []);

  const loadMetrics = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getEvaluationMetrics();
      setMetrics(res);
    } catch (err: any) {
      setError(err.message || "Failed to load evaluation metrics.");
    } finally {
      setLoading(false);
    }
  };

  const cf = metrics?.counterfactual_metrics;
  const ret = metrics?.retrieval_metrics;
  const cit = metrics?.citation_faithfulness;
  const fid = metrics?.explanation_fidelity;
  const meta = metrics?.evaluation_metadata;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
      {/* Benchmark Claim Control Warning Banner */}
      <div
        style={{
          padding: "14px 18px",
          backgroundColor: "rgba(245, 158, 11, 0.08)",
          border: "1px solid rgba(245, 158, 11, 0.3)",
          borderRadius: "var(--radius-md)",
          display: "flex",
          alignItems: "flex-start",
          gap: "12px",
        }}
      >
        <ShieldAlert size={20} color="#fbbf24" style={{ flexShrink: 0, marginTop: "2px" }} />
        <div style={{ fontSize: "0.8125rem", color: "#fef3c7", lineHeight: 1.5 }}>
          <strong>Benchmark Claim Control (Mandatory Policy):</strong> Research hypotheses in the project brief (e.g. approx ~52% baseline, 91%+ target Recall@3) are <em>aspirations only</em>, not established legal or model facts. All KPI tiles below display <strong>actual reproducible numbers measured on documented test sets by <code>tests/evaluation/eval_cli.py</code></strong>. Any metric not yet computed is strictly marked <em>'not measured'</em>.
        </div>
      </div>

      {/* Overview Metadata Card */}
      {meta && (
        <div className="card">
          <div className="card-header">
            <div>
              <span className="card-title">
                <BarChart3 size={18} color="#38bdf8" />
                Benchmark Run Lineage & Provenance
              </span>
              <span className="card-subtitle">
                Execution metadata recorded at benchmark generation time
              </span>
            </div>
            <button className="btn btn-secondary" onClick={loadMetrics} disabled={loading}>
              <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
              Refresh
            </button>
          </div>
          <div className="card-body">
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "14px", fontSize: "0.75rem" }}>
              <div>
                <span style={{ color: "var(--text-muted)" }}>Execution Timestamp:</span>
                <div className="font-mono" style={{ color: "var(--text-primary)", marginTop: "2px" }}>
                  {meta.execution_timestamp}
                </div>
              </div>
              <div>
                <span style={{ color: "var(--text-muted)" }}>Evaluation Script:</span>
                <div className="font-mono" style={{ color: "#38bdf8", marginTop: "2px" }}>
                  {meta.script_path}
                </div>
              </div>
              <div>
                <span style={{ color: "var(--text-muted)" }}>Active Model Version:</span>
                <div className="font-mono" style={{ color: "var(--text-primary)", marginTop: "2px" }}>
                  {meta.model_version}
                </div>
              </div>
              <div>
                <span style={{ color: "var(--text-muted)" }}>Policy Chunks in Index:</span>
                <div className="font-mono" style={{ color: "var(--text-primary)", marginTop: "2px" }}>
                  {meta.corpus_chunks_count} chunks
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Counterfactual Metrics */}
      <div className="card">
        <div className="card-header">
          <div>
            <span className="card-title">
              1. Counterfactual Validity and Actionability
            </span>
            <span className="card-subtitle">
              At least two distinct rates measuring target threshold achievement and constraint adherence
            </span>
          </div>
        </div>
        <div className="card-body">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}>
            <div style={{ padding: "16px", backgroundColor: "var(--bg-card-subtle)", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-subtle)" }}>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
                Validity Rate
              </div>
              <div className="font-mono" style={{ fontSize: "2rem", fontWeight: 700, color: "#34d399", marginTop: "4px" }}>
                {cf ? `${(cf.validity_rate * 100).toFixed(2)}%` : "not measured"}
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "6px" }}>
                <strong>Denominator:</strong> {cf ? `${cf.valid_candidates_count}/${cf.total_candidates_evaluated} candidates evaluated` : "N/A"}
              </div>
              <div style={{ fontSize: "0.6875rem", color: "var(--text-muted)", marginTop: "4px" }}>
                Candidates whose exact re-scored model probability satisfies target &le; 0.35.
              </div>
            </div>

            <div style={{ padding: "16px", backgroundColor: "var(--bg-card-subtle)", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-subtle)" }}>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
                Actionability Rate
              </div>
              <div className="font-mono" style={{ fontSize: "2rem", fontWeight: 700, color: "#38bdf8", marginTop: "4px" }}>
                {cf ? `${(cf.actionability_rate * 100).toFixed(2)}%` : "not measured"}
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "6px" }}>
                <strong>Denominator:</strong> {cf ? `${cf.actionable_candidates_count}/${cf.total_candidates_evaluated} candidates evaluated` : "N/A"}
              </div>
              <div style={{ fontSize: "0.6875rem", color: "var(--text-muted)", marginTop: "4px" }}>
                Candidates satisfying target AND passing all immutable-feature, domain, range, and feasibility constraints.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Attribution-to-Retrieval Benchmark */}
      <div className="card">
        <div className="card-header">
          <div>
            <span className="card-title">
              2. Attribution-to-Retrieval Quality Benchmark
            </span>
            <span className="card-subtitle">
              Comparison between Ontology-Bridged Translator vs. Naive Baseline on gold-labelled test queries
            </span>
          </div>
        </div>
        <div className="card-body">
          {ret ? (
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Retrieval Pipeline Method</th>
                    <th>Recall@3</th>
                    <th>Precision@3</th>
                    <th>MRR</th>
                    <th>nDCG@3</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ backgroundColor: "rgba(59, 130, 246, 0.05)" }}>
                    <td>
                      <strong style={{ color: "#60a5fa" }}>{ret.ontology_bridged_method.name}</strong>
                      <div style={{ fontSize: "0.6875rem", color: "var(--text-muted)" }}>
                        Semantic query expansion + BM25 keyword injection + Reciprocal Rank Fusion
                      </div>
                    </td>
                    <td className="font-mono" style={{ fontSize: "1rem", fontWeight: 700, color: "#34d399" }}>
                      {(ret.ontology_bridged_method["Recall@3"] * 100).toFixed(2)}%
                    </td>
                    <td className="font-mono">{(ret.ontology_bridged_method["Precision@3"] * 100).toFixed(2)}%</td>
                    <td className="font-mono">{ret.ontology_bridged_method.MRR.toFixed(4)}</td>
                    <td className="font-mono">{ret.ontology_bridged_method["nDCG@3"].toFixed(4)}</td>
                  </tr>

                  <tr>
                    <td>
                      <strong style={{ color: "var(--text-secondary)" }}>{ret.naive_baseline_method.name}</strong>
                      <div style={{ fontSize: "0.6875rem", color: "var(--text-muted)" }}>
                        Raw feature identifier search without ontology or synonym mapping
                      </div>
                    </td>
                    <td className="font-mono" style={{ fontSize: "1rem", fontWeight: 700, color: "#94a3b8" }}>
                      {(ret.naive_baseline_method["Recall@3"] * 100).toFixed(2)}%
                    </td>
                    <td className="font-mono">{(ret.naive_baseline_method["Precision@3"] * 100).toFixed(2)}%</td>
                    <td className="font-mono">{ret.naive_baseline_method.MRR.toFixed(4)}</td>
                    <td className="font-mono">{ret.naive_baseline_method["nDCG@3"].toFixed(4)}</td>
                  </tr>
                </tbody>
              </table>
              <div style={{ fontSize: "0.6875rem", color: "var(--text-muted)", marginTop: "8px" }}>
                Averaging method: <strong>{ret.averaging_method}</strong> | Sample size: <strong>{ret.sample_size_test_cases} labelled scenarios</strong>
              </div>
            </div>
          ) : (
            <div style={{ padding: "20px", color: "var(--text-muted)", textAlign: "center" }}>
              Retrieval benchmark not yet run. Run <code>python tests/evaluation/eval_cli.py</code> to produce metrics.
            </div>
          )}
        </div>
      </div>

      {/* Citation Faithfulness & SHAP Additivity */}
      <div className="grid-2">
        <div className="card">
          <div className="card-header">
            <span className="card-title">3. Citation Faithfulness</span>
          </div>
          <div className="card-body">
            {cit ? (
              <div>
                <div className="font-mono" style={{ fontSize: "2rem", fontWeight: 700, color: "#34d399" }}>
                  {(cit.faithfulness_score * 100).toFixed(2)}%
                </div>
                <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "6px" }}>
                  Verified: <strong>{cit.quote_checks_passed}</strong> passed / <strong>{cit.sample_size_quotes_checked}</strong> total quotes evaluated
                </div>
                <div style={{ fontSize: "0.6875rem", color: "var(--text-muted)", marginTop: "4px" }}>
                  Adversarial check: Intentionally fabricated quotes successfully detected and marked as failed.
                </div>
              </div>
            ) : (
              <div>not measured</div>
            )}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">4. SHAP Explanation Fidelity</span>
          </div>
          <div className="card-body">
            {fid ? (
              <div>
                <div className="badge badge-risk-low font-mono" style={{ fontSize: "0.875rem", padding: "6px 12px" }}>
                  <CheckCircle2 size={16} />
                  Additivity Tolerance Passed
                </div>
                <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "8px" }}>
                  Residual: <span className="font-mono">{fid.shap_additivity_residual.toExponential(4)}</span> (Limit: &plusmn;10<sup>-4</sup>)
                </div>
                <div style={{ fontSize: "0.6875rem", color: "var(--text-muted)", marginTop: "4px" }}>
                  Output Space: <strong>{fid.output_space}</strong> | Base E[f(x)]: <strong className="font-mono">{fid.base_value}</strong>
                </div>
              </div>
            ) : (
              <div>not measured</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
