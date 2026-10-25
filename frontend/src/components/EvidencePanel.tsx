import React, { useState } from "react";
import { BookOpen, CheckCircle2, XCircle, AlertCircle, ExternalLink, ShieldCheck } from "lucide-react";
import { EvidenceItem, VerifierSummary } from "../types/api";

interface EvidencePanelProps {
  evidence: EvidenceItem[];
  verifier: VerifierSummary;
}

export const EvidencePanel: React.FC<EvidencePanelProps> = ({
  evidence,
  verifier,
}) => {
  const [selectedItem, setSelectedItem] = useState<EvidenceItem | null>(null);

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <span className="card-title">
            <BookOpen size={18} color="#34d399" />
            Retrieved Authoritative Policy Evidence & Verification
          </span>
          <span className="card-subtitle">
            Exact excerpts retrieved from versioned policy corpus and verified via deterministic substring matching
          </span>
        </div>
        <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
          <span
            className={`badge ${
              verifier.quote_checks_failed === 0 ? "badge-risk-low" : "badge-warning"
            } font-mono`}
          >
            <ShieldCheck size={12} />
            Faithfulness: {(verifier.faithfulness_score * 100).toFixed(1)}% (
            {verifier.quote_checks_passed}/{verifier.quote_checks_total})
          </span>
        </div>
      </div>

      <div className="card-body">
        {evidence.length === 0 ? (
          <div style={{ padding: "20px", textAlign: "center", color: "var(--text-muted)" }}>
            <AlertCircle size={24} style={{ margin: "0 auto 8px" }} />
            <p>Evidence Gap: No applicable authoritative policy clauses retrieved for current risk drivers.</p>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            {evidence.map((item, idx) => {
              const isVerified = item.verification_status === "verified";
              const isFailed = item.verification_status === "failed";

              return (
                <div
                  key={item.chunk_id || idx}
                  style={{
                    backgroundColor: "var(--bg-card-subtle)",
                    border: "1px solid var(--border-subtle)",
                    borderRadius: "var(--radius-sm)",
                    padding: "14px",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "8px" }}>
                    <div>
                      <div style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--text-primary)" }}>
                        {item.title}
                      </div>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "2px" }}>
                        Issuer: <strong>{item.issuer}</strong> | Jurisdiction: <strong className="font-mono">{item.jurisdiction}</strong> | Version: <span className="font-mono">{item.version}</span> | Effective: {item.effective_date}
                      </div>
                    </div>

                    <div style={{ display: "flex", gap: "6px" }}>
                      <span className="badge badge-neutral font-mono">
                        P.{item.page} &sect; {item.section}
                      </span>
                      <span
                        className={`badge ${
                          isVerified ? "badge-risk-low" : isFailed ? "badge-risk-high" : "badge-warning"
                        }`}
                      >
                        {isVerified && <CheckCircle2 size={12} />}
                        {isFailed && <XCircle size={12} />}
                        {!isVerified && !isFailed && <AlertCircle size={12} />}
                        {item.verification_status.toUpperCase()}
                      </span>
                    </div>
                  </div>

                  <div className="quote-box">
                    &ldquo;{item.quote}&rdquo;
                  </div>

                  <div className="quote-meta">
                    <span>Source Family: <strong style={{ color: "var(--text-secondary)" }}>{item.source_type}</strong></span>
                    <span>&bull;</span>
                    <span className="font-mono">Chunk: {item.chunk_id}</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        <div style={{ marginTop: "14px", fontSize: "0.6875rem", color: "var(--text-muted)", borderTop: "1px solid var(--border-subtle)", paddingTop: "10px" }}>
          <strong>Exact Match Verification Policy:</strong> {verifier.normalized_match_policy}
        </div>
      </div>
    </div>
  );
};
