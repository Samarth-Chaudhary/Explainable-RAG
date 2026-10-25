import React, { useState } from "react";
import { X, CheckSquare, AlertTriangle } from "lucide-react";
import { api } from "../api/client";

interface ReviewerDispositionModalProps {
  auditId: string;
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export const ReviewerDispositionModal: React.FC<ReviewerDispositionModalProps> = ({
  auditId,
  isOpen,
  onClose,
  onSuccess,
}) => {
  const [disposition, setDisposition] = useState<string>("approve");
  const [notes, setNotes] = useState<string>("");
  const [reviewerId, setReviewerId] = useState<string>("CREDIT-OFFICER-001");
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!notes.trim()) {
      setError("Reviewer rationale notes are mandatory.");
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      await api.recordDisposition(auditId, {
        disposition,
        notes,
        reviewer_id: reviewerId,
      });
      alert(`Disposition '${disposition}' successfully recorded on tamper-evident ledger!`);
      onSuccess();
      onClose();
    } catch (err: any) {
      setError(err.message || "Failed to record disposition.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(0, 0, 0, 0.75)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 2000,
      }}
    >
      <div
        className="card"
        style={{
          width: "100%",
          maxWidth: "520px",
          boxShadow: "var(--shadow-lg)",
          border: "1px solid var(--border-medium)",
        }}
      >
        <div className="card-header">
          <span className="card-title">
            <CheckSquare size={18} color="#34d399" />
            Record Formal Reviewer Disposition
          </span>
          <button
            onClick={onClose}
            style={{ background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer" }}
          >
            <X size={18} />
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            {error && (
              <div
                style={{
                  padding: "8px 12px",
                  backgroundColor: "var(--risk-high-bg)",
                  border: "1px solid var(--risk-high-border)",
                  borderRadius: "var(--radius-sm)",
                  color: "#fca5a5",
                  fontSize: "0.75rem",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                }}
              >
                <AlertTriangle size={14} />
                <span>{error}</span>
              </div>
            )}

            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>
                Audit Target Reference
              </label>
              <input
                type="text"
                value={auditId}
                disabled
                style={{
                  width: "100%",
                  marginTop: "4px",
                  padding: "8px 10px",
                  backgroundColor: "var(--bg-surface)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "var(--radius-sm)",
                  color: "var(--text-muted)",
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.8125rem",
                }}
              />
            </div>

            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>
                Credit Reviewer ID
              </label>
              <input
                type="text"
                value={reviewerId}
                onChange={(e) => setReviewerId(e.target.value)}
                required
                style={{
                  width: "100%",
                  marginTop: "4px",
                  padding: "8px 10px",
                  backgroundColor: "var(--bg-surface)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "var(--radius-sm)",
                  color: "var(--text-primary)",
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.8125rem",
                }}
              />
            </div>

            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>
                Decision Action
              </label>
              <select
                value={disposition}
                onChange={(e) => setDisposition(e.target.value)}
                style={{
                  width: "100%",
                  marginTop: "4px",
                  padding: "8px 10px",
                  backgroundColor: "var(--bg-surface)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "var(--radius-sm)",
                  color: "var(--text-primary)",
                  fontSize: "0.8125rem",
                }}
              >
                <option value="approve">Approve under candidate counterfactual restructuring</option>
                <option value="reject-for-rework">Reject for applicant rework (counterfactual terms)</option>
                <option value="escalate">Escalate to Senior Credit Committee</option>
              </select>
            </div>

            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>
                Reviewer Rationale & Justification Notes
              </label>
              <textarea
                rows={4}
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Document policy clauses considered, rationale for candidate path selection, and risk mitigating factors..."
                required
                style={{
                  width: "100%",
                  marginTop: "4px",
                  padding: "8px 10px",
                  backgroundColor: "var(--bg-surface)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "var(--radius-sm)",
                  color: "var(--text-primary)",
                  fontSize: "0.8125rem",
                  fontFamily: "var(--font-sans)",
                }}
              />
            </div>
          </div>

          <div
            style={{
              padding: "14px 20px",
              borderTop: "1px solid var(--border-subtle)",
              display: "flex",
              justifyContent: "flex-end",
              gap: "10px",
              backgroundColor: "rgba(255, 255, 255, 0.02)",
            }}
          >
            <button type="button" className="btn btn-secondary" onClick={onClose} disabled={submitting}>
              Cancel
            </button>
            <button type="submit" className="btn btn-success" disabled={submitting}>
              {submitting ? "Appending Event..." : "Record on Ledger"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
