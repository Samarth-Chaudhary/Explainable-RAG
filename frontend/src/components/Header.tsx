import React from "react";
import { Shield, FileText, Database, BarChart3, UserCheck, Layers } from "lucide-react";
import { UserRole } from "../types/api";

interface HeaderProps {
  activeTab: "workspace" | "history" | "documents" | "metrics";
  onTabChange: (tab: "workspace" | "history" | "documents" | "metrics") => void;
  activeRole: UserRole;
  onRoleChange: (role: UserRole) => void;
  activeModelName?: string | null;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onTabChange,
  activeRole,
  onRoleChange,
  activeModelName,
}) => {
  return (
    <>
      <header className="app-header">
        <div className="header-brand">
          <Shield size={24} color="#3b82f6" />
          <div>
            <div className="brand-title">
              XAI-RAG <span style={{ color: "var(--text-muted)", fontWeight: 400 }}>|</span> Credit Decision Auditor
            </div>
            <div className="brand-subtitle">
              Tabular XGBoost + TreeSHAP + DiCE Counterfactuals + Hybrid RAG + Tamper-Evident Ledger
            </div>
          </div>
        </div>

        <div className="header-controls">
          {activeModelName && (
            <div className="badge badge-neutral font-mono" style={{ padding: "4px 10px" }}>
              <Layers size={12} style={{ marginRight: "4px" }} />
              Model: {activeModelName}
            </div>
          )}

          <div className="role-selector-wrap">
            <UserCheck size={14} color="#38bdf8" />
            <span className="role-label">Active Role:</span>
            <select
              className="role-select"
              value={activeRole}
              onChange={(e) => onRoleChange(e.target.value as UserRole)}
            >
              <option value="analyst">Analyst</option>
              <option value="reviewer">Reviewer</option>
              <option value="admin">Admin</option>
            </select>
          </div>
        </div>
      </header>

      <nav className="app-tabs">
        <button
          className={`tab-btn ${activeTab === "workspace" ? "active" : ""}`}
          onClick={() => onTabChange("workspace")}
        >
          <Layers size={16} />
          Audit Workspace
        </button>
        <button
          className={`tab-btn ${activeTab === "history" ? "active" : ""}`}
          onClick={() => onTabChange("history")}
        >
          <FileText size={16} />
          Audit History
        </button>
        <button
          className={`tab-btn ${activeTab === "documents" ? "active" : ""}`}
          onClick={() => onTabChange("documents")}
        >
          <Database size={16} />
          Document Administration
        </button>
        <button
          className={`tab-btn ${activeTab === "metrics" ? "active" : ""}`}
          onClick={() => onTabChange("metrics")}
        >
          <BarChart3 size={16} />
          Quantitative Metrics
        </button>
      </nav>
    </>
  );
};
