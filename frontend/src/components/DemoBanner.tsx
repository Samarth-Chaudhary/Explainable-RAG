import React from "react";
import { AlertTriangle, ShieldAlert } from "lucide-react";

interface DemoBannerProps {
  mode?: string;
}

export const DemoBanner: React.FC<DemoBannerProps> = ({ mode = "demo" }) => {
  return (
    <div className="demo-banner">
      <div className="demo-banner-left">
        <AlertTriangle size={16} color="#fbbf24" />
        <span className="demo-badge-pill">
          {mode === "demo" ? "DEMO / SYNTHETIC DATA" : "CONFIGURED ENVIRONMENT"}
        </span>
        <span>
          Illustrative values & synthetic policies only. Model math, SHAP attributions, and citations are verified against local artifacts.
        </span>
      </div>
      <div className="human-review-marker">
        <ShieldAlert size={14} style={{ display: "inline", verticalAlign: "middle", marginRight: "4px" }} />
        <strong>HUMAN REVIEW REQUIRED</strong> — Decision Support Only
      </div>
    </div>
  );
};
