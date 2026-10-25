import React from "react";
import { Check, Clock, AlertCircle, Loader2 } from "lucide-react";
import { StageExecutionRecord } from "../types/api";

interface StageProgressStepperProps {
  stages: StageExecutionRecord[];
  currentStage: string | null;
  overallStatus: string;
}

const ORDERED_STAGE_KEYS = [
  "validate_input",
  "predict",
  "explain",
  "counterfactuals",
  "translate_query",
  "retrieve",
  "generate_report",
  "verify_citations",
  "persist",
  "render",
];

const STAGE_LABELS: Record<string, string> = {
  validate_input: "Validation",
  predict: "Inference",
  explain: "TreeSHAP",
  counterfactuals: "DiCE What-If",
  translate_query: "Ontology",
  retrieve: "Hybrid Search",
  generate_report: "Structured Synthesis",
  verify_citations: "Citation Verifier",
  persist: "Ledger",
  render: "Report Render",
};

export const StageProgressStepper: React.FC<StageProgressStepperProps> = ({
  stages,
  currentStage,
  overallStatus,
}) => {
  const stageMap = new Map<string, StageExecutionRecord>();
  for (const s of stages) {
    stageMap.set(s.stage_name, s);
  }

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <span className="card-title">Run Audit Stage Pipeline</span>
          <span className="card-subtitle">
            Truthful telemetry emitted directly by RunAuditOrchestrator backend stages
          </span>
        </div>
        <div className="badge font-mono" style={{ textTransform: "uppercase" }}>
          Status: {overallStatus}
        </div>
      </div>
      <div className="card-body">
        <div className="stepper-container">
          {ORDERED_STAGE_KEYS.map((key, index) => {
            const stage = stageMap.get(key);
            const status = stage ? stage.status : "pending";
            const isCurrent = currentStage === key;

            return (
              <div key={key} className={`step-item ${status} ${isCurrent ? "running" : ""}`}>
                <div className="step-indicator">
                  {status === "succeeded" && <Check size={14} />}
                  {status === "running" && <Loader2 size={14} className="animate-spin" />}
                  {status === "failed" && <AlertCircle size={14} />}
                  {status === "pending" && <span>{index + 1}</span>}
                  {status === "skipped" && <span>-</span>}
                </div>
                <div className="step-name">{STAGE_LABELS[key] || key}</div>
                {stage?.duration_ms != null && (
                  <div className="step-duration">{Math.round(stage.duration_ms)}ms</div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
