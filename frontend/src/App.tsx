import React, { useState, useEffect } from "react";
import { DemoBanner } from "./components/DemoBanner";
import { Header } from "./components/Header";
import { StageProgressStepper } from "./components/StageProgressStepper";
import { ApplicantWorkspace } from "./components/ApplicantWorkspace";
import { PredictionSummaryCard } from "./components/PredictionSummaryCard";
import { ShapAttributionCard } from "./components/ShapAttributionCard";
import { CounterfactualCard } from "./components/CounterfactualCard";
import { EvidencePanel } from "./components/EvidencePanel";
import { DossierView } from "./components/DossierView";
import { ReviewerDispositionModal } from "./components/ReviewerDispositionModal";
import { AuditHistoryTab } from "./components/AuditHistoryTab";
import { DocumentAdminTab } from "./components/DocumentAdminTab";
import { MetricsTab } from "./components/MetricsTab";
import { api, setActiveUserRole } from "./api/client";
import { AuditJobStatusResponse, AuditReport, UserRole } from "./types/api";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"workspace" | "history" | "documents" | "metrics">("workspace");
  const [activeRole, setActiveRoleState] = useState<UserRole>("analyst");
  const [activeReport, setActiveReport] = useState<AuditReport | null>(null);
  const [jobStatus, setJobStatus] = useState<AuditJobStatusResponse | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [isDispositionOpen, setIsDispositionOpen] = useState<boolean>(false);
  const [activeModelName, setActiveModelName] = useState<string | null>("CreditRiskXGBoostDemo");

  useEffect(() => {
    // Load active model name
    api.getModels()
      .then((res) => {
        if (res.active_model) setActiveModelName(res.active_model);
      })
      .catch((err) => console.warn("Failed to fetch model catalog", err));
  }, []);

  const handleRoleChange = (role: UserRole) => {
    setActiveRoleState(role);
    setActiveUserRole(role);
  };

  const handleRunAudit = async (applicantData: Record<string, any>, applicantRef: string) => {
    setIsRunning(true);
    setActiveReport(null);
    setJobStatus(null);

    const idempotencyKey = `IDEMP-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;

    try {
      // 1. Trigger audit execution (async background job for truthful stage telemetry)
      const initJob = await api.runAudit({
        applicant_data: applicantData,
        applicant_ref: applicantRef,
        idempotency_key: idempotencyKey,
        run_async: true,
        user_role: activeRole,
      });

      setJobStatus(initJob);

      // 2. Poll job status until terminal state
      const pollInterval = setInterval(async () => {
        try {
          const pollRes = await api.getJobStatus(initJob.job_id);
          setJobStatus(pollRes);

          if (pollRes.status === "completed" || pollRes.status === "failed" || pollRes.status === "partial") {
            clearInterval(pollInterval);
            setIsRunning(false);

            if (pollRes.audit_id) {
              const fullReport = await api.getAuditReport(pollRes.audit_id);
              setActiveReport(fullReport);
            }
          }
        } catch (pollErr) {
          console.error("Polling error", pollErr);
          clearInterval(pollInterval);
          setIsRunning(false);
        }
      }, 400);
    } catch (err: any) {
      alert(`Audit initiation failed: ${err.message}`);
      setIsRunning(false);
    }
  };

  const handleSelectAuditFromHistory = async (auditId: string) => {
    try {
      const report = await api.getAuditReport(auditId);
      setActiveReport(report);
      setActiveTab("workspace");
    } catch (err: any) {
      alert(`Failed to reload audit '${auditId}': ${err.message}`);
    }
  };

  return (
    <div className="app-container">
      {/* 1. Persistent Top Warning Banner */}
      <DemoBanner mode={activeReport?.mode || "demo"} />

      {/* 2. Institutional Navigation Header */}
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        activeRole={activeRole}
        onRoleChange={handleRoleChange}
        activeModelName={activeModelName}
      />

      {/* 3. Main Workspace Area */}
      <main className="app-main">
        {activeTab === "workspace" && (
          <>
            {/* Applicant Selector & Feature Workspace */}
            <ApplicantWorkspace onRunAudit={handleRunAudit} isRunning={isRunning} />

            {/* Truthful Stage Progress Stepper */}
            {jobStatus && (
              <StageProgressStepper
                stages={jobStatus.stages}
                currentStage={jobStatus.current_stage}
                overallStatus={jobStatus.status}
              />
            )}

            {/* Loaded / Generated Audit Report Results */}
            {activeReport && (
              <>
                {/* Prediction Summary */}
                <PredictionSummaryCard
                  prediction={activeReport.prediction}
                  model={activeReport.model}
                  generatedAt={activeReport.generated_at}
                />

                {/* 2-Column Grid: SHAP Attribution & Counterfactuals */}
                <div className="grid-2">
                  <ShapAttributionCard explanation={activeReport.explanation} />
                  <CounterfactualCard counterfactuals={activeReport.counterfactuals} />
                </div>

                {/* Retrieved Policy Evidence & Provenance */}
                <EvidencePanel
                  evidence={activeReport.evidence}
                  verifier={activeReport.verifier}
                />

                {/* Structured Dossier & Export Actions */}
                <DossierView
                  report={activeReport}
                  activeRole={activeRole}
                  onOpenDispositionModal={() => setIsDispositionOpen(true)}
                />
              </>
            )}
          </>
        )}

        {activeTab === "history" && (
          <AuditHistoryTab onSelectAudit={handleSelectAuditFromHistory} />
        )}

        {activeTab === "documents" && (
          <DocumentAdminTab activeRole={activeRole} />
        )}

        {activeTab === "metrics" && <MetricsTab />}
      </main>

      {/* Reviewer Disposition Modal */}
      {activeReport && (
        <ReviewerDispositionModal
          auditId={activeReport.audit_id}
          isOpen={isDispositionOpen}
          onClose={() => setIsDispositionOpen(false)}
          onSuccess={() => {
            // Reload report to refresh reviewer state
            api.getAuditReport(activeReport.audit_id).then(setActiveReport);
          }}
        />
      )}
    </div>
  );
};
