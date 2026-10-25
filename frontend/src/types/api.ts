/**
 * Strict TypeScript types for XAI-RAG.
 * Mirrors backend Pydantic models (AuditReport, Orchestrator, Models, Evaluation, Documents).
 */

export type Mode = "demo" | "configured";
export type AuditStatus = "complete" | "partial" | "failed";
export type StageStatus = "pending" | "running" | "succeeded" | "failed" | "skipped";
export type JobState = "pending" | "running" | "completed" | "failed" | "partial";
export type FeatureDirection = "risk_increasing" | "risk_reducing" | "neutral";
export type OutputSpace = "log_odds" | "probability";
export type FeasibilityStatus = "feasible" | "infeasible" | "partially_feasible";
export type EvidenceStatus = "verified" | "unverified" | "failed";
export type UserRole = "analyst" | "reviewer" | "admin";

export interface ModelSummary {
  name: string;
  version: string;
  artifact_sha256: string;
  calibration_status: string;
}

export interface PredictionSummary {
  raw_output: number;
  default_probability: number;
  threshold: number;
  label: string;
  output_notes: string;
}

export interface FeatureContribution {
  name: string;
  label: string;
  value: number;
  unit: string;
  shap_value: number;
  direction: FeatureDirection;
}

export interface ExplanationSummary {
  output_space: OutputSpace;
  base_value: number;
  shap_additivity_residual: number;
  features: FeatureContribution[];
}

export interface DerivedRecalculation {
  feature: string;
  formula: string;
  previous_value: number;
  new_value: number;
}

export interface FeatureChange {
  feature: string;
  current: number;
  proposed: number;
  delta: number;
}

export interface CounterfactualCandidate {
  id: string;
  changes: FeatureChange[];
  derived_recalculations: DerivedRecalculation[];
  predicted_probability: number;
  target_met: boolean;
  constraints_passed: string[];
  feasibility_status: FeasibilityStatus;
  caveats: string[];
}

export interface SearchTraceSummary {
  ontology_version: string;
  features_used: string[];
  dense_query: string;
  bm25_terms: string[];
  metadata_filters: Record<string, any>;
  retrieved_chunk_ids: string[];
  index_version: string;
}

export interface EvidenceItem {
  source_id: string;
  title: string;
  issuer: string;
  jurisdiction: string;
  source_type: string;
  version: string;
  effective_date: string;
  page: number;
  section: string;
  quote: string;
  chunk_id: string;
  url: string;
  verification_status: EvidenceStatus;
}

export interface DossierSummary {
  adverse_action_explanation: string;
  candidate_review_paths: string[];
  regulatory_justification: string;
  evidence_gaps: string[];
  reviewer_notes: string;
}

export interface VerifierSummary {
  quote_checks_total: number;
  quote_checks_passed: number;
  quote_checks_failed: number;
  faithfulness_score: number;
  normalized_match_policy: string;
}

/**
 * Strict AuditReport contract shared across backend and frontend.
 */
export interface AuditReport {
  audit_id: string;
  applicant_ref: string;
  generated_at: string;
  mode: Mode;
  status: AuditStatus;
  model: ModelSummary;
  prediction: PredictionSummary;
  explanation: ExplanationSummary;
  counterfactuals: CounterfactualCandidate[];
  search_trace: SearchTraceSummary;
  evidence: EvidenceItem[];
  dossier: DossierSummary;
  verifier: VerifierSummary;
  human_review_required: boolean;
}

export interface StageExecutionRecord {
  stage_name: string;
  status: StageStatus;
  started_at: string | null;
  completed_at: string | null;
  duration_ms: number | null;
  error_message: string | null;
  metadata: Record<string, any>;
}

export interface AuditJobStatusResponse {
  job_id: string;
  audit_id: string | null;
  status: JobState;
  current_stage: string | null;
  progress_percentage: number;
  created_at: string;
  updated_at: string;
  stages: StageExecutionRecord[];
  error_message: string | null;
  report: AuditReport | null;
}

export interface ApplicantSummary {
  applicant_ref: string;
  label: string;
  requested_amount_inr: number;
  income_inr: number;
  data_mode: string;
  file_path: string;
}

export interface ApplicantDetail {
  applicant_ref: string;
  data_mode: string;
  features: Record<string, number>;
  notes: string;
}

export interface AuditRecordSummary {
  audit_id: string;
  applicant_ref: string;
  generated_at: string;
  mode: Mode;
  status: AuditStatus;
  model_name: string;
  model_version: string;
  default_probability: number | null;
  decision_label: string | null;
  human_review_required: boolean;
}

export interface DocumentRecord {
  source_id: string;
  title: string;
  issuer: string;
  jurisdiction: string;
  source_family: string;
  version: string;
  effective_date: string;
  status: "active" | "superseded";
  sha256_hash: string;
  total_pages: number;
  total_chunks: number;
  parser_status: string;
  access_classification: string;
  ingestion_timestamp: string | null;
}

export interface EvaluationMetricsResponse {
  evaluation_metadata?: {
    execution_timestamp: string;
    script_path: string;
    model_version: string;
    feature_schema_hash: string;
    index_version: string;
    corpus_chunks_count: number;
    data_mode: string;
  };
  counterfactual_metrics?: {
    sample_size_applicants: number;
    total_candidates_evaluated: number;
    valid_candidates_count: number;
    actionable_candidates_count: number;
    validity_rate: number;
    actionability_rate: number;
    validity_rate_definition: string;
    actionability_rate_definition: string;
    mean_generation_latency_seconds: number;
    mean_target_miss_magnitude: number;
  };
  retrieval_metrics?: {
    k: number;
    sample_size_test_cases: number;
    averaging_method: string;
    benchmark_claim_control: string;
    ontology_bridged_method: {
      name: string;
      "Recall@3": number;
      "Precision@3": number;
      MRR: number;
      "nDCG@3": number;
    };
    naive_baseline_method: {
      name: string;
      "Recall@3": number;
      "Precision@3": number;
      MRR: number;
      "nDCG@3": number;
    };
  };
  citation_faithfulness?: {
    sample_size_quotes_checked: number;
    quote_checks_passed: number;
    quote_checks_failed: number;
    faithfulness_score: number;
    normalization_rules: string;
    target_display_quotations: number;
    fabricated_quote_detected_correctly: boolean;
  };
  explanation_fidelity?: {
    output_space: string;
    base_value: number;
    shap_additivity_residual: number;
    additivity_residual_within_tolerance: boolean;
    features_attributed_count: number;
    reconstruction_check: string;
  };
  benchmark_claim_control?: string;
  status?: string;
}
