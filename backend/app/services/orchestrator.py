"""Run Audit Orchestrator Service.

Executes and coordinates the 10-stage credit decision auditing pipeline:
1. validate_input: Validates applicant features against the FeatureSchema.
2. predict: XGBoost deterministic inference with threshold comparison.
3. explain: TreeSHAP attribution with additivity gating (blocks narrative if invalid).
4. counterfactuals: Constrained DiCE what-if generation with derived recalculations.
5. translate_query: Semantic ontology bridge translating risk drivers to hybrid search.
6. retrieve: Hybrid FAISS + BM25 retrieval with Reciprocal Rank Fusion & RBAC.
7. generate_report: Citation-locked structured report generation (local deterministic or LLM).
8. verify_citations: Verbatim quote verification under strict Unicode NFKC normalization.
9. persist: Tamper-evident hash-chained audit ledger persistence.
10. render: Validated AuditReport rendering payload.

Supports synchronous and asynchronous job execution, client idempotency keys,
detailed per-stage status/timing telemetry, and fail-safe partial report persistence.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any
import uuid

import pandas as pd
from sqlalchemy.orm import Session

from backend.app.config import Settings, get_settings
from backend.app.models.audit_models import Base
from backend.app.repositories.audit_repository import AuditRepository
from backend.app.schemas.audit_report import (
    AuditReport,
    ReportDossierSummary,
    ReportEvidenceItem,
    ReportExplanationSummary,
    ReportFeatureContribution,
    ReportModelSummary,
    ReportPredictionSummary,
    ReportSearchTraceSummary,
    ReportVerifierSummary,
)
from backend.app.schemas.counterfactual import CounterfactualResult
from backend.app.schemas.explanation import ExplanationResult
from backend.app.schemas.feature_schema import FeatureSchema
from backend.app.schemas.handoff import (
    Chunk,
    GeneratorInputPayload,
    RetrievalResult,
    RetrievalTrace,
    TranslationResult,
)
from backend.app.schemas.source_registry import SourceRegistryRecord
from backend.app.schemas.model_manifest import ModelManifest
from backend.app.schemas.orchestrator import (
    AuditJobStatusResponse,
    AuditRunRequest,
    JobState,
    StageExecutionRecord,
    StageStatus,
)
from backend.app.schemas.prediction import (
    InputValidationError,
    PredictionResult,
    ThresholdConfig,
)
from backend.app.services.audit_ledger import AuditLedgerService
from backend.app.services.citation_verifier import (
    CitationVerifierService,
    NORMALIZED_MATCH_POLICY,
)
from backend.app.services.counterfactual import CounterfactualService
from backend.app.services.document_ingestion import DocumentIngestionService
from backend.app.services.explanation import ExplanationService
from backend.app.services.ontology_translator import OntologyService, OntologyTranslatorService
from backend.app.services.prediction import ModelLoader, PredictionService
from backend.app.services.report_generator import ExplanationAdditivityError, ReportGeneratorService
from backend.app.services.retrieval import HybridRetrievalService

logger = logging.getLogger(__name__)

STAGE_NAMES = [
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
]


class RunAuditOrchestrator:
    """Coordinates and executes the full XAI-RAG decision auditing pipeline."""

    def __init__(
        self,
        settings: Settings | None = None,
        thread_pool_workers: int = 4,
    ):
        self.settings = settings or get_settings()
        self.executor = ThreadPoolExecutor(max_workers=thread_pool_workers)
        self._jobs: dict[str, AuditJobStatusResponse] = {}
        self._idempotency_map: dict[str, str] = {}  # idempotency_key -> job_id
        self._lock = threading.Lock()

        # Cached components initialized lazily or on startup
        self._schema: FeatureSchema | None = None
        self._manifest: ModelManifest | None = None
        self._model_loader: ModelLoader | None = None
        self._pred_service: PredictionService | None = None
        self._background_df: pd.DataFrame | None = None
        self._ontology_service: OntologyService | None = None
        self._ontology_translator: OntologyTranslatorService | None = None
        self._ingestion_service: DocumentIngestionService | None = None
        self._retrieval_service: HybridRetrievalService | None = None
        self._report_service: ReportGeneratorService | None = None
        self._verifier_service: CitationVerifierService | None = None

        self._initialize_core_services()

    def _initialize_core_services(self) -> None:
        """Initialize and validate model, ontology, and retrieval dependencies."""
        schema_path = Path(self.settings.feature_schema_path)
        if schema_path.is_file():
            with open(schema_path, "r", encoding="utf-8") as f:
                self._schema = FeatureSchema.model_validate(json.load(f))

        manifest_path = Path(self.settings.model_manifest_path)
        if manifest_path.is_file():
            with open(manifest_path, "r", encoding="utf-8") as f:
                self._manifest = ModelManifest.model_validate(json.load(f))

        if self._schema and self._manifest:
            self._model_loader = ModelLoader(
                model_path=self.settings.model_path,
                manifest_path=self.settings.model_manifest_path,
                feature_schema=self._schema,
            )
            threshold_cfg = ThresholdConfig(
                value=self.settings.threshold_value,
                source=self.settings.threshold_source,
                owner=self.settings.threshold_owner,
            )
            self._pred_service = PredictionService(
                loader=self._model_loader,
                feature_schema=self._schema,
                threshold_config=threshold_cfg,
            )

        bg_path = Path(self.settings.background_data_path)
        if bg_path.is_file():
            self._background_df = pd.read_json(bg_path)

        ontology_path = Path(getattr(self.settings, "ontology_path", "data/ontology/credit_risk_ontology_v1.json"))
        if ontology_path.is_file() and self._schema:
            self._ontology_service = OntologyService(str(ontology_path), feature_schema=self._schema)
            self._ontology_translator = OntologyTranslatorService(self._ontology_service)

        self._ingestion_service = DocumentIngestionService(max_upload_size_bytes=self.settings.max_upload_size_bytes)
        self._retrieval_service = HybridRetrievalService()
        self._bootstrap_policy_indexes()

        self._report_service = ReportGeneratorService(mode=self.settings.mode)
        self._verifier_service = CitationVerifierService(fail_closed=False)

    def _bootstrap_policy_indexes(self) -> None:
        """Ingest synthetic policy fixtures into hybrid search indexes if available."""
        if not self._retrieval_service or not self._ingestion_service:
            return

        policies_dir = Path("data/policies")
        if not policies_dir.is_dir():
            return

        chunks_to_index: list[Chunk] = []
        now = datetime.now(timezone.utc)

        sovereign_policies = [
            {
                "filename": "fed_sr11_7.pdf",
                "source_id": "FED-SR11-7",
                "title": "Supervisory Guidance on Model Risk Management (SR 11-7)",
                "issuer": "Federal Reserve Board & OCC",
                "jurisdiction": "US",
                "source_family": "model_risk_management",
                "pub_date": "2011-04-04",
                "version": "SR 11-7",
                "url": "https://www.federalreserve.gov/boarddocs/srletters/2011/sr1107a1.pdf",
            },
            {
                "filename": "basel_credit_risk.pdf",
                "source_id": "BCBS-75",
                "title": "Principles for the Management of Credit Risk (BCBS 75)",
                "issuer": "Basel Committee on Banking Supervision (BIS)",
                "jurisdiction": "Global / International",
                "source_family": "credit_risk_principles",
                "pub_date": "2000-09-01",
                "version": "BCBS 75",
                "url": "https://www.bis.org/publications/200009-guidelines-principles-management-credit-risk.pdf",
            },
            {
                "filename": "rbi_digital_lending_2025.pdf",
                "source_id": "RBI-DIGITAL-LENDING-2025",
                "title": "Reserve Bank of India (Digital Lending) Directions, 2025",
                "issuer": "Reserve Bank of India (RBI)",
                "jurisdiction": "IN",
                "source_family": "digital_lending_directions",
                "pub_date": "2025-05-08",
                "version": "RBI/2025-26/36",
                "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/36NT8C402BE7C2A349E0BFFF3C526668CD7A.PDF",
            },
            {
                "filename": "rbi_fair_practices_code.pdf",
                "source_id": "RBI-FAIR-PRACTICES-CODE",
                "title": "Master Circular - Fair Practices Code for Non-Banking Financial Companies",
                "issuer": "Reserve Bank of India (RBI)",
                "jurisdiction": "IN",
                "source_family": "fair_practices_code",
                "pub_date": "2014-07-01",
                "version": "DNBS(PD)CC.No.388/03.10.042/2014-15",
                "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/AI141P060614F.PDF",
            },
        ]

        import hashlib
        for pol in sovereign_policies:
            pdf_path = policies_dir / pol["filename"]
            if pdf_path.is_file():
                file_bytes = pdf_path.read_bytes()
                sha256 = hashlib.sha256(file_bytes).hexdigest()
                meta = SourceRegistryRecord(
                    source_id=pol["source_id"],
                    title=pol["title"],
                    issuer=pol["issuer"],
                    jurisdiction=pol["jurisdiction"],
                    source_family=pol["source_family"],
                    publication_date=pol["pub_date"],
                    effective_date=pol["pub_date"],
                    version_or_update_date=pol["version"],
                    status="active",
                    official_url_or_owner=pol["url"],
                    sha256_hash=sha256,
                    ingestion_timestamp=now,
                    access_classification="public",
                    parser_status="pending",
                )
                try:
                    ingest_res = self._ingestion_service.ingest_pdf(pdf_path, meta)
                    chunks_to_index.extend(ingest_res.chunks)
                except Exception as exc:
                    logger.warning("Failed to ingest %s: %s", pol["filename"], exc)

        if chunks_to_index:
            self._retrieval_service.build_indexes(chunks_to_index)
            logger.info("Bootstrap indexed %d sovereign regulatory chunks into HybridRetrievalService", len(chunks_to_index))

    def get_job(self, job_id: str) -> AuditJobStatusResponse | None:
        """Fetch job status and stage progress."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[AuditJobStatusResponse]:
        """List all tracked audit jobs."""
        with self._lock:
            return list(self._jobs.values())

    def execute_audit(
        self,
        request: AuditRunRequest,
        session: Session,
    ) -> AuditJobStatusResponse:
        """Execute or enqueue Run Audit pipeline with idempotency protection."""
        # 1. Idempotency Check
        if request.idempotency_key:
            with self._lock:
                existing_job_id = self._idempotency_map.get(request.idempotency_key)
                if existing_job_id and existing_job_id in self._jobs:
                    logger.info("Idempotent request returning existing job %s", existing_job_id)
                    return self._jobs[existing_job_id]

        job_id = f"job-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        # Initialize per-stage tracker
        initial_stages = [
            StageExecutionRecord(stage_name=name, status="pending")
            for name in STAGE_NAMES
        ]

        job_status = AuditJobStatusResponse(
            job_id=job_id,
            audit_id=None,
            idempotency_key=request.idempotency_key,
            status="queued" if request.run_async else "running",
            progress_percentage=0,
            current_stage=STAGE_NAMES[0],
            stages=initial_stages,
            report=None,
            error=None,
            created_at=now,
            completed_at=None,
        )

        with self._lock:
            self._jobs[job_id] = job_status
            if request.idempotency_key:
                self._idempotency_map[request.idempotency_key] = job_id

        if request.run_async:
            self.executor.submit(self._run_pipeline_worker, job_id, request, session)
            return job_status
        else:
            return self._run_pipeline_worker(job_id, request, session)

    def _update_stage(
        self,
        job_id: str,
        stage_name: str,
        status: StageStatus,
        error_msg: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update individual stage status, timing, and overall progress."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return

            now = datetime.now(timezone.utc)
            updated_stages: list[StageExecutionRecord] = []
            completed_count = 0

            for st in job.stages:
                if st.stage_name == stage_name:
                    start_time = st.started_at or now if status == "running" else st.started_at
                    end_time = now if status in ("succeeded", "failed", "skipped") else None
                    duration = None
                    if start_time and end_time:
                        duration = (end_time - start_time).total_seconds() * 1000.0

                    updated_st = StageExecutionRecord(
                        stage_name=st.stage_name,
                        status=status,
                        started_at=start_time,
                        completed_at=end_time,
                        duration_ms=duration,
                        error_message=error_msg,
                        metadata=metadata or st.metadata,
                    )
                    updated_stages.append(updated_st)
                else:
                    updated_stages.append(st)

                if st.status in ("succeeded", "skipped"):
                    completed_count += 1

            progress = int((completed_count / len(STAGE_NAMES)) * 100)
            self._jobs[job_id] = AuditJobStatusResponse(
                job_id=job.job_id,
                audit_id=job.audit_id,
                idempotency_key=job.idempotency_key,
                status="running" if status == "running" else job.status,
                progress_percentage=progress,
                current_stage=stage_name,
                stages=updated_stages,
                report=job.report,
                error=job.error,
                created_at=job.created_at,
                completed_at=job.completed_at,
            )

    def _run_pipeline_worker(
        self,
        job_id: str,
        request: AuditRunRequest,
        session: Session,
    ) -> AuditJobStatusResponse:
        """Synchronous execution worker that steps through all 10 pipeline stages."""
        audit_id = f"AUDIT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        with self._lock:
            job = self._jobs[job_id]
            self._jobs[job_id] = AuditJobStatusResponse(
                job_id=job.job_id,
                audit_id=audit_id,
                idempotency_key=job.idempotency_key,
                status="running",
                progress_percentage=job.progress_percentage,
                current_stage=STAGE_NAMES[0],
                stages=job.stages,
                report=None,
                error=None,
                created_at=job.created_at,
                completed_at=None,
            )

        repo = AuditRepository(session)
        ledger_service = AuditLedgerService(repo)

        applicant_ref = request.applicant_ref or request.applicant_data.get("applicant_ref") or "APP-ANON"
        raw_applicant_features = request.applicant_data.get("features", request.applicant_data)

        # Variables accumulated across stages
        prediction_res: PredictionResult | None = None
        explanation_res: ExplanationResult | None = None
        counterfactual_res: CounterfactualResult | None = None
        translation_res: TranslationResult | None = None
        retrieval_res: RetrievalResult | None = None
        draft_report = None
        final_report: AuditReport | None = None
        failure_reason: str | None = None

        try:
            # ------------------------------------------------------------------
            # Stage 1: validate_input
            # ------------------------------------------------------------------
            self._update_stage(job_id, "validate_input", "running")
            t0 = time.perf_counter()
            if not self._pred_service:
                raise RuntimeError("PredictionService is not initialized")

            # Validates against FeatureSchema
            validated_primitives = self._pred_service.validate_applicant_data(raw_applicant_features)
            applicant_input_dict = {**raw_applicant_features, **validated_primitives}
            duration_ms = (time.perf_counter() - t0) * 1000.0

            ledger_service.append_event(
                audit_id,
                "INPUT_VALIDATED",
                request.user_role,
                {"applicant_ref": applicant_ref, "feature_count": len(validated_primitives)},
            )
            self._update_stage(job_id, "validate_input", "succeeded", metadata={"duration_ms": duration_ms})

            # ------------------------------------------------------------------
            # Stage 2: predict
            # ------------------------------------------------------------------
            self._update_stage(job_id, "predict", "running")
            t0 = time.perf_counter()
            prediction_res = self._pred_service.predict(applicant_input_dict)
            duration_ms = (time.perf_counter() - t0) * 1000.0

            ledger_service.append_event(
                audit_id,
                "PREDICTION_COMPLETED",
                "system",
                {
                    "raw_output": prediction_res.raw_output,
                    "probability": prediction_res.default_probability,
                    "label": prediction_res.decision_label,
                    "threshold": prediction_res.threshold.value,
                },
            )
            self._update_stage(
                job_id,
                "predict",
                "succeeded",
                metadata={"probability": prediction_res.default_probability, "label": prediction_res.decision_label},
            )

            # ------------------------------------------------------------------
            # Stage 3: explain (TreeSHAP with additivity gating)
            # ------------------------------------------------------------------
            self._update_stage(job_id, "explain", "running")
            t0 = time.perf_counter()
            query_df, _ = self._pred_service.prepare_feature_vector(applicant_input_dict)

            explanation_service = ExplanationService(
                model=self._model_loader.model,
                feature_schema=self._schema,
                background_data=self._background_df,
                output_space=self.settings.shap_output_space,
                tolerance=self.settings.shap_tolerance,
                top_k=self.settings.shap_top_k,
                contribution_cutoff=self.settings.shap_contribution_cutoff,
                background_id=self._manifest.explainer_background_data_id if self._manifest else None,
            )
            explanation_res = explanation_service.explain(feature_df=query_df, prediction=prediction_res)
            duration_ms = (time.perf_counter() - t0) * 1000.0

            ledger_service.append_event(
                audit_id,
                "EXPLANATION_COMPLETED",
                "system",
                {
                    "output_space": explanation_res.output_space,
                    "additivity_status": explanation_res.additivity_status,
                    "residual": explanation_res.shap_additivity_residual,
                },
            )

            # Gating check: if additivity is invalid, block narrative generation
            if explanation_res.additivity_status == "invalid":
                raise ExplanationAdditivityError(
                    f"SHAP additivity check failed (residual {explanation_res.shap_additivity_residual} "
                    f"> tolerance {explanation_res.tolerance}). Report narrative generation blocked."
                )

            self._update_stage(
                job_id,
                "explain",
                "succeeded",
                metadata={
                    "additivity_status": explanation_res.additivity_status,
                    "top_drivers": [d.name for d in explanation_res.top_risk_increasing[:3]],
                },
            )

            # ------------------------------------------------------------------
            # Stage 4: counterfactuals (Constrained DiCE)
            # ------------------------------------------------------------------
            self._update_stage(job_id, "counterfactuals", "running")
            t0 = time.perf_counter()
            cf_service = CounterfactualService(
                prediction_service=self._pred_service,
                feature_schema=self._schema,
                training_df=self._background_df,
                dice_method=self.settings.dice_method,
                target_threshold=self.settings.dice_target_probability_threshold,
            )
            counterfactual_res = cf_service.generate_counterfactuals(
                applicant_data=applicant_input_dict,
                total_cfs=self.settings.dice_total_cfs,
            )
            duration_ms = (time.perf_counter() - t0) * 1000.0

            ledger_service.append_event(
                audit_id,
                "COUNTERFACTUALS_COMPLETED",
                "system",
                {
                    "candidates_found": len(counterfactual_res.candidates),
                    "no_feasible_found": counterfactual_res.no_feasible_found,
                },
            )
            self._update_stage(
                job_id,
                "counterfactuals",
                "succeeded",
                metadata={"candidates_count": len(counterfactual_res.candidates)},
            )

            # ------------------------------------------------------------------
            # Stage 5: translate_query (Ontology Bridge)
            # ------------------------------------------------------------------
            self._update_stage(job_id, "translate_query", "running")
            t0 = time.perf_counter()
            if not self._ontology_translator:
                raise RuntimeError("OntologyTranslatorService is not initialized")

            translation_res = self._ontology_translator.translate(explanation_res, counterfactual_res)
            duration_ms = (time.perf_counter() - t0) * 1000.0

            ledger_service.append_event(
                audit_id,
                "QUERY_TRANSLATED",
                "system",
                {
                    "dense_query": translation_res.query_payload.dense_query,
                    "bm25_terms": translation_res.query_payload.bm25_terms,
                    "ontology_version": translation_res.ontology_version,
                },
            )
            self._update_stage(
                job_id,
                "translate_query",
                "succeeded",
                metadata={"matches": translation_res.ontology_matches, "unmapped": translation_res.unmapped_features},
            )

            # ------------------------------------------------------------------
            # Stage 6: retrieve (Hybrid FAISS + BM25 + RRF)
            # ------------------------------------------------------------------
            self._update_stage(job_id, "retrieve", "running")
            t0 = time.perf_counter()
            if not self._retrieval_service:
                raise RuntimeError("HybridRetrievalService is not initialized")

            retrieval_res = self._retrieval_service.retrieve(
                translation_res.query_payload,
                user_role=request.user_role,
            )
            duration_ms = (time.perf_counter() - t0) * 1000.0

            ledger_service.append_event(
                audit_id,
                "RETRIEVAL_FUSED",
                "system",
                {
                    "evidence_count": len(retrieval_res.evidence),
                    "no_evidence_found": retrieval_res.no_evidence_found,
                    "index_version": self._retrieval_service.index_version or "INDEX-DEFAULT",
                },
            )
            self._update_stage(
                job_id,
                "retrieve",
                "succeeded",
                metadata={
                    "evidence_count": len(retrieval_res.evidence),
                    "no_evidence_found": retrieval_res.no_evidence_found,
                },
            )

            # ------------------------------------------------------------------
            # Stage 7: generate_report (Structured Dossier)
            # ------------------------------------------------------------------
            self._update_stage(job_id, "generate_report", "running")
            t0 = time.perf_counter()

            index_ver = self._retrieval_service.index_version or "INDEX-DEFAULT"
            gen_payload = GeneratorInputPayload(
                audit_id=audit_id,
                applicant_ref=applicant_ref,
                mode=self.settings.mode,
                model_manifest=self._manifest,
                prediction=prediction_res,
                explanation=explanation_res,
                counterfactuals=counterfactual_res,
                translation=translation_res,
                retrieval=retrieval_res,
                index_version=index_ver,
                user_role=request.user_role,
            )

            if not self._report_service:
                self._report_service = ReportGeneratorService(mode=self.settings.mode)

            draft_report = self._report_service.generate(gen_payload)
            duration_ms = (time.perf_counter() - t0) * 1000.0

            ledger_service.append_event(
                audit_id,
                "REPORT_DRAFTED",
                "system",
                {
                    "provider": draft_report.llm_provider,
                    "status": draft_report.status,
                    "evidence_count": len(draft_report.evidence),
                },
            )
            self._update_stage(
                job_id,
                "generate_report",
                "succeeded",
                metadata={"status": draft_report.status, "provider": draft_report.llm_provider},
            )

            # ------------------------------------------------------------------
            # Stage 8: verify_citations (Verbatim Exact Matching)
            # ------------------------------------------------------------------
            self._update_stage(job_id, "verify_citations", "running")
            t0 = time.perf_counter()

            all_candidate_chunks = retrieval_res.retrieval_trace.retrieved_chunks
            if not self._verifier_service:
                self._verifier_service = CitationVerifierService(fail_closed=False)

            verifier_res, final_report = self._verifier_service.verify(draft_report, all_candidate_chunks)
            duration_ms = (time.perf_counter() - t0) * 1000.0

            ledger_service.append_event(
                audit_id,
                "CITATIONS_VERIFIED",
                "system",
                {
                    "total_citations": verifier_res.total_citations,
                    "passed": verifier_res.passed_citations,
                    "failed": verifier_res.failed_citations,
                    "faithfulness_score": verifier_res.faithfulness_score,
                },
            )
            self._update_stage(
                job_id,
                "verify_citations",
                "succeeded",
                metadata={
                    "total": verifier_res.total_citations,
                    "passed": verifier_res.passed_citations,
                    "failed": verifier_res.failed_citations,
                    "faithfulness": verifier_res.faithfulness_score,
                },
            )

            # ------------------------------------------------------------------
            # Stage 9: persist (Tamper-Evident Ledger)
            # ------------------------------------------------------------------
            self._update_stage(job_id, "persist", "running")
            t0 = time.perf_counter()

            persisted_audit_id = ledger_service.persist_audit(
                final_report,
                retrieval_res.retrieval_trace,
                actor=request.user_role,
            )
            chain_valid, chain_issues = ledger_service.verify_chain(audit_id)
            duration_ms = (time.perf_counter() - t0) * 1000.0

            self._update_stage(
                job_id,
                "persist",
                "succeeded",
                metadata={"persisted_id": persisted_audit_id, "chain_valid": chain_valid},
            )

            # ------------------------------------------------------------------
            # Stage 10: render (Final Presentation Envelope)
            # ------------------------------------------------------------------
            self._update_stage(job_id, "render", "running")
            # Ready for rendering
            self._update_stage(job_id, "render", "succeeded")

            # Finalize Job Success
            now_done = datetime.now(timezone.utc)
            with self._lock:
                final_job = AuditJobStatusResponse(
                    job_id=job_id,
                    audit_id=audit_id,
                    idempotency_key=request.idempotency_key,
                    status="completed",
                    progress_percentage=100,
                    current_stage="completed",
                    stages=self._jobs[job_id].stages,
                    report=final_report,
                    error=None,
                    created_at=job.created_at,
                    completed_at=now_done,
                )
                self._jobs[job_id] = final_job
                return final_job

        except Exception as exc:
            logger.exception("Pipeline execution failed at stage for audit %s: %s", audit_id, exc)
            failure_reason = str(exc)

            # Build emergency partial AuditReport so audit lineage is never lost
            failed_report = self._build_emergency_failure_report(
                audit_id=audit_id,
                applicant_ref=applicant_ref,
                prediction=prediction_res,
                explanation=explanation_res,
                counterfactuals=counterfactual_res,
                failure_reason=failure_reason,
            )

            try:
                # Always persist failed audit into ledger
                empty_trace = RetrievalTrace(
                    trace_id=f"TRACE-FAIL-{uuid.uuid4().hex[:6]}",
                    query_payload=translation_res.query_payload if translation_res else HybridQueryPayload(dense_query="", bm25_terms=[], top_k=1),
                    dense_scores={},
                    bm25_scores={},
                    fused_scores={},
                    retrieved_chunks=[],
                    execution_time_ms=0.0,
                )
                ledger_service.append_event(
                    audit_id,
                    "PIPELINE_FAILED",
                    "system",
                    {"error": failure_reason},
                )
                ledger_service.persist_audit(failed_report, empty_trace, actor="system")
            except Exception as persist_err:
                logger.error("Failed to persist emergency failure audit: %s", persist_err)

            now_failed = datetime.now(timezone.utc)
            with self._lock:
                failed_job = AuditJobStatusResponse(
                    job_id=job_id,
                    audit_id=audit_id,
                    idempotency_key=request.idempotency_key,
                    status="failed",
                    progress_percentage=self._jobs[job_id].progress_percentage,
                    current_stage=self._jobs[job_id].current_stage,
                    stages=self._jobs[job_id].stages,
                    report=failed_report,
                    error=failure_reason,
                    created_at=job.created_at,
                    completed_at=now_failed,
                )
                self._jobs[job_id] = failed_job
                return failed_job

    def _build_emergency_failure_report(
        self,
        audit_id: str,
        applicant_ref: str,
        prediction: PredictionResult | None,
        explanation: ExplanationResult | None,
        counterfactuals: CounterfactualResult | None,
        failure_reason: str,
    ) -> AuditReport:
        """Construct a schema-valid fallback AuditReport preserving prior stage outputs."""
        now = datetime.now(timezone.utc)
        model_summary = ReportModelSummary(
            name=self._manifest.name if self._manifest else "CreditRiskXGBoostDemo",
            version=self._manifest.version if self._manifest else "1.0.0",
            artifact_sha256=self._manifest.artifact_sha256 if self._manifest else "UNKNOWN",
            calibration_status="calibrated",
        )

        if prediction:
            pred_summary = ReportPredictionSummary(
                raw_output=prediction.raw_output,
                default_probability=prediction.default_probability,
                threshold=prediction.threshold.value,
                label=prediction.decision_label,
                output_notes=f"Inference succeeded; downstream failure: {failure_reason}",
            )
        else:
            pred_summary = ReportPredictionSummary(
                raw_output=0.0,
                default_probability=0.0,
                threshold=0.35,
                label="INFERENCE_FAILED",
                output_notes=f"Prediction stage failed: {failure_reason}",
            )

        if explanation:
            expl_summary = ReportExplanationSummary(
                output_space=explanation.output_space,
                base_value=explanation.base_value,
                shap_additivity_residual=explanation.shap_additivity_residual,
                features=[
                    ReportFeatureContribution(
                        name=f.name,
                        label=f.label,
                        value=f.value,
                        unit=f.unit,
                        shap_value=f.shap_value,
                        direction=f.direction,
                    )
                    for f in explanation.features
                ],
            )
        else:
            expl_summary = ReportExplanationSummary(
                output_space="log_odds",
                base_value=0.0,
                shap_additivity_residual=0.0,
                features=[],
            )

        cf_candidates = []
        if counterfactuals and counterfactuals.candidates:
            from backend.app.schemas.audit_report import ReportCounterfactualCandidate
            for c in counterfactuals.candidates:
                cf_candidates.append(
                    ReportCounterfactualCandidate(
                        id=c.id,
                        changes=[ch.model_dump() for ch in c.changes],
                        derived_recalculations=[rec.model_dump() for rec in c.derived_recalculations],
                        predicted_probability=c.predicted_probability,
                        target_met=c.target_met,
                        constraints_passed=c.constraints_passed,
                        feasibility_status=c.feasibility_status,
                        caveats=c.caveats,
                    )
                )

        search_trace = ReportSearchTraceSummary(
            ontology_version=(
                self._ontology_service.ontology_file.ontology_version
                if (self._ontology_service and getattr(self._ontology_service, "ontology_file", None))
                else "1.0.0"
            ),
            features_used=[],
            dense_query="",
            bm25_terms=[],
            metadata_filters={},
            retrieved_chunk_ids=[],
            index_version="INDEX-FAILED",
        )

        dossier = ReportDossierSummary(
            adverse_action_explanation=f"Audit execution interrupted: {failure_reason}",
            candidate_review_paths=[],
            regulatory_justification="No regulatory synthesis generated due to pipeline failure.",
            evidence_gaps=[f"Downstream failure: {failure_reason}"],
            reviewer_notes="CRITICAL: Pipeline failed prior to complete verification. Inspect stage logs.",
        )

        verifier_summary = ReportVerifierSummary(
            quote_checks_total=0,
            quote_checks_passed=0,
            quote_checks_failed=0,
            faithfulness_score=0.0,
            normalized_match_policy=NORMALIZED_MATCH_POLICY,
        )

        return AuditReport(
            audit_id=audit_id,
            applicant_ref=applicant_ref,
            generated_at=now,
            mode=self.settings.mode,
            status="failed",
            model=model_summary,
            prediction=pred_summary,
            explanation=expl_summary,
            counterfactuals=cf_candidates,
            search_trace=search_trace,
            evidence=[],
            dossier=dossier,
            verifier=verifier_summary,
            human_review_required=True,
        )
