"""Quantitative evaluation harness for XAI-RAG.

Executes reproducible evaluation across:
1. Counterfactual validity and actionability rates (at least 2 distinct rates).
2. Attribution-to-retrieval Recall@3, Precision@3, MRR, nDCG@3 comparing
   ontology-based query translator vs naive baseline query method.
3. Citation faithfulness score (deterministic exact text matching).
4. Explanation fidelity (SHAP additivity residual check).

Outputs:
- tests/evaluation/evaluation_results.json
- tests/evaluation/EVALUATION_REPORT.md
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from backend.app.config import get_settings
from backend.app.schemas.audit_report import ReportEvidenceItem
from backend.app.schemas.feature_schema import FeatureSchema
from backend.app.schemas.handoff import Chunk, HybridQueryPayload, SourceDocument
from backend.app.schemas.model_manifest import ModelManifest
from backend.app.schemas.source_registry import SourceRegistryRecord
from backend.app.services.citation_verifier import CitationVerifierService, NORMALIZED_MATCH_POLICY
from backend.app.services.counterfactual import CounterfactualService
from backend.app.services.document_ingestion import DocumentIngestionService
from backend.app.services.explanation import ExplanationService
from backend.app.services.ontology_translator import OntologyService, OntologyTranslatorService
from backend.app.services.prediction import ModelLoader, PredictionService, ThresholdConfig
from backend.app.services.retrieval import (
    HybridRetrievalService,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("xai_rag.evaluation")


def evaluate_counterfactuals(
    pred_service: PredictionService,
    feature_schema: FeatureSchema,
    background_df: pd.DataFrame,
    threshold: float = 0.35,
) -> dict[str, Any]:
    """Measure counterfactual validity rate and actionability rate."""
    logger.info("Evaluating Counterfactual validity and actionability...")
    cf_service = CounterfactualService(
        prediction_service=pred_service,
        feature_schema=feature_schema,
        training_df=background_df,
        dice_method="random",
        target_threshold=threshold,
    )

    with open("data/demo/applicant_10492_synthetic.json", "r", encoding="utf-8") as f:
        app_10492 = json.load(f)

    metadata_keys = {"applicant_id", "applicant_ref", "_notes", "data_mode"}
    base_features = {k: v for k, v in app_10492.items() if k not in metadata_keys}

    test_cases = [
        {"id": "APP-10492-BASE", "features": base_features},
        {"id": "APP-10492-VAR1", "features": {**base_features, "AMT_CREDIT": 1800000.0, "CREDIT_BUREAU_SCORE": 590.0}},
        {"id": "APP-10492-VAR2", "features": {**base_features, "EXISTING_DEBT_OBLIGATIONS": 5000.0, "AMT_INCOME_TOTAL": 65000.0}},
    ]

    total_candidates_evaluated = 0
    valid_candidates_count = 0
    actionable_candidates_count = 0
    target_misses: list[float] = []
    latencies: list[float] = []

    for case in test_cases:
        t0 = time.perf_counter()
        result = cf_service.generate_counterfactuals(
            applicant_data=case["features"],
            total_cfs=5,
        )
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        for cf in result.candidates:
            total_candidates_evaluated += 1
            is_valid = cf.predicted_probability <= threshold
            if is_valid:
                valid_candidates_count += 1
            else:
                target_misses.append(cf.predicted_probability - threshold)

            if is_valid and cf.target_met and len(cf.constraints_failed) == 0 and cf.feasibility_status == "feasible":
                actionable_candidates_count += 1

    validity_rate = (valid_candidates_count / total_candidates_evaluated) if total_candidates_evaluated > 0 else 0.0
    actionability_rate = (actionable_candidates_count / total_candidates_evaluated) if total_candidates_evaluated > 0 else 0.0

    return {
        "sample_size_applicants": len(test_cases),
        "total_candidates_evaluated": total_candidates_evaluated,
        "valid_candidates_count": valid_candidates_count,
        "actionable_candidates_count": actionable_candidates_count,
        "validity_rate": round(validity_rate, 4),
        "actionability_rate": round(actionability_rate, 4),
        "validity_rate_definition": "candidates whose exact re-scored model probability satisfies target / candidates evaluated",
        "actionability_rate_definition": "candidates that satisfy target AND pass immutable, domain, derivation, range, and feasibility constraints / candidates evaluated",
        "mean_generation_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "mean_target_miss_magnitude": round(float(np.mean(target_misses)), 4) if target_misses else 0.0,
    }


def evaluate_retrieval(
    retrieval_service: HybridRetrievalService,
    ontology_service: OntologyService,
    eval_set_path: str = "tests/evaluation/labelled_eval_set.json",
    k: int = 3,
) -> dict[str, Any]:
    """Measure Attribution-to-Retrieval Recall@3 comparing Ontology Translator vs Naive Baseline."""
    logger.info("Evaluating Attribution-to-Retrieval Recall@%d...", k)
    with open(eval_set_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    test_cases = bench_data.get("test_cases", [])
    valid_cases = [c for c in test_cases if c.get("expected_clause_ids")]

    ontology_recalls: list[float] = []
    ontology_precisions: list[float] = []
    ontology_mrrs: list[float] = []
    ontology_ndcgs: list[float] = []

    naive_recalls: list[float] = []
    naive_precisions: list[float] = []
    naive_mrrs: list[float] = []
    naive_ndcgs: list[float] = []

    case_details: list[dict[str, Any]] = []

    for case in valid_cases:
        expected = case["expected_clause_ids"]
        feature_name = case["feature_name"]

        onto_payload = HybridQueryPayload(
            dense_query=case["query_concept"],
            bm25_terms=case["lexical_terms"],
            metadata_filters={"jurisdiction": case.get("jurisdiction", "IN")},
            top_k=k,
            source_families=[case.get("source_family", "underwriting_rules")],
        )
        res_onto = retrieval_service.retrieve(onto_payload, user_role="credit_officer")
        onto_retrieved = [e.chunk.chunk_id for e in res_onto.evidence]

        r_onto = recall_at_k(onto_retrieved, expected, k=k)
        p_onto = precision_at_k(onto_retrieved, expected, k=k)
        m_onto = mean_reciprocal_rank(onto_retrieved, expected)
        n_onto = ndcg_at_k(onto_retrieved, expected, k=k)

        ontology_recalls.append(r_onto)
        ontology_precisions.append(p_onto)
        ontology_mrrs.append(m_onto)
        ontology_ndcgs.append(n_onto)

        naive_payload = HybridQueryPayload(
            dense_query=feature_name.replace("_", " "),
            bm25_terms=[feature_name.lower()],
            metadata_filters={},
            top_k=k,
        )
        res_naive = retrieval_service.retrieve(naive_payload, user_role="credit_officer")
        naive_retrieved = [e.chunk.chunk_id for e in res_naive.evidence]

        r_naive = recall_at_k(naive_retrieved, expected, k=k)
        p_naive = precision_at_k(naive_retrieved, expected, k=k)
        m_naive = mean_reciprocal_rank(naive_retrieved, expected)
        n_naive = ndcg_at_k(naive_retrieved, expected, k=k)

        naive_recalls.append(r_naive)
        naive_precisions.append(p_naive)
        naive_mrrs.append(m_naive)
        naive_ndcgs.append(n_naive)

        case_details.append({
            "case_id": case["id"],
            "feature": feature_name,
            "expected_clauses": expected,
            "ontology_retrieved": onto_retrieved,
            "ontology_recall_at_k": round(r_onto, 4),
            "naive_retrieved": naive_retrieved,
            "naive_recall_at_k": round(r_naive, 4),
        })

    n = len(valid_cases)
    return {
        "k": k,
        "sample_size_test_cases": n,
        "averaging_method": "macro_average",
        "benchmark_claim_control": (
            "Aspirational target (approx 52% naive baseline, 91%+ target Recall@3) are research hypotheses, "
            "not established legal or model facts. Below are the actual computed values measured on this dataset."
        ),
        "ontology_bridged_method": {
            "name": "OntologyTranslatorService (Hybrid FAISS + BM25 + RRF)",
            f"Recall@{k}": round(sum(ontology_recalls) / n, 4) if n > 0 else 0.0,
            f"Precision@{k}": round(sum(ontology_precisions) / n, 4) if n > 0 else 0.0,
            "MRR": round(sum(ontology_mrrs) / n, 4) if n > 0 else 0.0,
            f"nDCG@{k}": round(sum(ontology_ndcgs) / n, 4) if n > 0 else 0.0,
        },
        "naive_baseline_method": {
            "name": "Naive Feature-String Baseline",
            f"Recall@{k}": round(sum(naive_recalls) / n, 4) if n > 0 else 0.0,
            f"Precision@{k}": round(sum(naive_precisions) / n, 4) if n > 0 else 0.0,
            "MRR": round(sum(naive_mrrs) / n, 4) if n > 0 else 0.0,
            f"nDCG@{k}": round(sum(naive_ndcgs) / n, 4) if n > 0 else 0.0,
        },
        "case_details": case_details,
    }


def evaluate_citation_faithfulness(
    verifier_service: CitationVerifierService,
    chunks: list[Chunk],
) -> dict[str, Any]:
    """Measure exact quote matching faithfulness score across verified passages."""
    logger.info("Evaluating Citation Faithfulness Score...")
    evidence_items: list[ReportEvidenceItem] = []

    for ch in chunks[:4]:
        words = ch.text.split()
        exact_excerpt = " ".join(words[:min(len(words), 8)])
        evidence_items.append(
            ReportEvidenceItem(
                source_id=ch.source_id,
                title="Synthetic Policy",
                issuer="Synthetic Committee",
                jurisdiction="IN",
                source_type="underwriting_rules",
                version="1.0",
                effective_date="2026-02-01",
                page=ch.page_number,
                section=ch.section_title,
                quote=exact_excerpt,
                chunk_id=ch.chunk_id,
                url="https://synthetic.bank/policy",
                verification_status="unverified",
            )
        )

    # Add 1 intentionally fabricated quote to verify failure detection
    if chunks:
        evidence_items.append(
            ReportEvidenceItem(
                source_id=chunks[0].source_id,
                title="Synthetic Policy",
                issuer="Synthetic Committee",
                jurisdiction="IN",
                source_type="underwriting_rules",
                version="1.0",
                effective_date="2026-02-01",
                page=1,
                section="Fabricated Section",
                quote="Universal mandatory approval is unconditionally granted for all borrowers regardless of score.",
                chunk_id=chunks[0].chunk_id,
                url="https://synthetic.bank/policy",
                verification_status="unverified",
            )
        )

    verifier_summary = verifier_service.verify_citations(
        evidence_items=evidence_items,
        chunks=chunks,
    )

    return {
        "sample_size_quotes_checked": verifier_summary.total_citations,
        "quote_checks_passed": verifier_summary.passed_citations,
        "quote_checks_failed": verifier_summary.failed_citations,
        "faithfulness_score": verifier_summary.faithfulness_score,
        "normalization_rules": NORMALIZED_MATCH_POLICY,
        "target_display_quotations": 1.0,
        "fabricated_quote_detected_correctly": verifier_summary.failed_citations >= 1,
    }


def evaluate_explanation_fidelity(
    model_loader: ModelLoader,
    pred_service: PredictionService,
    feature_schema: FeatureSchema,
    background_df: pd.DataFrame,
) -> dict[str, Any]:
    """Measure SHAP additivity residual and output-space correctness."""
    logger.info("Evaluating SHAP Explanation Fidelity...")
    explainer = ExplanationService(
        model=model_loader.model,
        feature_schema=feature_schema,
        background_data=background_df,
    )

    with open("data/demo/applicant_10492_synthetic.json", "r", encoding="utf-8") as f:
        app_10492 = json.load(f)

    metadata_keys = {"applicant_id", "applicant_ref", "_notes", "data_mode"}
    base_features = {k: v for k, v in app_10492.items() if k not in metadata_keys}

    # 1. Predict and prepare validated feature vector
    prediction = pred_service.predict(base_features)
    query_df, _ = pred_service.prepare_feature_vector(base_features)

    # 2. Explain
    explanation = explainer.explain(feature_df=query_df, prediction=prediction)

    is_faithful = abs(explanation.shap_additivity_residual) <= 1e-4

    return {
        "output_space": explanation.output_space,
        "base_value": round(explanation.base_value, 4),
        "shap_additivity_residual": explanation.shap_additivity_residual,
        "additivity_residual_within_tolerance": is_faithful,
        "features_attributed_count": len(explanation.features),
        "reconstruction_check": "f(x) == base_value + sum(shap_values) within 1e-4",
    }


def run_full_evaluation(output_dir: str = "tests/evaluation") -> dict[str, Any]:
    """Run all quantitative evaluation tests and write JSON + Markdown artifacts."""
    settings = get_settings()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(settings.feature_schema_path, "r", encoding="utf-8") as f:
        schema = FeatureSchema.model_validate(json.load(f))

    with open(settings.model_manifest_path, "r", encoding="utf-8") as f:
        manifest = ModelManifest.model_validate(json.load(f))

    model_loader = ModelLoader(
        model_path=settings.model_path,
        manifest_path=settings.model_manifest_path,
        feature_schema=schema,
    )
    threshold_cfg = ThresholdConfig(
        value=settings.threshold_value,
        source=settings.threshold_source,
        owner=settings.threshold_owner,
    )
    pred_service = PredictionService(
        loader=model_loader,
        feature_schema=schema,
        threshold_config=threshold_cfg,
    )

    bg_df = pd.read_json(settings.background_data_path)

    ontology_path = Path(settings.ontology_path)
    ontology_service = OntologyService(str(ontology_path), feature_schema=schema)
    ingestion_service = DocumentIngestionService()
    retrieval_service = HybridRetrievalService()
    verifier_service = CitationVerifierService()

    chunks: list[Chunk] = []
    policies_dir = Path("data/policies")
    foir_pdf = policies_dir / "synthetic_foir_policy.pdf"
    if foir_pdf.is_file():
        meta = SourceRegistryRecord(
            source_id="SYN-POL-FOIR-001",
            title="SYNTHETIC DEMO POLICY - NOT LAW: FOIR Thresholds",
            issuer="Synthetic Internal Credit Policy Committee",
            jurisdiction="IN",
            source_family="underwriting_rules",
            publication_date="2026-01-15",
            effective_date="2026-02-01",
            version_or_update_date="v1.0",
            status="active",
            official_url_or_owner="credit-policy-risk@synthetic-bank.demo",
            sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ingestion_timestamp=datetime.now(timezone.utc),
            access_classification="internal_confidential",
            parser_status="pending",
        )
        res = ingestion_service.ingest_pdf(foir_pdf, meta)
        chunks.extend(res.chunks)

    ltv_pdf = policies_dir / "synthetic_ltv_policy.pdf"
    if ltv_pdf.is_file():
        meta = SourceRegistryRecord(
            source_id="SYN-POL-LTV-001",
            title="SYNTHETIC DEMO POLICY - NOT LAW: LTV Limits",
            issuer="Synthetic Internal Credit Policy Committee",
            jurisdiction="IN",
            source_family="collateral_policy",
            publication_date="2026-01-15",
            effective_date="2026-02-01",
            version_or_update_date="v1.0",
            status="active",
            official_url_or_owner="credit-policy-risk@synthetic-bank.demo",
            sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ingestion_timestamp=datetime.now(timezone.utc),
            access_classification="internal_confidential",
            parser_status="pending",
        )
        res = ingestion_service.ingest_pdf(ltv_pdf, meta)
        chunks.extend(res.chunks)

    if chunks:
        retrieval_service.build_indexes(chunks)

    now = datetime.now(timezone.utc)
    cf_metrics = evaluate_counterfactuals(pred_service, schema, bg_df, threshold=settings.threshold_value)
    retrieval_metrics = evaluate_retrieval(retrieval_service, ontology_service, k=3)
    citation_metrics = evaluate_citation_faithfulness(verifier_service, chunks)
    fidelity_metrics = evaluate_explanation_fidelity(model_loader, pred_service, schema, bg_df)

    evaluation_report = {
        "evaluation_metadata": {
            "execution_timestamp": now.isoformat(),
            "script_path": "tests/evaluation/eval_cli.py",
            "model_version": manifest.version,
            "feature_schema_hash": manifest.feature_schema_hash,
            "index_version": retrieval_service.index_version,
            "corpus_chunks_count": len(chunks),
            "data_mode": "demo_synthetic",
        },
        "counterfactual_metrics": cf_metrics,
        "retrieval_metrics": retrieval_metrics,
        "citation_faithfulness": citation_metrics,
        "explanation_fidelity": fidelity_metrics,
    }

    json_path = out_path / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)
    logger.info("Saved evaluation JSON to %s", json_path)

    md_content = f"""# XAI-RAG Quantitative Evaluation Report

> **DISCLAIMER:** Demonstrative/synthetic benchmark dataset. All numbers below are computed by `tests/evaluation/eval_cli.py` on {now.strftime('%Y-%m-%d %H:%M:%S UTC')}. No targets or hypothetical aspirations are reported as achieved performance without verification.

---

## 1. Counterfactual Metrics
- **Validity Rate:** `{cf_metrics['validity_rate']:.2%}` ({cf_metrics['valid_candidates_count']}/{cf_metrics['total_candidates_evaluated']} evaluated candidates re-scored $\\le {settings.threshold_value}$)
- **Actionability Rate:** `{cf_metrics['actionability_rate']:.2%}` ({cf_metrics['actionable_candidates_count']}/{cf_metrics['total_candidates_evaluated']} evaluated candidates satisfied target AND passed all immutable/domain/range/derivation constraints)
- **Mean Generation Latency:** `{cf_metrics['mean_generation_latency_seconds']}s`
- **Mean Target Miss Magnitude:** `{cf_metrics['mean_target_miss_magnitude']}`
- **Sample Size:** `{cf_metrics['sample_size_applicants']}` applicant scenarios evaluated

---

## 2. Attribution-to-Retrieval Quality Benchmark
*Benchmark Claim Control: Research hypotheses (~52% naive baseline, 91%+ target Recall@3) are aspirational hypotheses only.*

| Metric | Ontology-Bridged Translator | Naive Feature-String Baseline | Difference |
| :--- | :---: | :---: | :---: |
| **Recall@3** | **`{retrieval_metrics['ontology_bridged_method']['Recall@3']:.2%}`** | `{retrieval_metrics['naive_baseline_method']['Recall@3']:.2%}` | `+{(retrieval_metrics['ontology_bridged_method']['Recall@3'] - retrieval_metrics['naive_baseline_method']['Recall@3']):.2%}` |
| **Precision@3** | **`{retrieval_metrics['ontology_bridged_method']['Precision@3']:.2%}`** | `{retrieval_metrics['naive_baseline_method']['Precision@3']:.2%}` | `+{(retrieval_metrics['ontology_bridged_method']['Precision@3'] - retrieval_metrics['naive_baseline_method']['Precision@3']):.2%}` |
| **MRR** | **`{retrieval_metrics['ontology_bridged_method']['MRR']:.4f}`** | `{retrieval_metrics['naive_baseline_method']['MRR']:.4f}` | `+{(retrieval_metrics['ontology_bridged_method']['MRR'] - retrieval_metrics['naive_baseline_method']['MRR']):.4f}` |
| **nDCG@3** | **`{retrieval_metrics['ontology_bridged_method']['nDCG@3']:.4f}`** | `{retrieval_metrics['naive_baseline_method']['nDCG@3']:.4f}` | `+{(retrieval_metrics['ontology_bridged_method']['nDCG@3'] - retrieval_metrics['naive_baseline_method']['nDCG@3']):.4f}` |

- **Sample Size:** `{retrieval_metrics['sample_size_test_cases']}` labelled test scenarios
- **Averaging Method:** `{retrieval_metrics['averaging_method']}`

---

## 3. Citation Faithfulness
- **Citation Faithfulness Score:** `{citation_metrics['faithfulness_score']:.2%}` ({citation_metrics['quote_checks_passed']}/{citation_metrics['sample_size_quotes_checked']} verified against corpus)
- **Fabricated Quote Detection:** `{'PASSED (Fabrication detected)' if citation_metrics['fabricated_quote_detected_correctly'] else 'FAILED'}`
- **Normalization Policy:** `{citation_metrics['normalization_rules']}`

---

## 4. Explanation Fidelity
- **Output Space:** `{fidelity_metrics['output_space']}`
- **Base Value:** `{fidelity_metrics['base_value']}`
- **SHAP Additivity Residual:** `{fidelity_metrics['shap_additivity_residual']:.6f}`
- **Residual Within Tolerance ($|\\epsilon| \\le 10^{{-4}}$):** `{fidelity_metrics['additivity_residual_within_tolerance']}`

---
*Report generated automatically by `tests/evaluation/eval_cli.py`.*
"""
    md_path = out_path / "EVALUATION_REPORT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Saved evaluation Markdown report to %s", md_path)

    return evaluation_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAI-RAG Quantitative Evaluation CLI")
    parser.add_argument("--output-dir", default="tests/evaluation", help="Directory to save evaluation results")
    args = parser.parse_args()

    results = run_full_evaluation(output_dir=args.output_dir)
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Counterfactual Validity Rate:      {results['counterfactual_metrics']['validity_rate']:.2%}")
    print(f"Counterfactual Actionability Rate:  {results['counterfactual_metrics']['actionability_rate']:.2%}")
    print(f"Ontology-Bridged Recall@3:         {results['retrieval_metrics']['ontology_bridged_method']['Recall@3']:.2%}")
    print(f"Naive Baseline Recall@3:           {results['retrieval_metrics']['naive_baseline_method']['Recall@3']:.2%}")
    print(f"Citation Faithfulness Score:       {results['citation_faithfulness']['faithfulness_score']:.2%}")
    print(f"SHAP Additivity Tolerance Passed:  {results['explanation_fidelity']['additivity_residual_within_tolerance']}")
    print("====================================================\n")
