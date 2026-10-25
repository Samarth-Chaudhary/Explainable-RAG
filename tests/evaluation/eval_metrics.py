"""Evaluation metrics and benchmark runner for Part 2 retrieval quality controls.

Implements Recall@K, Precision@K, MRR, and nDCG@K with explicit macro-averaging.
Note: brief's ~52% baseline and 91%+ target are hypotheses/aspirations only,
not hardcoded achieved figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.schemas.handoff import Chunk
from backend.app.services.retrieval import (
    HybridRetrievalService,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def run_retrieval_benchmark(
    retrieval_service: HybridRetrievalService,
    eval_set_path: str | Path,
    k: int = 3,
) -> dict[str, Any]:
    """Execute evaluation benchmark across labelled queries.

    Computes macro-averaged Recall@k, Precision@k, MRR, and nDCG@k.
    """
    path = Path(eval_set_path)
    with open(path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    test_cases = bench_data.get("test_cases", [])
    recalls: list[float] = []
    precisions: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    case_results: list[dict[str, Any]] = []

    for case in test_cases:
        expected = case.get("expected_clause_ids", [])
        # If expected is empty (e.g. unknown coverage test), skip from positive retrieval metrics
        if not expected:
            continue

        # Build query payload
        from backend.app.schemas.handoff import HybridQueryPayload

        payload = HybridQueryPayload(
            dense_query=case["query_concept"],
            bm25_terms=case["lexical_terms"],
            metadata_filters={"jurisdiction": case.get("jurisdiction", "IN")},
            top_k=k,
            source_families=[case.get("source_family", "underwriting_rules")],
        )

        retrieval_res = retrieval_service.retrieve(payload, user_role="credit_officer")
        retrieved_ids = [e.chunk.chunk_id for e in retrieval_res.evidence]

        r = recall_at_k(retrieved_ids, expected, k=k)
        p = precision_at_k(retrieved_ids, expected, k=k)
        m = mean_reciprocal_rank(retrieved_ids, expected)
        n = ndcg_at_k(retrieved_ids, expected, k=k)

        recalls.append(r)
        precisions.append(p)
        mrrs.append(m)
        ndcgs.append(n)

        case_results.append({
            "id": case["id"],
            "feature": case["feature_name"],
            "expected": expected,
            "retrieved": retrieved_ids,
            "recall": round(r, 4),
            "precision": round(p, 4),
            "mrr": round(m, 4),
            "ndcg": round(n, 4),
        })

    num_cases = len(recalls)
    macro_recall = (sum(recalls) / num_cases) if num_cases > 0 else 0.0
    macro_precision = (sum(precisions) / num_cases) if num_cases > 0 else 0.0
    macro_mrr = (sum(mrrs) / num_cases) if num_cases > 0 else 0.0
    macro_ndcg = (sum(ndcgs) / num_cases) if num_cases > 0 else 0.0

    return {
        "k": k,
        "evaluated_cases_count": num_cases,
        "macro_averaged_metrics": {
            f"Recall@{k}": round(macro_recall, 4),
            f"Precision@{k}": round(macro_precision, 4),
            "MRR": round(macro_mrr, 4),
            f"nDCG@{k}": round(macro_ndcg, 4),
        },
        "averaging_method": "macro_average (unweighted mean across labelled cases with positive ground truth)",
        "aspirations": bench_data.get("aspirations", {}),
        "case_results": case_results,
    }
