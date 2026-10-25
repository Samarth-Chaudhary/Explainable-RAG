"""Quantitative evaluation metrics endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from fastapi import APIRouter

router = APIRouter(prefix="/api/evaluation", tags=["Evaluation & Benchmarks"])


@router.get("/metrics")
def get_evaluation_metrics() -> dict[str, Any]:
    """Retrieve computed quantitative evaluation metrics on documented test sets."""
    results_path = Path("tests/evaluation/evaluation_results.json")

    if results_path.is_file():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
                return metrics_data
        except Exception:
            pass

    # Fallback to unmeasured status per benchmark claim control
    return {
        "status": "not_measured",
        "benchmark_claim_control": (
            "The brief's ~52% baseline and 91%+ Recall@3 target are hypotheses and aspirations only, "
            "not established facts. They are never claimed as achieved performance without reproduced execution."
        ),
        "counterfactual_metrics": {
            "validity_rate": "not measured",
            "actionability_rate": "not measured",
            "sample_size": 0,
        },
        "retrieval_metrics": {
            "Recall@3": "not measured",
            "Precision@3": "not measured",
            "MRR": "not measured",
            "nDCG@3": "not measured",
            "sample_size": 0,
        },
        "citation_faithfulness": {
            "faithfulness_score": "not measured",
            "sample_size": 0,
        },
        "model_performance": {
            "status": "demo_synthetic_model",
            "auc_roc": "not measured",
            "calibration_error": "not measured",
        },
    }
