"""Unit tests for Pillar D: Hybrid Retrieval (FAISS + BM25 + RRF) and Quality Controls."""

import pytest
from pathlib import Path

from backend.app.schemas.handoff import Chunk, HybridQueryPayload
from backend.app.schemas.source_registry import SourceRegistryRecord
from backend.app.services.document_ingestion import DocumentIngestionService
from backend.app.services.retrieval import (
    HybridRetrievalService,
    LocalDeterministicEmbedder,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from tests.evaluation.eval_metrics import run_retrieval_benchmark


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="SYN-POL-FOIR-001-P1-C1",
            source_id="SYN-POL-FOIR-001",
            text="Section 2.1 FOIR Ceiling. Borrowers must maintain a Fixed Obligation to Income Ratio not exceeding 65 percent.",
            page_number=1,
            section_title="Section 2.1 FOIR Ceiling",
            token_count=18,
            metadata={
                "jurisdiction": "IN",
                "source_family": "underwriting_rules",
                "access_classification": "internal_confidential",
                "status": "active",
            },
        ),
        Chunk(
            chunk_id="SYN-POL-FOIR-001-P2-C2",
            source_id="SYN-POL-FOIR-001",
            text="Section 2.2 Tenor Restructuring. In cases of elevated FOIR, credit officers may extend loan repayment tenor up to 60 months.",
            page_number=2,
            section_title="Section 2.2 Tenor Restructuring",
            token_count=20,
            metadata={
                "jurisdiction": "IN",
                "source_family": "underwriting_rules",
                "access_classification": "internal_confidential",
                "status": "active",
            },
        ),
        Chunk(
            chunk_id="SYN-POL-LTV-001-P1-C1",
            source_id="SYN-POL-LTV-001",
            text="Section 3.1 Collateral Valuation and LTV Limits. Maximum permissible Loan-to-Value exposure is capped at 75 percent.",
            page_number=1,
            section_title="Section 3.1 Collateral Valuation and LTV Limits",
            token_count=18,
            metadata={
                "jurisdiction": "IN",
                "source_family": "collateral_policy",
                "access_classification": "internal_confidential",
                "status": "active",
            },
        ),
        Chunk(
            chunk_id="SYN-POL-SUPERSEDED-001-P1-C1",
            source_id="SYN-POL-SUPERSEDED-001",
            text="Section 1.0 Obsolete Guidance. Historic lending guidelines (superseded and inactive).",
            page_number=1,
            section_title="Section 1.0 Obsolete Guidance",
            token_count=10,
            metadata={
                "jurisdiction": "IN",
                "source_family": "legacy_rules",
                "access_classification": "internal_confidential",
                "status": "superseded",
            },
        ),
    ]


def test_local_deterministic_embedder():
    """Test local embedding vector properties: dimension, normalization, determinism."""
    embedder = LocalDeterministicEmbedder(dimension=128)
    text1 = "Fixed obligation to income ratio threshold 65 percent"
    vec1 = embedder.embed_text(text1)

    assert vec1.shape == (128,)
    norm = float(sum(vec1**2) ** 0.5)
    assert abs(norm - 1.0) < 1e-4, "Vector must be unit normalized"

    # Determinism: same string gives identical vector
    vec2 = embedder.embed_text(text1)
    assert (vec1 == vec2).all()


def test_hybrid_index_build(sample_chunks: list[Chunk]):
    """Test building FAISS and BM25 indexes and deterministic version hashing."""
    service = HybridRetrievalService()
    build_res = service.build_indexes(sample_chunks)

    assert build_res.chunk_count == len(sample_chunks)
    assert len(build_res.index_version) == 64
    assert service.index_version == build_res.index_version


def test_filter_before_rank_and_superseded_exclusion(sample_chunks: list[Chunk]):
    """Test that metadata filters and superseded exclusion execute before ranking."""
    service = HybridRetrievalService(include_superseded=False)
    service.build_indexes(sample_chunks)

    payload = HybridQueryPayload(
        dense_query="FOIR debt burden repayment capacity limits",
        bm25_terms=["FOIR", "obligation", "repayment"],
        metadata_filters={"jurisdiction": "IN"},
        top_k=5,
        source_families=["underwriting_rules"],
    )

    result = service.retrieve(payload, user_role="credit_officer")
    assert not result.no_evidence_found
    retrieved_ids = [e.chunk.chunk_id for e in result.evidence]

    # Active FOIR chunks should be retrieved
    assert "SYN-POL-FOIR-001-P1-C1" in retrieved_ids
    # Superseded chunk should be excluded
    assert "SYN-POL-SUPERSEDED-001-P1-C1" not in retrieved_ids


def test_reciprocal_rank_fusion_determinism(sample_chunks: list[Chunk]):
    """Test that RRF fusion scores produce deterministic, ordered rankings."""
    service = HybridRetrievalService()
    service.build_indexes(sample_chunks)

    payload = HybridQueryPayload(
        dense_query="Collateral valuation LTV haircut rules",
        bm25_terms=["LTV", "collateral", "valuation"],
        metadata_filters={},
        top_k=3,
        source_families=["collateral_policy"],
    )

    res1 = service.retrieve(payload, user_role="credit_officer")
    res2 = service.retrieve(payload, user_role="credit_officer")

    # Identical queries yield identical fused rankings
    ids1 = [e.chunk.chunk_id for e in res1.evidence]
    ids2 = [e.chunk.chunk_id for e in res2.evidence]
    assert ids1 == ids2
    assert "SYN-POL-LTV-001-P1-C1" in ids1


def test_role_based_access_control(sample_chunks: list[Chunk]):
    """Test that unauthorized user roles cannot retrieve internal confidential policies."""
    service = HybridRetrievalService()
    service.build_indexes(sample_chunks)

    payload = HybridQueryPayload(
        dense_query="FOIR and LTV policies",
        bm25_terms=["FOIR", "LTV"],
        metadata_filters={},
        top_k=3,
        source_families=[],
    )

    # Customer role is not authorized for internal policies
    result_customer = service.retrieve(payload, user_role="customer")
    assert result_customer.no_evidence_found is True
    assert len(result_customer.evidence) == 0

    # Credit officer is authorized
    result_officer = service.retrieve(payload, user_role="credit_officer")
    assert result_officer.no_evidence_found is False
    assert len(result_officer.evidence) > 0


def test_empty_corpus_or_no_match_returns_evidence_gap():
    """Test that empty corpus or unmapped query returns clear gap without inventing law."""
    service = HybridRetrievalService()
    service.build_indexes([])  # empty index

    payload = HybridQueryPayload(
        dense_query="Some unknown query with no corpus",
        bm25_terms=["unknown", "policy"],
        metadata_filters={},
        top_k=3,
        source_families=[],
    )

    result = service.retrieve(payload, user_role="credit_officer")
    assert result.no_evidence_found is True
    assert len(result.evidence) == 0
    assert "No authoritative evidence found in the configured corpus." in result.missing_coverage


def test_retrieval_evaluation_metrics_and_benchmark(sample_chunks: list[Chunk]):
    """Test IR evaluation metric functions and run labelled benchmark."""
    # Test metric math directly
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    expected = ["doc2", "doc5"]

    r3 = recall_at_k(retrieved, expected, k=3)
    p3 = precision_at_k(retrieved, expected, k=3)
    mrr = mean_reciprocal_rank(retrieved, expected)
    ndcg3 = ndcg_at_k(retrieved, expected, k=3)

    assert r3 == 0.5  # 1 of 2 expected in top 3
    assert abs(p3 - 1 / 3) < 1e-4  # 1 of 3 retrieved is relevant
    assert mrr == 0.5  # first relevant item is at rank 2 -> 1/2
    assert ndcg3 > 0.0

    # Run benchmark against labelled eval set
    service = HybridRetrievalService()
    service.build_indexes(sample_chunks)

    eval_path = Path("tests/evaluation/labelled_eval_set.json")
    bench_results = run_retrieval_benchmark(service, eval_path, k=3)

    assert bench_results["evaluated_cases_count"] >= 3
    assert "Recall@3" in bench_results["macro_averaged_metrics"]
    assert "MRR" in bench_results["macro_averaged_metrics"]
    # Check that aspirations are clearly recorded as aspirations, not claimed as achieved facts
    assert "naive_baseline_hypothesis_recall_at_3" in bench_results["aspirations"]
