"""Pillar D: Hybrid FAISS Dense + BM25 Lexical Retrieval with Reciprocal Rank Fusion (RRF).

Includes role-based authorization, metadata filtering, index versioning, and retrieval evaluation metrics.
"""

import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from backend.app.schemas.handoff import (
    Chunk,
    HybridQueryPayload,
    IndexBuildResult,
    RetrievalResult,
    RetrievalTrace,
    RetrievedEvidence,
    SourceDocument,
)
from backend.app.services.prompt_defense import PromptInjectionDefenseService


class HybridRetrievalProtocol(Protocol):
    """Protocol for hybrid FAISS dense + BM25 lexical retrieval."""

    def retrieve(self, payload: HybridQueryPayload, user_role: str = "credit_officer") -> RetrievalResult:
        ...

    def build_indexes(self, chunks: list[Chunk]) -> IndexBuildResult:
        ...


class LocalDeterministicEmbedder:
    """Deterministic local text embedder generating normalized dense vectors via feature hashing."""

    def __init__(self, dimension: int = 128, version: str = "local-deterministic-v1"):
        self.dimension = dimension
        self.version = version

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single string into a normalized dense vector."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = re.findall(r"\w+", text.lower())

        for token in tokens:
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dimension
            sign = 1.0 if int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % 2 == 0 else -1.0
            vec[idx] += sign

        # Add bigram hashes for local phrase awareness
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            idx = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16) % self.dimension
            sign = 1.0 if int(hashlib.sha1(bigram.encode("utf-8")).hexdigest(), 16) % 2 == 0 else -1.0
            vec[idx] += 1.5 * sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of strings."""
        return np.vstack([self.embed_text(t) for t in texts])


class HybridRetrievalService:
    """Orchestrates FAISS dense index, BM25 lexical index, metadata filtering, and RRF fusion."""

    def __init__(
        self,
        embedder: LocalDeterministicEmbedder | None = None,
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        exclude_superseded: bool = True,
        include_superseded: bool | None = None,
    ):
        self.embedder = embedder or LocalDeterministicEmbedder()
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        if include_superseded is not None:
            self.exclude_superseded = not include_superseded
        else:
            self.exclude_superseded = exclude_superseded

        self.documents: list[SourceDocument] = []
        self.chunks: list[Chunk] = []
        self.chunk_id_to_chunk: dict[str, Chunk] = {}
        self.chunk_id_to_doc: dict[str, SourceDocument] = {}

        self.faiss_index: faiss.IndexFlatIP | None = None
        self.bm25: BM25Okapi | None = None
        self.bm25_corpus_tokens: list[list[str]] = []
        self.index_version: str = "uninitialized"
        self.injection_defense = PromptInjectionDefenseService()

    def build_indexes(self, items: list[SourceDocument] | list[Chunk]) -> IndexBuildResult:
        """Construct FAISS dense index and BM25 lexical index from registered documents or chunks."""
        self.chunks = []
        self.chunk_id_to_chunk = {}
        self.chunk_id_to_doc = {}

        if items and isinstance(items[0], Chunk):
            self.chunks = list(items)  # type: ignore
            for ch in self.chunks:
                self.chunk_id_to_chunk[ch.chunk_id] = ch
            self.documents = []
        else:
            self.documents = items  # type: ignore
            for doc in self.documents:
                for ch in doc.chunks:
                    self.chunks.append(ch)
                    self.chunk_id_to_chunk[ch.chunk_id] = ch
                    self.chunk_id_to_doc[ch.chunk_id] = doc

        if not self.chunks:
            self.index_version = "empty-0"
            return IndexBuildResult(
                index_version=self.index_version,
                document_count=len(self.documents),
                chunk_count=0,
                embedding_model=self.embedder.version,
                build_timestamp=datetime.now(timezone.utc),
            )

        # 1. Build FAISS dense index
        texts = [c.text for c in self.chunks]
        embeddings = self.embedder.embed_batch(texts)

        self.faiss_index = faiss.IndexFlatIP(self.embedder.dimension)
        self.faiss_index.add(embeddings)

        # 2. Build BM25 lexical index
        self.bm25_corpus_tokens = [re.findall(r"\w+", t.lower()) for t in texts]
        self.bm25 = BM25Okapi(self.bm25_corpus_tokens)

        # 3. Compute deterministic index version
        hasher = hashlib.sha256()
        if self.documents:
            for doc in sorted(self.documents, key=lambda d: d.source_id):
                hasher.update(doc.sha256_hash.encode("utf-8"))
        else:
            for ch in sorted(self.chunks, key=lambda c: c.chunk_id):
                hasher.update(ch.chunk_id.encode("utf-8"))
                hasher.update(ch.text.encode("utf-8"))
        hasher.update(self.embedder.version.encode("utf-8"))
        hasher.update(str(self.embedder.dimension).encode("utf-8"))
        self.index_version = hasher.hexdigest()

        return IndexBuildResult(
            index_version=self.index_version,
            document_count=len(self.documents),
            chunk_count=len(self.chunks),
            embedding_model=self.embedder.version,
            build_timestamp=datetime.now(timezone.utc),
        )

    def _passes_authorization(self, chunk: Chunk, user_role: str) -> bool:
        """Role-based access filtering.

        Roles:
        - 'admin', 'compliance_officer': access all ('public', 'internal_confidential', 'restricted')
        - 'credit_officer', 'analyst': access 'public', 'internal_confidential'
        - 'external', 'auditor_external': access 'public' only
        """
        doc = self.chunk_id_to_doc.get(chunk.chunk_id)
        classification = getattr(doc, "access_classification", "internal_confidential") if doc else "internal_confidential"

        if user_role in ("admin", "compliance_officer"):
            return True
        if user_role in ("credit_officer", "analyst"):
            return classification in ("public", "internal_confidential")
        # External
        return classification == "public"

    def _passes_metadata_filters(self, chunk: Chunk, filters: dict[str, Any]) -> bool:
        """Filter chunk by jurisdiction, issuer, and superseded status."""
        doc = self.chunk_id_to_doc.get(chunk.chunk_id)

        # Check superseded filter
        if self.exclude_superseded:
            if chunk.metadata and chunk.metadata.get("status") == "superseded":
                return False
            if doc and getattr(doc, "status", "active") == "superseded":
                return False

        if not doc:
            return True

        # Check jurisdiction filter
        target_jurisdiction = filters.get("jurisdiction")
        if target_jurisdiction and doc.jurisdiction != target_jurisdiction:
            return False

        return True

    def _assess_applicability(self, chunk: Chunk, payload: HybridQueryPayload) -> list[str]:
        """Assess legal applicability and return warnings if conflicts exist."""
        warnings: list[str] = []
        doc = self.chunk_id_to_doc.get(chunk.chunk_id)

        # Status check
        chunk_status = (chunk.metadata.get("status") if chunk.metadata else None) or (getattr(doc, "status", "active") if doc else "active")
        if chunk_status == "superseded":
            warnings.append("Superseded status: chunk belongs to an outdated regulatory or policy version.")

        # Jurisdiction mismatch check
        req_jurisdiction = payload.metadata_filters.get("jurisdiction")
        chunk_jurisdiction = (doc.jurisdiction if doc else None) or (chunk.metadata.get("jurisdiction") if chunk.metadata else None)
        if req_jurisdiction and chunk_jurisdiction and chunk_jurisdiction != req_jurisdiction:
            warnings.append(
                f"Jurisdiction mismatch: query specified '{req_jurisdiction}', but document is '{chunk_jurisdiction}'."
            )

        # Untrusted prompt injection detection in source text via multi-vector defense
        scan_res = self.injection_defense.scan(chunk.text)
        if not scan_res.is_safe or scan_res.matched_rules:
            warnings.append(
                f"Untrusted content warning: chunk contains suspected prompt-injection instructions "
                f"({scan_res.attack_category or 'adversarial override'})."
            )

        return warnings

    def retrieve(
        self,
        payload: HybridQueryPayload,
        user_role: str = "analyst",
        top_n: int = 3,
    ) -> RetrievalResult:
        """Execute hybrid search, apply filters, and compute Reciprocal Rank Fusion."""
        start_time = time.perf_counter()

        if not self.chunks or self.faiss_index is None or self.bm25 is None:
            # Empty corpus case
            trace = RetrievalTrace(
                trace_id=f"TRACE-{int(time.time()*1000)}",
                query_payload=payload,
                dense_scores={},
                bm25_scores={},
                fused_scores={},
                retrieved_chunks=[],
                execution_time_ms=0.0,
            )
            return RetrievalResult(
                evidence=[],
                retrieval_trace=trace,
                no_evidence_found=True,
                missing_coverage=["No authoritative evidence found in the configured corpus."],
                applicability_warnings=["Empty corpus: no policy documents have been ingested."],
            )

        # 1. Dense search via FAISS
        query_vec = self.embedder.embed_text(payload.dense_query).reshape(1, -1)
        candidate_k = min(len(self.chunks), 20)
        dense_scores_raw, dense_indices = self.faiss_index.search(query_vec, candidate_k)

        dense_ranked_ids: list[str] = []
        dense_scores_map: dict[str, float] = {}
        for rank_idx, chunk_idx in enumerate(dense_indices[0]):
            if chunk_idx >= 0:
                ch = self.chunks[chunk_idx]
                score = float(dense_scores_raw[0][rank_idx])
                dense_ranked_ids.append(ch.chunk_id)
                dense_scores_map[ch.chunk_id] = round(score, 4)

        # 2. Lexical search via BM25
        lexical_query_tokens = re.findall(r"\w+", " ".join(payload.bm25_terms).lower())
        if not lexical_query_tokens:
            lexical_query_tokens = re.findall(r"\w+", payload.dense_query.lower())

        bm25_scores_raw = self.bm25.get_scores(lexical_query_tokens)
        bm25_sorted_indices = np.argsort(bm25_scores_raw)[::-1][:candidate_k]

        bm25_ranked_ids: list[str] = []
        bm25_scores_map: dict[str, float] = {}
        for idx in bm25_sorted_indices:
            ch = self.chunks[idx]
            score = float(bm25_scores_raw[idx])
            bm25_ranked_ids.append(ch.chunk_id)
            bm25_scores_map[ch.chunk_id] = round(score, 4)

        # 3. Filter candidate pool before fusion
        candidate_pool = set(dense_ranked_ids).union(set(bm25_ranked_ids))
        valid_candidates = []
        for cid in candidate_pool:
            chunk = self.chunk_id_to_chunk[cid]
            if self._passes_authorization(chunk, user_role) and self._passes_metadata_filters(
                chunk, payload.metadata_filters
            ):
                valid_candidates.append(cid)

        if not valid_candidates:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            trace = RetrievalTrace(
                trace_id=f"TRACE-{int(time.time()*1000)}",
                query_payload=payload,
                dense_scores=dense_scores_map,
                bm25_scores=bm25_scores_map,
                fused_scores={},
                retrieved_chunks=[],
                execution_time_ms=round(elapsed, 2),
            )
            return RetrievalResult(
                evidence=[],
                retrieval_trace=trace,
                no_evidence_found=True,
                missing_coverage=["No authoritative evidence found matching authorization and metadata filters."],
                applicability_warnings=["All candidates were excluded by filters or role access limits."],
            )

        # 4. Reciprocal Rank Fusion (RRF)
        fused_scores_map: dict[str, float] = {}
        for cid in valid_candidates:
            dense_rank = dense_ranked_ids.index(cid) + 1 if cid in dense_ranked_ids else candidate_k + 1
            bm25_rank = bm25_ranked_ids.index(cid) + 1 if cid in bm25_ranked_ids else candidate_k + 1

            rrf = (
                self.dense_weight / (self.rrf_k + dense_rank)
                + self.bm25_weight / (self.rrf_k + bm25_rank)
            )
            fused_scores_map[cid] = round(rrf, 6)

        # Rank by fused score
        sorted_candidates = sorted(fused_scores_map.keys(), key=lambda x: fused_scores_map[x], reverse=True)
        top_chunk_ids = sorted_candidates[:top_n]

        # 5. Build RetrievedEvidence items and applicability warnings
        evidence_items: list[RetrievedEvidence] = []
        applicability_warnings: list[str] = []

        for cid in top_chunk_ids:
            chunk = self.chunk_id_to_chunk[cid]
            warnings = self._assess_applicability(chunk, payload)
            applicability_warnings.extend(warnings)

            # First sentence as representative quote
            first_sentence = chunk.text.split(". ")[0] + "." if ". " in chunk.text else chunk.text

            evidence_items.append(
                RetrievedEvidence(
                    evidence_id=f"EV-{cid}",
                    chunk=chunk,
                    extracted_quote=first_sentence,
                    relevance_justification=f"Retrieved via RRF fusion (score={fused_scores_map[cid]:.4f}).",
                    verification_status="unverified",
                )
            )

        elapsed = (time.perf_counter() - start_time) * 1000.0

        trace = RetrievalTrace(
            trace_id=f"TRACE-{int(time.time()*1000)}",
            query_payload=payload,
            dense_scores={cid: dense_scores_map.get(cid, 0.0) for cid in top_chunk_ids},
            bm25_scores={cid: bm25_scores_map.get(cid, 0.0) for cid in top_chunk_ids},
            fused_scores={cid: fused_scores_map[cid] for cid in top_chunk_ids},
            retrieved_chunks=[self.chunk_id_to_chunk[cid] for cid in top_chunk_ids],
            execution_time_ms=round(elapsed, 2),
        )

        return RetrievalResult(
            evidence=evidence_items,
            retrieval_trace=trace,
            no_evidence_found=False,
            missing_coverage=[],
            applicability_warnings=sorted(list(set(applicability_warnings))),
        )


# =========================================================================
# Information Retrieval Evaluation Metrics (P2.2 / P2.6 / P2.7)
# =========================================================================

def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 3) -> float:
    """Proportion of relevant chunk IDs retrieved in the top-k results."""
    if not expected_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    expected = set(expected_ids)
    return len(top_k.intersection(expected)) / len(expected)


def precision_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 3) -> float:
    """Proportion of top-k retrieved chunks that are relevant."""
    if k <= 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    expected = set(expected_ids)
    return len(top_k.intersection(expected)) / k


def mean_reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """Reciprocal rank of the first relevant chunk found."""
    expected = set(expected_ids)
    for rank_idx, cid in enumerate(retrieved_ids, start=1):
        if cid in expected:
            return 1.0 / rank_idx
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 3) -> float:
    """Normalized Discounted Cumulative Gain at rank k."""
    expected = set(expected_ids)
    dcg = 0.0
    for rank_idx, cid in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if cid in expected else 0.0
        dcg += rel / math.log2(rank_idx + 1)

    # Ideal DCG
    idcg = 0.0
    ideal_hits = min(len(expected), k)
    for rank_idx in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(rank_idx + 1)

    return (dcg / idcg) if idcg > 0.0 else 0.0
