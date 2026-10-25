"""Dedicated citation verification service (Part 2 implementation).

Performs exact substring matching between cited regulatory passages and the
ingested source chunks under a documented Unicode + whitespace normalization policy.

Guarantees:
- Never uses loose semantic similarity or embedding distance to claim a quote passed.
- Checks cited page number, section title, and source document ID against chunk metadata.
- Handles quotes spanning adjacent chunk boundaries.
- Emits detailed per-quote pass/fail audit traces with failure reasons.
- Sanitizes unverified citations so they are never presented as verified legal quotes.
- Computes faithfulness score: passed / total (skipped checks count as failures).
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Protocol
import unicodedata

from backend.app.schemas.audit_report import (
    AuditReport,
    EvidenceStatus,
    ReportDossierSummary,
    ReportEvidenceItem,
    ReportVerifierSummary,
)
from backend.app.schemas.handoff import Chunk, DraftReport, VerifierResult

logger = logging.getLogger(__name__)


class CitationVerifierProtocol(Protocol):
    """Protocol for verbatim quote and citation provenance verification."""

    def verify_citations(
        self,
        evidence_items: list[ReportEvidenceItem],
        chunks: list[Chunk],
    ) -> VerifierResult:
        ...

    def verify(
        self,
        draft: DraftReport,
        chunks: list[Chunk],
    ) -> tuple[VerifierResult, AuditReport]:
        ...

NORMALIZED_MATCH_POLICY = (
    "Unicode NFKC normalization + ASCII/Unicode quote/dash unification "
    "+ whitespace collapsing (\\s+ -> ' ') + trimming. "
    "Exact substring matching only. Verification establishes textual provenance only; "
    "it does not establish legal applicability, currency, or statutory validity."
)


def normalize_text_for_verification(text: str) -> str:
    """Documented text normalization rule:

    1. Unicode NFKC normalization.
    2. Quote unification: '’' and '‘' -> "'", '“' and '”' -> '"'.
    3. Dash unification: '—' (em dash) and '–' (en dash) -> '-'.
    4. Whitespace collapsing: all sequences of whitespace (spaces, tabs, newlines) collapsed to a single space.
    5. Strip leading and trailing whitespace.
    """
    if not text:
        return ""
    # 1. Unicode NFKC
    normalized = unicodedata.normalize("NFKC", text)
    # 2. Quote unification
    normalized = normalized.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    # 3. Dash unification
    normalized = normalized.replace("—", "-").replace("–", "-")
    # 4. Whitespace collapse
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


class CitationVerificationError(Exception):
    """Raised when critical citation verification fails in fail-closed mode."""


class CitationVerifierService:
    """Dedicated quote matching and citation verification service."""

    def __init__(self, fail_closed: bool = False):
        self.fail_closed = fail_closed
        self.normalization_policy = NORMALIZED_MATCH_POLICY

    def verify_single_quote(
        self,
        item: ReportEvidenceItem,
        chunk_map: dict[str, Chunk],
        chunks_by_source: dict[str, list[Chunk]] | None = None,
    ) -> dict[str, Any]:
        """Verify an individual quote against source chunks and metadata."""
        quote_raw = item.quote
        normalized_quote = normalize_text_for_verification(quote_raw)
        quote_hash = hashlib.sha256(normalized_quote.encode("utf-8")).hexdigest()

        result: dict[str, Any] = {
            "chunk_id": item.chunk_id,
            "source_id": item.source_id,
            "page": item.page,
            "section": item.section,
            "quote_length": len(quote_raw),
            "normalized_quote_hash": quote_hash,
            "passed": False,
            "failure_reason": None,
        }

        # Check 1: Non-empty quote
        if not normalized_quote:
            result["failure_reason"] = "EMPTY_QUOTE"
            return result

        # Check 2: Chunk existence
        target_chunk = chunk_map.get(item.chunk_id)
        if target_chunk is None:
            result["failure_reason"] = f"CHUNK_NOT_FOUND: {item.chunk_id}"
            return result

        # Check 3: Source ID metadata alignment
        if item.source_id != target_chunk.source_id:
            result["failure_reason"] = (
                f"SOURCE_MISMATCH: cited '{item.source_id}', chunk has '{target_chunk.source_id}'"
            )
            return result

        # Check 4: Page number metadata alignment
        if item.page != target_chunk.page_number:
            # Check if chunk metadata specifies a page span
            page_span = target_chunk.metadata.get("page_span")
            if not (isinstance(page_span, list) and item.page in page_span):
                result["failure_reason"] = (
                    f"PAGE_MISMATCH: cited page {item.page}, chunk is on page {target_chunk.page_number}"
                )
                return result

        # Check 5: Section title alignment
        norm_item_sec = normalize_text_for_verification(item.section).lower()
        norm_chunk_sec = normalize_text_for_verification(target_chunk.section_title).lower()
        if norm_item_sec not in norm_chunk_sec and norm_chunk_sec not in norm_item_sec:
            result["failure_reason"] = (
                f"SECTION_MISMATCH: cited section '{item.section}', chunk section '{target_chunk.section_title}'"
            )
            return result

        # Check 6: Exact substring match in primary target chunk
        normalized_chunk_text = normalize_text_for_verification(target_chunk.text)
        if normalized_quote in normalized_chunk_text:
            result["passed"] = True
            return result

        # Check 7: Quote spanning boundary across adjacent chunks
        if chunks_by_source and target_chunk.source_id in chunks_by_source:
            source_chunks = chunks_by_source[target_chunk.source_id]
            # Find index of target chunk
            idx = next((i for i, c in enumerate(source_chunks) if c.chunk_id == target_chunk.chunk_id), -1)
            if idx != -1 and idx + 1 < len(source_chunks):
                next_chunk = source_chunks[idx + 1]
                combined_text = normalize_text_for_verification(f"{target_chunk.text} {next_chunk.text}")
                if normalized_quote in combined_text:
                    result["passed"] = True
                    result["boundary_span"] = f"Spans chunks {target_chunk.chunk_id} and {next_chunk.chunk_id}"
                    return result

        result["failure_reason"] = "QUOTE_NOT_FOUND_IN_CHUNK"
        return result

    def verify_citations(
        self,
        evidence_items: list[ReportEvidenceItem],
        chunks: list[Chunk],
    ) -> VerifierResult:
        """Verify a list of evidence items against candidate chunks."""
        chunk_map = {c.chunk_id: c for c in chunks}
        chunks_by_source: dict[str, list[Chunk]] = {}
        for c in chunks:
            chunks_by_source.setdefault(c.source_id, []).append(c)

        total = len(evidence_items)
        passed = 0
        failed = 0
        details: list[dict[str, Any]] = []

        for item in evidence_items:
            res = self.verify_single_quote(item, chunk_map, chunks_by_source)
            details.append(res)
            if res["passed"]:
                passed += 1
            else:
                failed += 1

        faithfulness = (passed / total) if total > 0 else 1.0

        return VerifierResult(
            total_citations=total,
            passed_citations=passed,
            failed_citations=failed,
            faithfulness_score=round(faithfulness, 4),
            citation_details=details,
        )

    def verify(
        self,
        draft: DraftReport,
        chunks: list[Chunk],
    ) -> tuple[VerifierResult, AuditReport]:
        """Verify draft report citations and produce final sanitized AuditReport.

        - Exact match passes -> marked 'verified'.
        - Match fails -> marked 'failed' (or 'unverified'), quote is flagged,
          overall report marked 'partial', never presented as verified law.
        """
        verifier_result = self.verify_citations(draft.evidence, chunks)

        # Sanitize evidence items and update verification status
        sanitized_evidence: list[ReportEvidenceItem] = []
        any_failed = False

        for idx, item in enumerate(draft.evidence):
            detail = verifier_result.citation_details[idx]
            if detail["passed"]:
                new_status: EvidenceStatus = "verified"
                sanitized_quote = item.quote
            else:
                any_failed = True
                new_status = "failed"
                # Flag unverified quote so it cannot be mistaken for verified law
                sanitized_quote = f"[UNVERIFIED - {detail['failure_reason']}] {item.quote}"

            sanitized_evidence.append(
                ReportEvidenceItem(
                    source_id=item.source_id,
                    title=item.title,
                    issuer=item.issuer,
                    jurisdiction=item.jurisdiction,
                    source_type=item.source_type,
                    version=item.version,
                    effective_date=item.effective_date,
                    page=item.page,
                    section=item.section,
                    quote=sanitized_quote,
                    chunk_id=item.chunk_id,
                    url=item.url,
                    verification_status=new_status,
                )
            )

        # Update final audit status
        final_status = draft.status
        if any_failed:
            if self.fail_closed:
                raise CitationVerificationError(
                    f"Citation verification failed closed: {verifier_result.failed_citations} "
                    f"of {verifier_result.total_citations} quotes failed verification."
                )
            final_status = "partial"

        # Update dossier if quotes failed
        dossier = draft.dossier
        if any_failed:
            gap_note = f"Citation verification failed for {verifier_result.failed_citations} quotes"
            dossier = ReportDossierSummary(
                adverse_action_explanation=dossier.adverse_action_explanation,
                candidate_review_paths=dossier.candidate_review_paths,
                regulatory_justification=dossier.regulatory_justification,
                evidence_gaps=list(dict.fromkeys(dossier.evidence_gaps + [gap_note])),
                reviewer_notes=(
                    dossier.reviewer_notes + " WARNING: Some cited policy passages could not be verified verbatim. "
                    "Consult original circulars directly."
                ),
            )

        verifier_summary = ReportVerifierSummary(
            quote_checks_total=verifier_result.total_citations,
            quote_checks_passed=verifier_result.passed_citations,
            quote_checks_failed=verifier_result.failed_citations,
            faithfulness_score=verifier_result.faithfulness_score,
            normalized_match_policy=self.normalization_policy,
        )

        final_report = AuditReport(
            audit_id=draft.audit_id,
            applicant_ref=draft.applicant_ref,
            generated_at=draft.generated_at,
            mode=draft.mode,
            status=final_status,
            model=draft.model,
            prediction=draft.prediction,
            explanation=draft.explanation,
            counterfactuals=draft.counterfactuals,
            search_trace=draft.search_trace,
            evidence=sanitized_evidence,
            dossier=dossier,
            verifier=verifier_summary,
            human_review_required=True,
        )

        return verifier_result, final_report
