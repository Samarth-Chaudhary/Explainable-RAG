"""Unit tests for Pillar D: Document Ingestion, Registry, and Chunking."""

import pytest
from pathlib import Path
from typing import cast

from backend.app.schemas.source_registry import SourceRegistryRecord
from backend.app.services.document_ingestion import (
    DocumentIngestionService,
    IngestionSecurityError,
)


from datetime import datetime, timezone

@pytest.fixture
def foir_metadata() -> SourceRegistryRecord:
    return SourceRegistryRecord(
        source_id="SYN-POL-FOIR-001",
        title="SYNTHETIC DEMO POLICY - NOT LAW: Repayment Capacity and FOIR Thresholds",
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


@pytest.fixture
def superseded_metadata() -> SourceRegistryRecord:
    return SourceRegistryRecord(
        source_id="SYN-POL-SUPERSEDED-001",
        title="SYNTHETIC DEMO POLICY - NOT LAW: Legacy Restructuring Circular (SUPERSEDED)",
        issuer="Synthetic Historical Policy Registry",
        jurisdiction="IN",
        source_family="legacy_rules",
        publication_date="2020-01-01",
        effective_date="2020-01-01",
        version_or_update_date="v0.9-obsolete",
        status="superseded",
        official_url_or_owner="credit-policy-risk@synthetic-bank.demo",
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ingestion_timestamp=datetime.now(timezone.utc),
        access_classification="internal_confidential",
        parser_status="pending",
    )


def test_ingest_valid_foir_policy(foir_metadata: SourceRegistryRecord):
    """Test ingestion of valid multi-page PDF with section extraction and page numbers."""
    service = DocumentIngestionService()
    pdf_path = Path("tests/fixtures/synthetic_policies/synthetic_foir_policy.pdf")
    assert pdf_path.exists(), "Test PDF file missing"

    result = service.ingest_pdf(pdf_path, foir_metadata)

    assert result.parser_status == "parsed"
    assert result.ocr_applied is False
    assert len(result.chunks) > 0
    assert result.document.total_pages >= 2

    # Check page numbers and section titles
    pages = {c.page_number for c in result.chunks}
    assert 1 in pages
    assert 2 in pages

    # Verify that clauses have meaningful text and token counts
    for chunk in result.chunks:
        assert len(chunk.text) > 0
        assert chunk.token_count > 0
        assert chunk.source_id == "SYN-POL-FOIR-001"
        assert "document_checksum" in chunk.metadata


def test_duplicate_document_detection(foir_metadata: SourceRegistryRecord):
    """Test that ingesting identical file twice triggers duplicate detection."""
    service = DocumentIngestionService()
    pdf_path = Path("tests/fixtures/synthetic_policies/synthetic_foir_policy.pdf")

    res1 = service.ingest_pdf(pdf_path, foir_metadata)
    assert res1.parser_status == "parsed"

    # Second ingestion with same file
    res2 = service.ingest_pdf(pdf_path, foir_metadata)
    assert res2.parser_status == "parsed"
    assert "Duplicate document detected" in (res2.error_message or "")
    assert res2.document.source_id == res1.document.source_id


def test_malformed_file_magic_bytes_rejected(tmp_path: Path, foir_metadata: SourceRegistryRecord):
    """Test that files without %PDF magic bytes are rejected."""
    fake_file = tmp_path / "fake_policy.pdf"
    fake_file.write_text("This is not a real PDF file header.", encoding="utf-8")

    service = DocumentIngestionService()
    with pytest.raises(IngestionSecurityError, match="missing standard %PDF magic header"):
        service.ingest_pdf(fake_file, foir_metadata)


def test_path_traversal_detection(tmp_path: Path, foir_metadata: SourceRegistryRecord):
    """Test that paths with traversal patterns are intercepted."""
    service = DocumentIngestionService()
    traversal_path = tmp_path / "normal.pdf"
    traversal_path.write_bytes(b"%PDF-1.4 header")

    # Pass an object where name doesn't match
    class MockPath:
        name = "../evil.pdf"

        def __fspath__(self) -> str:
            return str(traversal_path)

    mock_evil = cast(Path, MockPath())
    with pytest.raises(IngestionSecurityError, match="Potential path traversal"):
        service._validate_file_upload(mock_evil)


def test_oversized_file_rejected(tmp_path: Path, foir_metadata: SourceRegistryRecord):
    """Test that files exceeding configured size limit are rejected."""
    service = DocumentIngestionService(max_upload_size_bytes=500)
    pdf_path = Path("tests/fixtures/synthetic_policies/synthetic_foir_policy.pdf")

    with pytest.raises(IngestionSecurityError, match="exceeds maximum permitted limit"):
        service.ingest_pdf(pdf_path, foir_metadata)


def test_scanned_pdf_zero_text_returns_clear_failure(tmp_path: Path, foir_metadata: SourceRegistryRecord):
    """Test that a blank/scanned PDF with 0 extractable text fails safely without fabricating text."""
    import pymupdf

    empty_pdf = tmp_path / "blank_scanned.pdf"
    doc = pymupdf.open()
    doc.new_page(width=595, height=842)  # blank page without text
    doc.save(str(empty_pdf))
    doc.close()

    service = DocumentIngestionService()
    result = service.ingest_pdf(empty_pdf, foir_metadata)

    assert result.parser_status == "failed"
    assert result.chunks == []
    assert "OCR engine is required" in (result.error_message or "")


def test_superseded_document_tracked(superseded_metadata: SourceRegistryRecord):
    """Test that superseded documents are ingested with appropriate metadata."""
    service = DocumentIngestionService(allow_superseded=True)
    pdf_path = Path("tests/fixtures/synthetic_policies/synthetic_superseded_policy.pdf")

    result = service.ingest_pdf(pdf_path, superseded_metadata)
    assert result.parser_status == "parsed"
    assert result.document.source_id == "SYN-POL-SUPERSEDED-001"
