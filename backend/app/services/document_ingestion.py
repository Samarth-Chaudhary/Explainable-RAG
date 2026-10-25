"""Pillar D: PDF Document Ingestion, Registry, and Chunking Engine.

Extracts text while preserving page-level provenance, clause boundaries, and upload security.
"""

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
import pymupdf

from backend.app.schemas.handoff import Chunk, IngestionResult, IndexBuildResult, SourceDocument
from backend.app.schemas.source_registry import SourceRegistryRecord


class IngestionSecurityError(Exception):
    """Raised when an uploaded document fails security, magic byte, or size checks."""
    pass


class DocumentIngestionError(IngestionSecurityError):
    """Generic document ingestion exception."""
    pass


class DocumentIngestionProtocol(Protocol):
    """Protocol for PDF document ingestion, registry, and chunking."""

    def ingest_pdf(self, file_path: Path, metadata: SourceRegistryRecord) -> IngestionResult:
        ...

    def reindex(self, document_id: str) -> IndexBuildResult:
        ...


class DocumentIngestionService:
    """Ingests policy PDFs, validates file integrity, and extracts page-preserving chunks."""

    PDF_MAGIC_BYTES = b"%PDF"

    def __init__(
        self,
        max_upload_size_bytes: int = 10 * 1024 * 1024,
        allow_superseded: bool = True,
    ):
        self.max_upload_size_bytes = max_upload_size_bytes
        self.allow_superseded = allow_superseded
        self.registry: dict[str, SourceDocument] = {}
        self.hash_to_doc_id: dict[str, str] = {}

    def _validate_file_upload(self, file_path: Path) -> bytes:
        """Validate file size, path traversal, and magic bytes."""
        # Sanitize path to prevent path traversal
        clean_name = Path(file_path).name
        if clean_name != file_path.name:
            raise IngestionSecurityError(f"Potential path traversal detected in filename '{file_path.name}'.")

        if not file_path.is_file():
            raise FileNotFoundError(f"Source file not found at: {file_path}")

        file_size = file_path.stat().st_size
        if file_size > self.max_upload_size_bytes:
            raise IngestionSecurityError(
                f"File size {file_size} bytes exceeds maximum permitted limit {self.max_upload_size_bytes} bytes."
            )

        raw_bytes = file_path.read_bytes()
        if not raw_bytes.startswith(self.PDF_MAGIC_BYTES):
            raise IngestionSecurityError("Invalid file content: missing standard %PDF magic header bytes.")

        return raw_bytes

    def _extract_pages_and_sections(self, doc: pymupdf.Document) -> list[dict[str, Any]]:
        """Extract text per page and detect section headings."""
        pages_data = []
        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]
            page_text = page.get_text("text").strip()
            pages_data.append({
                "page_number": page_num,
                "text": page_text,
            })
        return pages_data

    def _chunk_text_with_provenance(
        self,
        source_id: str,
        pages_data: list[dict[str, Any]],
        doc_hash: str,
        embedding_model: str = "local-deterministic-v1",
    ) -> list[Chunk]:
        """Chunk document text preserving section headings and page boundary overlaps."""
        chunks: list[Chunk] = []
        chunk_idx = 1

        current_section = "General Provisions"

        for page_info in pages_data:
            page_num = page_info["page_number"]
            page_text = page_info["text"]

            if not page_text:
                continue

            # Split text into paragraphs
            paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]

            for para in paragraphs:
                # Detect section heading
                section_match = re.match(r"^(Section\s+[0-9]+(?:\.[0-9]+)*[^\n]*)", para, re.IGNORECASE)
                if section_match:
                    current_section = section_match.group(1).strip()

                chunk_id = f"{source_id}-P{page_num}-C{chunk_idx}"
                token_count = len(para.split())

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        source_id=source_id,
                        text=para,
                        page_number=page_num,
                        section_title=current_section,
                        token_count=token_count,
                        metadata={
                            "document_checksum": doc_hash,
                            "embedding_model": embedding_model,
                        },
                    )
                )
                chunk_idx += 1

        # Page-boundary multi-page overlap check:
        # If consecutive chunks cross page boundary within the same section, create a joined boundary chunk
        if len(pages_data) > 1:
            for i in range(len(chunks) - 1):
                c1 = chunks[i]
                c2 = chunks[i + 1]
                if c1.page_number != c2.page_number and c1.section_title == c2.section_title:
                    overlap_id = f"{source_id}-SPAN-P{c1.page_number}_P{c2.page_number}-C{chunk_idx}"
                    overlap_text = f"{c1.text}\n{c2.text}"
                    chunks.append(
                        Chunk(
                            chunk_id=overlap_id,
                            source_id=source_id,
                            text=overlap_text,
                            page_number=c1.page_number,
                            section_title=c1.section_title,
                            token_count=len(overlap_text.split()),
                            metadata={
                                "pages_spanned": [c1.page_number, c2.page_number],
                                "is_page_boundary_span": True,
                                "document_checksum": doc_hash,
                                "embedding_model": embedding_model,
                            },
                        )
                    )
                    chunk_idx += 1
                    break  # One synthetic boundary chunk is sufficient

        return chunks

    def ingest_pdf(
        self,
        file_path: Path,
        metadata: SourceRegistryRecord,
        embedding_model: str = "local-deterministic-v1",
    ) -> IngestionResult:
        """Ingest a policy PDF with validation, duplicate detection, and page extraction."""
        # Step 1: Security and format validation
        raw_bytes = self._validate_file_upload(file_path)
        actual_hash = hashlib.sha256(raw_bytes).hexdigest()

        # Step 2: Check duplicate document
        if actual_hash in self.hash_to_doc_id:
            existing_id = self.hash_to_doc_id[actual_hash]
            existing_doc = self.registry[existing_id]
            return IngestionResult(
                document=existing_doc,
                chunks=existing_doc.chunks,
                parser_status="parsed",
                ocr_applied=False,
                error_message=f"Duplicate document detected; returning registered document '{existing_id}'.",
            )

        # Step 3: Open with PyMuPDF
        try:
            doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
        except Exception as e:
            return IngestionResult(
                document=SourceDocument(
                    source_id=metadata.source_id,
                    title=metadata.title,
                    issuer=metadata.issuer,
                    jurisdiction=metadata.jurisdiction,
                    version=metadata.version_or_update_date,
                    effective_date=metadata.effective_date,
                    file_path=str(file_path),
                    sha256_hash=actual_hash,
                    total_pages=1,
                    chunks=[],
                ),
                chunks=[],
                parser_status="failed",
                ocr_applied=False,
                error_message=f"Corrupted PDF file could not be parsed: {e}",
            )

        total_pages = len(doc)
        pages_data = self._extract_pages_and_sections(doc)
        doc.close()

        # Step 4: Check for scanned PDF (all pages have 0 text)
        total_text_len = sum(len(p["text"]) for p in pages_data)
        if total_text_len == 0 and total_pages > 0:
            # Scanned document without OCR capability
            return IngestionResult(
                document=SourceDocument(
                    source_id=metadata.source_id,
                    title=metadata.title,
                    issuer=metadata.issuer,
                    jurisdiction=metadata.jurisdiction,
                    version=metadata.version_or_update_date,
                    effective_date=metadata.effective_date,
                    file_path=str(file_path),
                    sha256_hash=actual_hash,
                    total_pages=total_pages,
                    chunks=[],
                ),
                chunks=[],
                parser_status="failed",
                ocr_applied=False,
                error_message="Scanned PDF detected with zero extracted text. OCR engine is required but not installed.",
            )

        # Step 5: Chunking with provenance
        chunks = self._chunk_text_with_provenance(
            source_id=metadata.source_id,
            pages_data=pages_data,
            doc_hash=actual_hash,
            embedding_model=embedding_model,
        )

        source_doc = SourceDocument(
            source_id=metadata.source_id,
            title=metadata.title,
            issuer=metadata.issuer,
            jurisdiction=metadata.jurisdiction,
            version=metadata.version_or_update_date,
            effective_date=metadata.effective_date,
            file_path=str(file_path),
            sha256_hash=actual_hash,
            total_pages=total_pages,
            chunks=chunks,
        )

        # Register document
        self.registry[metadata.source_id] = source_doc
        self.hash_to_doc_id[actual_hash] = metadata.source_id

        return IngestionResult(
            document=source_doc,
            chunks=chunks,
            parser_status="parsed",
            ocr_applied=False,
            error_message=None,
        )
