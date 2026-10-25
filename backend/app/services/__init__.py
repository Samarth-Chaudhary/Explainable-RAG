"""Services package for XAI-RAG."""

from backend.app.services.audit_ledger import AuditLedgerProtocol, AuditLedgerService
from backend.app.services.citation_verifier import CitationVerifierProtocol, CitationVerifierService
from backend.app.services.counterfactual import CounterfactualService
from backend.app.services.document_ingestion import DocumentIngestionProtocol, DocumentIngestionService
from backend.app.services.explanation import (
    CrossSpaceComparisonError,
    DegradedExplanationError,
    ExplanationService,
)
from backend.app.services.export_pdf import PDFExportProtocol, PDFExportService
from backend.app.services.ontology_translator import OntologyTranslatorProtocol, OntologyTranslatorService
from backend.app.services.prediction import (
    ModelIntegrityError,
    ModelLoader,
    PredictionService,
    SchemaMismatchError,
)
from backend.app.services.retrieval import HybridRetrievalProtocol, HybridRetrievalService
from backend.app.services.report_generator import ReportGeneratorProtocol, ReportGeneratorService

__all__ = [
    # Active Pillar A & B Services
    "ModelLoader",
    "PredictionService",
    "ModelIntegrityError",
    "SchemaMismatchError",
    "ExplanationService",
    "CrossSpaceComparisonError",
    "DegradedExplanationError",
    "CounterfactualService",
    # Part 2 / 3 Service Interfaces
    "OntologyTranslatorProtocol",
    "OntologyTranslatorService",
    "DocumentIngestionProtocol",
    "DocumentIngestionService",
    "HybridRetrievalProtocol",
    "HybridRetrievalService",
    "ReportGeneratorProtocol",
    "ReportGeneratorService",
    "CitationVerifierProtocol",
    "CitationVerifierService",
    "AuditLedgerProtocol",
    "AuditLedgerService",
    "PDFExportProtocol",
    "PDFExportService",
]
