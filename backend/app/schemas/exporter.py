"""Exports Pydantic model JSON Schemas into the repository for contract stability."""

import json
from pathlib import Path
from backend.app.schemas.audit_report import AuditReport
from backend.app.schemas.counterfactual import CounterfactualResult
from backend.app.schemas.explanation import ExplanationResult
from backend.app.schemas.feature_schema import FeatureSchema
from backend.app.schemas.handoff import (
    AuditEvent,
    Chunk,
    ErrorEnvelope,
    HybridQueryPayload,
    JobStatus,
    RetrievalTrace,
    RetrievedEvidence,
    SourceDocument,
    VerifierResult,
)
from backend.app.schemas.model_manifest import ModelManifest
from backend.app.schemas.ontology import OntologyFile, OntologyRecord
from backend.app.schemas.prediction import PredictionResult
from backend.app.schemas.source_registry import SourceRegistry, SourceRegistryRecord


EXPORT_MODELS = {
    "AuditReport.json": AuditReport,
    "PredictionResult.json": PredictionResult,
    "ExplanationResult.json": ExplanationResult,
    "CounterfactualResult.json": CounterfactualResult,
    "FeatureSchema.json": FeatureSchema,
    "ModelManifest.json": ModelManifest,
    "OntologyRecord.json": OntologyRecord,
    "OntologyFile.json": OntologyFile,
    "SourceRegistryRecord.json": SourceRegistryRecord,
    "SourceRegistry.json": SourceRegistry,
    "HybridQueryPayload.json": HybridQueryPayload,
    "Chunk.json": Chunk,
    "SourceDocument.json": SourceDocument,
    "RetrievalTrace.json": RetrievalTrace,
    "RetrievedEvidence.json": RetrievedEvidence,
    "VerifierResult.json": VerifierResult,
    "AuditEvent.json": AuditEvent,
    "ErrorEnvelope.json": ErrorEnvelope,
    "JobStatus.json": JobStatus,
}


def export_json_schemas(output_dir: str = "data/schemas") -> dict[str, str]:
    """Export JSON Schema for all shared models to the specified directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exported_files = {}

    for filename, model_cls in EXPORT_MODELS.items():
        schema_dict = model_cls.model_json_schema()
        file_dest = out_path / filename
        with open(file_dest, "w", encoding="utf-8") as f:
            json.dump(schema_dict, f, indent=2)
        exported_files[filename] = str(file_dest)

    return exported_files


if __name__ == "__main__":
    results = export_json_schemas()
    print(f"Exported {len(results)} JSON schemas to data/schemas/")
