"""Pillar C: Ontology Service and SHAP-to-Query Translation Engine.

Bridges quantitative ML attributions and counterfactuals to qualitative policy search intent.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from backend.app.schemas.counterfactual import CounterfactualResult
from backend.app.schemas.explanation import ExplanationResult, FeatureContributionSummary
from backend.app.schemas.feature_schema import FeatureSchema
from backend.app.schemas.handoff import HybridQueryPayload, TranslationResult
from backend.app.schemas.ontology import OntologyFile, OntologyRecord


class InvalidOntologyError(Exception):
    """Raised when an ontology file fails structural or domain validation."""
    pass


class OntologyTranslatorProtocol(Protocol):
    """Protocol for SHAP-to-query translation service."""

    def translate(
        self,
        explanation: ExplanationResult,
        counterfactuals: CounterfactualResult | None = None,
        top_driver_threshold: float = 0.05,
    ) -> TranslationResult:
        ...


class OntologyService:
    """Manages loading, integrity validation, and querying of versioned domain ontologies."""

    def __init__(self, ontology_path: str, feature_schema: FeatureSchema | None = None):
        self.ontology_path = Path(ontology_path)
        self.feature_schema = feature_schema
        self.ontology_file: OntologyFile | None = None
        self.content_hash: str = ""
        self.records_by_name: dict[str, OntologyRecord] = {}
        self._load_and_validate()

    @property
    def ontology(self) -> OntologyFile | None:
        """Expose ontology_file as ontology attribute."""
        return self.ontology_file

    def _load_and_validate(self) -> None:
        """Load and validate domain ontology file."""
        if not self.ontology_path.is_file():
            raise FileNotFoundError(f"Ontology file not found at: {self.ontology_path}")

        raw_bytes = self.ontology_path.read_bytes()
        self.content_hash = hashlib.sha256(raw_bytes).hexdigest()

        try:
            raw_data = json.loads(raw_bytes.decode("utf-8"))
            self.ontology_file = OntologyFile.model_validate(raw_data)
        except Exception as e:
            raise InvalidOntologyError(f"Ontology schema validation failed: {e}") from e

        # Validate unique feature names
        seen_names = set()
        for record in self.ontology_file.records:
            if record.feature_name in seen_names:
                raise InvalidOntologyError(
                    f"Duplicate feature_name in ontology: '{record.feature_name}'"
                )
            seen_names.add(record.feature_name)
            self.records_by_name[record.feature_name] = record

        # Validate against feature schema if supplied
        if self.feature_schema:
            known_primitives = {
                f.name for f in self.feature_schema.get_primitive_features()
            }
            for record in self.ontology_file.records:
                # Check unit consistency if feature is in schema
                schema_feat = self.feature_schema.get_feature(record.feature_name)
                if schema_feat and schema_feat.unit != record.unit:
                    raise InvalidOntologyError(
                        f"Unit mismatch for feature '{record.feature_name}': "
                        f"Ontology specifies '{record.unit}', schema specifies '{schema_feat.unit}'"
                    )

                # Check formula references
                if record.feature_type == "derived" and record.formula:
                    # Check if referenced inputs resolve to known primitives
                    for inp in record.immutable_inputs + record.actionable_inputs:
                        if inp not in known_primitives and inp != record.feature_name:
                            raise InvalidOntologyError(
                                f"Ontology formula input '{inp}' for '{record.feature_name}' "
                                f"does not resolve to any known primitive feature in FeatureSchema."
                            )

    @property
    def version(self) -> str:
        """Return semantic version of the active ontology."""
        return self.ontology_file.ontology_version if self.ontology_file else "unknown"

    @property
    def ontology_version(self) -> str:
        """Return semantic version of the active ontology."""
        return self.version

    def get_record(self, feature_name: str) -> OntologyRecord | None:
        """Retrieve ontology record for a specific feature."""
        return self.records_by_name.get(feature_name)


class OntologyTranslatorService:
    """Translates SHAP explanation drivers and validated counterfactuals into policy search queries."""

    def __init__(self, ontology_service: OntologyService):
        self.ontology_service = ontology_service

    def translate(
        self,
        explanation: ExplanationResult,
        counterfactuals: CounterfactualResult | None = None,
        top_k_features: int = 3,
        contribution_cutoff: float = 0.05,
        jurisdiction: str = "IN",
    ) -> TranslationResult:
        """Execute translation algorithm (Step 6).

        1. Read top risk-increasing features and validated DiCE deltas.
        2. Resolve selected features against versioned ontology.
        3. Build structured hybrid query payload.
        4. Include both risk-driver evidence and proposed adjustments.
        5. Log mapping details and handle unmapped features safely.
        """
        # Step 1: Feature Selection
        # Filter risk-increasing features meeting cutoff
        selected_drivers: list[FeatureContributionSummary] = [
            f for f in explanation.top_risk_increasing
            if abs(f.shap_value) >= contribution_cutoff
        ][:top_k_features]

        # If top_risk_increasing is empty, take highest magnitude features
        if not selected_drivers and explanation.features:
            selected_drivers = explanation.features[:top_k_features]

        features_selected = [f.name for f in selected_drivers]

        # Step 2: Resolve against ontology
        ontology_matches: list[str] = []
        unmapped_features: list[str] = []
        resolved_records: list[OntologyRecord] = []

        for f_name in features_selected:
            rec = self.ontology_service.get_record(f_name)
            if rec:
                ontology_matches.append(f_name)
                resolved_records.append(rec)
            else:
                unmapped_features.append(f_name)

        # Step 3 & 4: Extract counterfactual deltas and check validation status
        validated_changes: dict[str, dict[str, Any]] = {}
        counterfactual_status = "unvalidated_or_absent"

        if counterfactuals and not counterfactuals.no_feasible_found and counterfactuals.candidates:
            # Pick primary feasible candidate
            top_candidate = counterfactuals.candidates[0]
            if top_candidate.target_met and top_candidate.feasibility_status == "feasible":
                counterfactual_status = "validated_by_model_and_constraints"
                for ch in top_candidate.changes:
                    validated_changes[ch.feature] = {
                        "current": ch.current,
                        "proposed": ch.proposed,
                        "delta": ch.delta,
                    }
                for rec in top_candidate.derived_recalculations:
                    validated_changes[rec.feature] = {
                        "current": rec.previous_value,
                        "proposed": rec.new_value,
                        "delta": rec.new_value - rec.previous_value,
                    }

        # Build query components
        dense_query_parts: list[str] = []
        bm25_terms: list[str] = []
        metadata_filters: dict[str, Any] = {
            "jurisdiction": jurisdiction,
            "filter_tags": [],
        }
        source_families: set[str] = set()

        for rec in resolved_records:
            feat_name = rec.feature_name
            # Find driver value
            driver_val = next((f.value for f in selected_drivers if f.name == feat_name), None)

            # Check if this feature has a validated counterfactual target
            target_info = validated_changes.get(feat_name)
            if target_info:
                target_val = target_info["proposed"]
                query_text = (
                    f"Applicant {rec.label} is {driver_val:.2f} ({rec.unit}); "
                    f"model-validated feasible target is {target_val:.2f} ({rec.unit}). "
                    f"Identify applicable underwriting thresholds, prudential caps, and permitted review remedies."
                )
            else:
                query_text = (
                    f"Applicant {rec.label} is {driver_val if driver_val is not None else 'elevated'} ({rec.unit}). "
                    f"Identify applicable credit policy limits and remediation guidelines."
                )

            dense_query_parts.append(query_text)
            bm25_terms.extend(rec.lexical_terms)
            metadata_filters["filter_tags"].extend(rec.metadata_tags)
            source_families.update(rec.source_families)

        # Handle unmapped features safely without inventing regulatory meaning
        if unmapped_features:
            dense_query_parts.append(
                f"Review credit guidelines for unmapped attributes: {', '.join(unmapped_features)}."
            )
            bm25_terms.extend(unmapped_features)

        # Fallback if no records matched
        if not dense_query_parts:
            normalized_query = (
                "Review applicable institutional credit policies, underwriting limits, "
                "and permissible adverse action restructuring remedies."
            )
            bm25_terms = ["credit policy", "underwriting", "restructuring"]
        else:
            normalized_query = " ".join(dense_query_parts)

        # Deduplicate terms and tags
        dedup_bm25 = sorted(list(set(bm25_terms)))
        metadata_filters["filter_tags"] = sorted(list(set(metadata_filters["filter_tags"])))

        hybrid_payload = HybridQueryPayload(
            dense_query=normalized_query,
            bm25_terms=dedup_bm25,
            metadata_filters=metadata_filters,
            top_k=5,
            source_families=sorted(list(source_families)),
        )

        return TranslationResult(
            query_payload=hybrid_payload,
            features_selected=features_selected,
            ontology_matches=ontology_matches,
            unmapped_features=unmapped_features,
            normalized_query=normalized_query,
            lexical_terms=dedup_bm25,
            metadata_filters=metadata_filters,
            ontology_version=self.ontology_service.version,
            ontology_content_hash=self.ontology_service.content_hash,
        )


def build_naive_baseline_query(explanation: ExplanationResult, top_k: int = 3) -> HybridQueryPayload:
    """Separate naive baseline query builder for comparative evaluation (Part 3).

    Constructs an ungrounded keyword search by simply concatenating raw feature names with 'loan rejection'.
    """
    top_features = [f.name for f in explanation.features[:top_k]]
    raw_query = " ".join(top_features) + " loan rejection"
    return HybridQueryPayload(
        dense_query=raw_query,
        bm25_terms=top_features + ["rejection", "loan"],
        metadata_filters={},
        top_k=5,
        source_families=[],
    )
