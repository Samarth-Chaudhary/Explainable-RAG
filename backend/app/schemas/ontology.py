"""Domain ontology schema mapping quantitative features to policy retrieval concepts."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


FeatureType = Literal["primitive", "derived"]
OntologyMutability = Literal["immutable", "actionable", "derived_recompute", "sensitive"]


class OntologyRecord(BaseModel):
    """Semantic mapping definition for an individual credit feature."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_name: str = Field(description="Programmatic feature identifier")
    label: str = Field(description="Business display label")
    plain_language_definition: str = Field(description="Non-technical plain language explanation")
    unit: str = Field(description="Measurement unit")
    feature_type: FeatureType = Field(description="'primitive' or 'derived'")
    formula: str | None = Field(default=None, description="Derivation formula if derived")
    mutability: OntologyMutability = Field(description="Mutability classification")
    immutable_inputs: list[str] = Field(default_factory=list, description="Underlying unalterable inputs")
    actionable_inputs: list[str] = Field(default_factory=list, description="Underlying actionable inputs")
    legal_topics: list[str] = Field(default_factory=list, description="Associated regulatory topics")
    metadata_tags: list[str] = Field(default_factory=list, description="Indexing and policy tags")
    lexical_terms: list[str] = Field(default_factory=list, description="Search terms for BM25 retrieval")
    semantic_template: str = Field(description="Query template for dense embedding generation")
    source_families: list[str] = Field(default_factory=list, description="Authoritative document families")
    ontology_version: str = Field(default="1.0", description="Version of this ontology record")


class OntologyFile(BaseModel):
    """Container for the versioned credit risk domain ontology."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    ontology_version: str = Field(default="1.0.0", description="Semantic version of the ontology")
    domain: str = Field(default="retail_and_commercial_credit_risk", description="Target domain")
    records: list[OntologyRecord] = Field(description="Collection of ontology records")

    def get_record(self, feature_name: str) -> OntologyRecord | None:
        """Find ontology record by feature name."""
        for rec in self.records:
            if rec.feature_name == feature_name:
                return rec
        return None
