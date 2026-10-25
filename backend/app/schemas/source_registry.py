"""Source registry schema tracking authoritative policy documents and lineage."""

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


SourceStatus = Literal["active", "superseded"]
ParserStatus = Literal["pending", "parsed", "failed"]
AccessClassification = Literal["public", "internal_confidential", "restricted"]


class SourceRegistryRecord(BaseModel):
    """Lineage and provenance metadata for an ingested policy document."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(description="Unique source document identifier")
    title: str = Field(description="Official title of the document")
    issuer: str = Field(description="Issuing regulatory body or internal committee")
    jurisdiction: str = Field(description="Jurisdiction (e.g. 'IN', 'GLOBAL', 'US')")
    source_family: str = Field(description="Category family: e.g. 'statutory_circular', 'credit_policy'")
    publication_date: str = Field(description="Date document was published (YYYY-MM-DD)")
    effective_date: str = Field(description="Date document took effect (YYYY-MM-DD)")
    version_or_update_date: str = Field(description="Version number or amendment date")
    status: SourceStatus = Field(default="active", description="'active' or 'superseded'")
    official_url_or_owner: str = Field(description="Authoritative source URL or policy custodian")
    sha256_hash: str = Field(description="Cryptographic SHA-256 digest of original PDF/source file")
    ingestion_timestamp: datetime = Field(description="Timestamp of document ingestion into system")
    parser_status: ParserStatus = Field(default="pending", description="Status of text extraction")
    access_classification: AccessClassification = Field(
        default="internal_confidential",
        description="Data classification level",
    )


class SourceRegistry(BaseModel):
    """Collection of registered policy documents."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    registry_version: str = Field(default="1.0.0")
    documents: list[SourceRegistryRecord] = Field(default_factory=list)
