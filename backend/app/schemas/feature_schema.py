"""Feature schema contract defining types, bounds, nullability, derivations, and mutability."""

import hashlib
import json
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


MutabilityClass = Literal["immutable", "actionable", "derived", "sensitive"]
DataType = Literal["float", "int", "str", "bool"]
NullHandlingRule = Literal["error", "zero", "mean", "median", "mode", "constant"]


class NumericRange(BaseModel):
    """Allowed numeric range bounds."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    min_value: float | None = None
    max_value: float | None = None


class FeatureDefinition(BaseModel):
    """Comprehensive specification for an individual model feature."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Unique programmatic feature identifier")
    label: str = Field(description="Human-readable business display label")
    dtype: DataType = Field(description="Data type of the feature")
    unit: str = Field(description="Measurement unit: e.g. 'INR', 'ratio', 'days', 'years', 'count'")
    allowed_range: NumericRange | None = Field(default=None, description="Permitted min/max numerical range")
    allowed_categories: list[str] | None = Field(default=None, description="Permitted categorical values")
    nullability: bool = Field(default=False, description="Whether null/NaN values are permitted")
    null_handling_rule: NullHandlingRule = Field(default="error", description="Policy for handling missing values")
    position_index: int = Field(ge=0, description="Strict position index in the model input feature vector")
    derived: bool = Field(default=False, description="Whether this feature is computed from primitive inputs")
    formula: str | None = Field(default=None, description="Documented mathematical derivation formula")
    mutability: MutabilityClass = Field(
        description="Feature mutability: 'immutable', 'actionable', 'derived', or 'sensitive'"
    )
    description: str = Field(default="", description="Business and domain definition")

    @field_validator("formula")
    @classmethod
    def validate_formula_for_derived(cls, v: str | None, info) -> str | None:
        """Ensure derived features always specify their derivation formula."""
        is_derived = info.data.get("derived", False)
        if is_derived and not v:
            raise ValueError("Derived feature must provide a documented formula.")
        return v


class FeatureSchema(BaseModel):
    """Container for the complete validated feature schema."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default="1.0.0", description="Semantic version of the feature schema")
    features: list[FeatureDefinition] = Field(description="List of all defined feature specifications")
    feature_order: list[str] = Field(description="Canonical sequence of feature names for model inference")
    preprocessing_notes: str = Field(
        default="Standard numeric alignment without target leakage or silent unit coercion",
        description="Documentation of preprocessing rules",
    )

    def get_feature(self, name: str) -> FeatureDefinition | None:
        """Retrieve a feature definition by name."""
        for feat in self.features:
            if feat.name == name:
                return feat
        return None

    def compute_feature_order_hash(self) -> str:
        """Compute deterministic SHA-256 hash of the exact canonical feature order."""
        order_bytes = json.dumps(self.feature_order, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(order_bytes).hexdigest()

    def get_primitive_features(self) -> list[FeatureDefinition]:
        """Return all non-derived features."""
        return [f for f in self.features if not f.derived]

    def get_derived_features(self) -> list[FeatureDefinition]:
        """Return all derived features."""
        return [f for f in self.features if f.derived]

    def get_actionable_features(self) -> list[FeatureDefinition]:
        """Return actionable levers eligible for counterfactual search."""
        return [f for f in self.features if f.mutability == "actionable"]

    def get_immutable_features(self) -> list[FeatureDefinition]:
        """Return frozen / immutable features that must never change."""
        return [f for f in self.features if f.mutability in ("immutable", "sensitive")]
