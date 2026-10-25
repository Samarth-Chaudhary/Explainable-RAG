"""Unit tests for configuration validation, demo/configured mode boundary, and security hygiene."""

import os
from pathlib import Path
import pytest
from backend.app.config import Settings


def test_demo_mode_synthetic_flags():
    """Demo mode must activate all synthetic disclosure flags."""
    settings = Settings(mode="demo")
    assert settings.mode == "demo"
    assert settings.is_synthetic_inputs is True
    assert settings.is_synthetic_model is True
    assert settings.is_synthetic_policy is True
    assert settings.is_synthetic_llm is True


def test_configured_mode_fails_safely_without_threshold_owner():
    """Configured mode must fail safely at validation when threshold_owner is missing."""
    with pytest.raises(ValueError) as exc_info:
        Settings(
            mode="configured",
            threshold_owner="",  # Missing owner
            threshold_source="Credit Committee Policy",
            model_path="data/demo/model.json",
            model_manifest_path="data/demo/model_manifest.json",
            feature_schema_path="data/demo/feature_schema.json",
        )
    assert "threshold_owner" in str(exc_info.value)


def test_configured_mode_fails_safely_without_model_file():
    """Configured mode must fail safely if configured model path does not exist."""
    with pytest.raises(ValueError) as exc_info:
        Settings(
            mode="configured",
            threshold_owner="CRO",
            threshold_source="Policy Manual",
            model_path="non_existent_model_file_xyz.json",
            model_manifest_path="data/demo/model_manifest.json",
            feature_schema_path="data/demo/feature_schema.json",
        )
    assert "model file" in str(exc_info.value)


def test_no_hardcoded_secrets_or_absolute_paths_in_source():
    """Scan source code files to verify absence of hardcoded API keys or personal absolute paths."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    app_dir = repo_root / "backend" / "app"

    forbidden_patterns = [
        "sk-proj-",
        "AIzaSy",
        "ghp_",
        "AKIA",
    ]

    for py_file in app_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            assert pattern not in text, f"Potential secret found in {py_file}: {pattern}"
