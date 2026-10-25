"""Test health check API endpoint and FastAPI app initialization."""

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify GET /health returns 200 with synthetic flags and threshold metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["mode"] == "demo"
    assert "synthetic_flags" in data
    assert data["synthetic_flags"]["is_synthetic_model"] is True
    assert "threshold" in data
    assert data["threshold"]["value"] == 0.35
