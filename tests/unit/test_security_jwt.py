"""Unit and Integration Tests for JWT Authentication, Password Hashing, and Role Authorization.

Validates:
1. PBKDF2-HMAC-SHA256 password hashing and constant-time verification.
2. HS256 JWT creation, server-side role claims, and token expiry.
3. Cryptographic rejection of unsigned tokens (alg=none), wrong signatures, and expired tokens.
4. API endpoint protection:
   - Bare client-supplied X-User-Role header without JWT is rejected with 401.
   - Missing Authorization header is rejected with 401.
   - Expired token is rejected with 401.
   - Unsigned / alg=none token is rejected with 401.
   - Forged signature is rejected with 401.
   - Role escalation (analyst attempting admin endpoint) is rejected with 403.
   - Conflicting X-User-Role header spoofing is rejected with 403.
   - Valid credentials return 200 with signed token.
"""

from __future__ import annotations

import datetime
from datetime import timezone
import jwt
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.security import (
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
    authenticate_user,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

client = TestClient(app)


def test_password_hashing_and_constant_time_verification():
    """Verify PBKDF2-HMAC-SHA256 password hashing and comparison."""
    password = "SuperSecretPassword2026!"
    hashed = hash_password(password)

    assert hashed.startswith("pbkdf2_sha256$100000$")
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False
    assert verify_password("", hashed) is False


def test_built_in_user_authentication():
    """Verify built-in user authentication against hashed credentials."""
    admin = authenticate_user("admin", "AdminSecure2026!")
    assert admin is not None
    assert admin["role"] == "admin"

    officer = authenticate_user("credit_officer", "OfficerSecure2026!")
    assert officer is not None
    assert officer["role"] == "credit_officer"

    analyst = authenticate_user("analyst", "AnalystSecure2026!")
    assert analyst is not None
    assert analyst["role"] == "analyst"

    # Invalid password
    bad_auth = authenticate_user("admin", "WrongPassword!")
    assert bad_auth is None

    # Non-existent user
    unknown_user = authenticate_user("nonexistent_user", "AnyPassword!")
    assert unknown_user is None


def test_jwt_creation_and_payload_claims():
    """Verify HS256 JWT issuance includes server-side role claims and timestamps."""
    token = create_access_token(username="test_auditor", role="auditor")
    payload = decode_access_token(token)

    assert payload["sub"] == "test_auditor"
    assert payload["role"] == "auditor"
    assert payload["iss"] == "xai-model-risk-suite"
    assert "exp" in payload
    assert "iat" in payload
    assert payload["exp"] > payload["iat"]


def test_jwt_expired_token_rejected():
    """Verify expired JWT is rejected with expiration error."""
    expired_delta = datetime.timedelta(seconds=-10)
    token = create_access_token(username="admin", role="admin", expires_delta=expired_delta)

    with pytest.raises(ValueError, match="Token signature has expired"):
        decode_access_token(token)


def test_jwt_unsigned_alg_none_token_rejected():
    """Verify unsigned token using alg=none is rejected."""
    payload = {
        "sub": "attacker",
        "role": "admin",
        "iss": "xai-model-risk-suite",
        "iat": int(datetime.datetime.now(timezone.utc).timestamp()),
        "exp": int((datetime.datetime.now(timezone.utc) + datetime.timedelta(hours=1)).timestamp()),
    }
    # Forge token with alg=none
    unsigned_token = jwt.encode(payload, key="", algorithm="none")

    with pytest.raises(ValueError, match="Invalid authentication token"):
        decode_access_token(unsigned_token)


def test_jwt_wrong_secret_key_rejected():
    """Verify token signed with an unauthorized secret key is rejected."""
    token = create_access_token(username="attacker", role="admin", secret_key="wrong-secret-key-123-min-32-chars-long")

    with pytest.raises(ValueError, match="Invalid authentication token"):
        decode_access_token(token)


# ------------------------------------------------------------------------------
# API Endpoint Authentication & Authorization Tests
# ------------------------------------------------------------------------------

def test_api_login_endpoint_success_and_failure():
    """Verify /api/auth/token endpoint issues valid JWT on correct credentials."""
    # Successful login
    login_res = client.post(
        "/api/auth/token",
        json={"username": "admin", "password": "AdminSecure2026!"},
    )
    assert login_res.status_code == 200
    data = login_res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["role"] == "admin"
    assert data["expires_in"] == 3600

    # Failed login
    bad_res = client.post(
        "/api/auth/token",
        json={"username": "admin", "password": "WrongPassword!"},
    )
    assert bad_res.status_code == 401
    assert "Incorrect username or password" in bad_res.json()["message"]


def test_api_rejects_missing_authorization_header():
    """Verify protected endpoint rejects requests lacking Authorization header."""
    res = client.get("/api/models")
    assert res.status_code == 401
    assert "Missing Authorization header" in res.json()["message"]


def test_api_rejects_bare_client_supplied_x_user_role_header():
    """Verify client-supplied X-User-Role header without JWT is explicitly rejected with 401."""
    # Attempt to access using legacy X-User-Role spoofing
    res = client.get("/api/models", headers={"X-User-Role": "admin"})
    assert res.status_code == 401
    assert "X-User-Role" in res.json()["message"]
    assert "untrusted and rejected" in res.json()["message"]


def test_api_rejects_expired_jwt():
    """Verify API endpoint rejects expired JWT with 401."""
    expired_token = create_access_token(
        username="admin",
        role="admin",
        expires_delta=datetime.timedelta(seconds=-60),
    )
    res = client.get("/api/models", headers={"Authorization": f"Bearer {expired_token}"})
    assert res.status_code == 401
    assert "expired" in res.json()["message"].lower()


def test_api_rejects_unsigned_alg_none_token():
    """Verify API endpoint rejects unsigned alg=none token with 401."""
    payload = {
        "sub": "hacker",
        "role": "admin",
        "iss": "xai-model-risk-suite",
        "iat": int(datetime.datetime.now(timezone.utc).timestamp()),
        "exp": int((datetime.datetime.now(timezone.utc) + datetime.timedelta(hours=1)).timestamp()),
    }
    unsigned_token = jwt.encode(payload, key="", algorithm="none")

    res = client.get("/api/models", headers={"Authorization": f"Bearer {unsigned_token}"})
    assert res.status_code == 401
    assert "Authentication failed" in res.json()["message"]


def test_api_rejects_forged_secret_key_token():
    """Verify API endpoint rejects token signed with an invalid key."""
    forged_token = create_access_token(
        username="hacker",
        role="admin",
        secret_key="attacker-tampered-secret-key-xyz",
    )
    res = client.get("/api/models", headers={"Authorization": f"Bearer {forged_token}"})
    assert res.status_code == 401
    assert "Authentication failed" in res.json()["message"]


def test_api_role_authorization_permits_authorized_role():
    """Verify valid token permits authorized roles."""
    analyst_token = create_access_token(username="analyst_user", role="analyst")
    res = client.get("/api/models", headers={"Authorization": f"Bearer {analyst_token}"})
    assert res.status_code == 200
    assert "models" in res.json()


def test_api_role_authorization_denies_unauthorized_role():
    """Verify role checks prevent unauthorized role actions with 403 Forbidden."""
    analyst_token = create_access_token(username="analyst_user", role="analyst")

    # Analyst trying to reindex documents (requires admin or compliance_reviewer)
    reindex_res = client.post(
        "/api/documents/SYN-POL-001/reindex",
        headers={"Authorization": f"Bearer {analyst_token}"},
    )
    assert reindex_res.status_code == 403
    assert "Access denied" in reindex_res.json()["message"]

    # Analyst trying to record reviewer disposition (requires reviewer, compliance_reviewer, or admin)
    disp_res = client.post(
        "/api/audits/AUDIT-123/disposition",
        json={"disposition": "approve", "notes": "Approved by analyst", "reviewer_id": "ANALYST-1"},
        headers={"Authorization": f"Bearer {analyst_token}"},
    )
    assert disp_res.status_code == 403
    assert "Access denied" in disp_res.json()["message"]


def test_api_rejects_conflicting_x_user_role_header_spoofing():
    """Verify that providing an X-User-Role header conflicting with JWT server claim is rejected with 403."""
    analyst_token = create_access_token(username="analyst_user", role="analyst")

    # Analyst token with spoofed X-User-Role: admin
    res = client.get(
        "/api/models",
        headers={
            "Authorization": f"Bearer {analyst_token}",
            "X-User-Role": "admin",
        },
    )
    assert res.status_code == 403
    assert "Header spoofing detected" in res.json()["message"]
