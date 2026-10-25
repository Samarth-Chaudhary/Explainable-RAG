"""Authentication, Password Hashing, and JWT Service.

Provides cryptographically defensible authentication:
1. PBKDF2-HMAC-SHA256 password hashing with constant-time verification.
2. JWT issuance and verification with server-side role claims and token expiry.
3. Strict rejection of unsigned tokens (alg=none), wrong keys, and expired tokens.
"""

from __future__ import annotations

import datetime
from datetime import timezone
import hashlib
import hmac
import secrets
from typing import Any
import jwt

from backend.app.config import get_settings

SETTINGS = get_settings()
JWT_SECRET_KEY = getattr(
    SETTINGS,
    "api_secret_key",
    None,
) or "xai-model-risk-defensible-jwt-secret-key-sha256-2026-min-32-chars"
JWT_ALGORITHM = "HS256"
DEFAULT_EXPIRY_MINUTES = 60

# PBKDF2 Configuration
PBKDF2_ITERATIONS = 100_000
PBKDF2_ALGORITHM = "sha256"


def hash_password(password: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with cryptographically random salt."""
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        PBKDF2_ALGORITHM,
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_{PBKDF2_ALGORITHM}${PBKDF2_ITERATIONS}${salt}${key.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against PBKDF2 hash using constant-time comparison."""
    try:
        parts = hashed_password.split("$")
        if len(parts) != 4 or parts[0] != f"pbkdf2_{PBKDF2_ALGORITHM}":
            return False
        iterations = int(parts[1])
        salt = parts[2]
        expected_hex = parts[3]

        derived_key = hashlib.pbkdf2_hmac(
            PBKDF2_ALGORITHM,
            plain_password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations,
        )
        return hmac.compare_digest(derived_key.hex(), expected_hex)
    except Exception:
        return False


# In-memory authoritatively hashed users with server-side role claims
USER_DATABASE: dict[str, dict[str, Any]] = {
    "admin": {
        "username": "admin",
        "hashed_password": hash_password("AdminSecure2026!"),
        "role": "admin",
        "full_name": "Model Governance Administrator",
        "active": True,
    },
    "credit_officer": {
        "username": "credit_officer",
        "hashed_password": hash_password("OfficerSecure2026!"),
        "role": "credit_officer",
        "full_name": "Senior Credit Officer",
        "active": True,
    },
    "auditor": {
        "username": "auditor",
        "hashed_password": hash_password("AuditorSecure2026!"),
        "role": "auditor",
        "full_name": "Internal Model Risk Auditor",
        "active": True,
    },
    "compliance_reviewer": {
        "username": "compliance_reviewer",
        "hashed_password": hash_password("ComplianceSecure2026!"),
        "role": "compliance_reviewer",
        "full_name": "Regulatory Compliance Reviewer",
        "active": True,
    },
    "analyst": {
        "username": "analyst",
        "hashed_password": hash_password("AnalystSecure2026!"),
        "role": "analyst",
        "full_name": "Credit Risk Analyst",
        "active": True,
    },
    "reviewer": {
        "username": "reviewer",
        "hashed_password": hash_password("ReviewerSecure2026!"),
        "role": "reviewer",
        "full_name": "Credit Risk Reviewer",
        "active": True,
    },
}


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    """Authenticate user against authoritative user database."""
    user = USER_DATABASE.get(username)
    if not user or not user.get("active"):
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(
    username: str,
    role: str,
    expires_delta: datetime.timedelta | None = None,
    secret_key: str = JWT_SECRET_KEY,
) -> str:
    """Create signed HS256 JWT containing server-side role claim and expiry."""
    now = datetime.datetime.now(timezone.utc)
    expiry = now + (expires_delta or datetime.timedelta(minutes=DEFAULT_EXPIRY_MINUTES))

    payload = {
        "sub": username,
        "role": role,
        "iss": "xai-model-risk-suite",
        "iat": int(now.timestamp()),
        "exp": int(expiry.timestamp()),
    }
    return jwt.encode(payload, secret_key, algorithm=JWT_ALGORITHM)


def decode_access_token(
    token: str,
    secret_key: str = JWT_SECRET_KEY,
) -> dict[str, Any]:
    """Decode and cryptographically verify JWT token.
    
    Rejects:
    - Unsigned / alg=none tokens
    - Tokens signed with invalid keys
    - Expired tokens
    - Malformed payloads
    """
    try:
        payload = jwt.decode(
            token,
            secret_key,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["exp", "iat", "sub", "role"]},
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token signature has expired.")
    except jwt.InvalidTokenError as e:
        raise ValueError(f"Invalid authentication token: {str(e)}")
