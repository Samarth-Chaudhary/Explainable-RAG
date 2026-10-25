"""FastAPI request dependencies, session providers, and role-based access hooks.

Enforces cryptographically verified JWT bearer tokens with server-side role claims.
Rejects client-supplied X-User-Role headers, unsigned tokens, and expired credentials.
"""

from __future__ import annotations

from typing import Annotated, Generator
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.config import Settings, get_settings
from backend.app.repositories.database import get_session, init_db
from backend.app.services.orchestrator import RunAuditOrchestrator
from backend.app.services.security import decode_access_token

_orchestrator_instance: RunAuditOrchestrator | None = None

VALID_ROLES = {
    "analyst",
    "reviewer",
    "admin",
    "compliance_reviewer",
    "compliance_officer",
    "credit_officer",
    "auditor",
    "model_risk_analyst",
}


def get_db() -> Generator[Session, None, None]:
    """Provide a transactional database session per request."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def get_orchestrator() -> RunAuditOrchestrator:
    """Provide singleton RunAuditOrchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        init_db()
        _orchestrator_instance = RunAuditOrchestrator()
    return _orchestrator_instance


def get_current_user_role(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    x_user_role: Annotated[str | None, Header(alias="X-User-Role")] = None,
) -> str:
    """Extract and cryptographically verify authenticated user role from JWT bearer token.

    Security Rules:
    1. If client supplies X-User-Role header without a valid JWT, reject with 401 Unauthorized.
    2. If Authorization header is missing or malformed, reject with 401 Unauthorized.
    3. Unsigned tokens (alg=none) or invalid signatures are rejected with 401 Unauthorized.
    4. Expired tokens are rejected with 401 Unauthorized.
    5. If client supplies both a valid JWT and an X-User-Role header that contradicts
       the server-side role claim, reject with 403 Forbidden (spoofing detected).
    6. Returns verified server-side role claim.
    """
    if not authorization:
        if x_user_role:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Client-supplied 'X-User-Role' header is untrusted and rejected. Valid JWT Bearer token required in Authorization header.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Bearer token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.strip().split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1]
    try:
        payload = decode_access_token(token)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    server_role = str(payload.get("role", "")).lower().strip()
    if server_role not in VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"JWT token contains invalid role claim '{server_role}'.",
        )

    # Detect header spoofing if client also passed conflicting X-User-Role
    if x_user_role:
        client_role = x_user_role.lower().strip()
        if client_role != server_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Header spoofing detected: X-User-Role '{client_role}' contradicts verified server-side JWT claim '{server_role}'.",
            )

    return server_role


def require_role(allowed_roles: list[str]):
    """Enforce role-based access control based on verified server-side JWT claim."""
    def role_checker(role: Annotated[str, Depends(get_current_user_role)]) -> str:
        if role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. User role '{role}' is not in authorized roles {allowed_roles}.",
            )
        return role

    return role_checker
