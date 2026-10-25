"""Authentication API routes for login, JWT issuance, and session verification."""

from __future__ import annotations

from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.services.security import (
    authenticate_user,
    create_access_token,
    decode_access_token,
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    """Credentials payload for JWT token issuance."""

    username: str = Field(..., description="User identifier")
    password: str = Field(..., description="User plaintext password")


class TokenResponse(BaseModel):
    """Cryptographically signed JWT bearer token response."""

    access_token: str
    token_type: str = "bearer"
    role: str
    username: str
    expires_in: int = 3600


class UserInfoResponse(BaseModel):
    """Authenticated user session information."""

    username: str
    role: str
    issuer: str


@router.post(
    "/token",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate user and issue cryptographically signed JWT with server-side role claims",
)
def login_for_access_token(credentials: LoginRequest) -> TokenResponse:
    """Validate credentials and return HS256 JWT with signed server-side role claims."""
    user = authenticate_user(credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(username=user["username"], role=user["role"])
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        role=user["role"],
        username=user["username"],
        expires_in=3600,
    )


@router.post(
    "/verify",
    response_model=UserInfoResponse,
    summary="Cryptographically verify an existing JWT token",
)
def verify_token_endpoint(token: str) -> UserInfoResponse:
    """Verify signature and return token claims."""
    try:
        payload = decode_access_token(token)
        return UserInfoResponse(
            username=payload["sub"],
            role=payload["role"],
            issuer=payload.get("iss", "unknown"),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
