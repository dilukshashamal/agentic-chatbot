from __future__ import annotations

import base64
import hmac
import json
import time
from hashlib import sha256
from typing import Any

from fastapi import Depends, Header, HTTPException, status

from app.core.config import Settings, get_settings


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _sign(payload: str, settings: Settings) -> str:
    digest = hmac.new(settings.admin_session_secret.encode("utf-8"), payload.encode("ascii"), sha256).digest()
    return _base64url_encode(digest)


def authenticate_admin(username: str, password: str, settings: Settings) -> bool:
    username_matches = hmac.compare_digest(username, settings.admin_username)
    password_matches = hmac.compare_digest(password, settings.admin_password)
    return username_matches and password_matches


def create_admin_token(settings: Settings) -> str:
    now = int(time.time())
    payload = _base64url_encode(
        json.dumps(
            {
                "sub": settings.admin_username,
                "iat": now,
                "exp": now + settings.admin_session_ttl_minutes * 60,
            },
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return f"{payload}.{_sign(payload, settings)}"


def verify_admin_token(token: str, settings: Settings) -> dict[str, Any]:
    try:
        payload, signature = token.split(".", 1)
        expected_signature = _sign(payload, settings)
        if not hmac.compare_digest(signature, expected_signature):
            raise ValueError("Invalid signature")
        claims = json.loads(_base64url_decode(payload))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin session.",
        ) from exc

    if claims.get("sub") != settings.admin_username or int(claims.get("exp", 0)) < int(time.time()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin session expired.",
        )
    return claims


def require_admin(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return verify_admin_token(token, settings)
