from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from app.config import get_settings


def issue_session_token(session_id: str) -> str:
    settings = get_settings()
    payload = {
        "sub": session_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=6),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def verify_session_token(token: str) -> str | None:
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload.get("sub")
    except JWTError:
        return None
