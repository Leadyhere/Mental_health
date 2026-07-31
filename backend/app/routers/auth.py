"""Stub auth: sessions are anonymous. The JWT issued alongside session
creation (see routers/sessions.py) is the only credential -- this endpoint
just lets a client check whether a token it's holding is still valid.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.deps import verify_session_token

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.get("/verify")
async def verify(token: str):
    session_id = verify_session_token(token)
    return {"valid": session_id is not None, "session_id": session_id}
