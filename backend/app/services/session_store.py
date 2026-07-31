"""Ephemeral session state storage.

Wraps Redis behind a factory: if REDIS_URL is set, use real async Redis;
otherwise fall back to an in-process fakeredis instance for local dev. Same
code path either way -- swapping in real Redis via docker-compose later is
an env-var change, not a code change.
"""
from __future__ import annotations

from uuid import UUID

from app.config import get_settings
from app.models.schemas import SessionState

_SESSION_KEY_PREFIX = "session:"
_shared_fake_redis = None


def _get_redis_client():
    global _shared_fake_redis
    settings = get_settings()
    if settings.redis_url:
        import redis.asyncio as redis
        return redis.from_url(settings.redis_url, decode_responses=True)

    if _shared_fake_redis is None:
        import fakeredis.aioredis
        _shared_fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return _shared_fake_redis


class SessionStore:
    def __init__(self):
        self._redis = _get_redis_client()
        self._idle_timeout_seconds = get_settings().session_idle_timeout_minutes * 60

    def _key(self, session_id: UUID | str) -> str:
        return f"{_SESSION_KEY_PREFIX}{session_id}"

    async def save(self, state: SessionState) -> None:
        await self._redis.set(
            self._key(state.session_id),
            state.model_dump_json(),
            ex=self._idle_timeout_seconds,
        )

    async def get(self, session_id: UUID | str) -> SessionState | None:
        raw = await self._redis.get(self._key(session_id))
        if raw is None:
            return None
        return SessionState.model_validate_json(raw)

    async def delete(self, session_id: UUID | str) -> None:
        await self._redis.delete(self._key(session_id))

    async def exists(self, session_id: UUID | str) -> bool:
        return bool(await self._redis.exists(self._key(session_id)))


_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
