import json
import copy
import threading
import time
import uuid
from typing import Protocol

from .config import Settings


class SessionStore(Protocol):
    name: str

    def create(self, training_consent: bool, locale: str) -> dict: ...
    def get(self, session_id: str) -> dict | None: ...
    def save(self, session: dict) -> bool: ...
    def delete(self, session_id: str) -> bool: ...


def new_session(training_consent: bool, locale: str, ttl_seconds: int) -> dict:
    now = time.time()
    return {
        "session_id": f"sess_{uuid.uuid4().hex}",
        "dataset_conversation_id": f"conv_{uuid.uuid4().hex}",
        "created_at": now,
        "expires_at": now + ttl_seconds,
        "training_consent": training_consent,
        "locale": locale,
        "chat_history": [],
        "filled_slots": {
            "trigger": None,
            "duration": None,
            "impact": None,
            "emotions": [],
            "coping": None,
            "support": None,
        },
        "turn_history": [],
        "current_risk_level": "Level 1 - Low (Watchful)",
        "highest_risk_level": "Level 1 - Low (Watchful)",
        "turn_count": 0,
    }


class MemorySessionStore:
    name = "memory"

    def __init__(self, ttl_seconds: int):
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, dict] = {}
        self._lock = threading.RLock()

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [key for key, value in self._sessions.items() if value["expires_at"] <= now]
        for key in expired:
            del self._sessions[key]

    def create(self, training_consent: bool, locale: str) -> dict:
        session = new_session(training_consent, locale, self.ttl_seconds)
        with self._lock:
            self._purge_expired()
            self._sessions[session["session_id"]] = copy.deepcopy(session)
        return copy.deepcopy(session)

    def get(self, session_id: str) -> dict | None:
        with self._lock:
            self._purge_expired()
            session = self._sessions.get(session_id)
            return copy.deepcopy(session) if session is not None else None

    def save(self, session: dict) -> bool:
        with self._lock:
            self._purge_expired()
            if session["session_id"] not in self._sessions:
                return False
            stored = copy.deepcopy(session)
            stored["expires_at"] = time.time() + self.ttl_seconds
            self._sessions[stored["session_id"]] = stored
            return True

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None


class RedisSessionStore:
    name = "redis"

    def __init__(self, client, ttl_seconds: int):
        self.client = client
        self.ttl_seconds = ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"mindtriage:session:{session_id}"

    def create(self, training_consent: bool, locale: str) -> dict:
        session = new_session(training_consent, locale, self.ttl_seconds)
        self.client.setex(
            self._key(session["session_id"]), self.ttl_seconds, json.dumps(session)
        )
        return session

    def get(self, session_id: str) -> dict | None:
        value = self.client.get(self._key(session_id))
        return json.loads(value) if value else None

    def save(self, session: dict) -> bool:
        session["expires_at"] = time.time() + self.ttl_seconds
        result = self.client.eval(
            """
            if redis.call('EXISTS', KEYS[1]) == 1 then
                redis.call('SETEX', KEYS[1], ARGV[1], ARGV[2])
                return 1
            end
            return 0
            """,
            1,
            self._key(session["session_id"]),
            self.ttl_seconds,
            json.dumps(session),
        )
        return bool(result)

    def delete(self, session_id: str) -> bool:
        return bool(self.client.delete(self._key(session_id)))


def build_session_store(settings: Settings) -> SessionStore:
    if settings.enable_redis:
        try:
            import redis

            client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            client.ping()
            return RedisSessionStore(client, settings.session_ttl_seconds)
        except Exception as exc:
            if settings.environment == "production":
                raise RuntimeError("Redis is required and unavailable in production") from exc
            print(f"[WARNING] Redis unavailable; using process memory: {exc}")
    return MemorySessionStore(settings.session_ttl_seconds)
