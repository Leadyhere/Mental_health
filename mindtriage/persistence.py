import hashlib
import hmac
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings


SCHEMA_SQL = (
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_hash TEXT PRIMARY KEY,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        training_consent INTEGER NOT NULL,
        locale TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS risk_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        risk_level TEXT NOT NULL,
        is_emergency INTEGER NOT NULL,
        model_version TEXT NOT NULL,
        reason_codes TEXT NOT NULL
    )
    """,
)


class AnalyticsRepository:
    """Stores no raw user text; PostgreSQL and SQLite are supported."""

    def __init__(self, settings: Settings):
        self.database_url = settings.database_url
        self.secret = settings.analytics_hash_secret.encode("utf-8")
        self._lock = threading.RLock()
        self.backend = "postgresql" if self.database_url.startswith(("postgresql://", "postgres://")) else "sqlite"
        self._initialize()

    def hash_session(self, session_id: str) -> str:
        return hmac.new(self.secret, session_id.encode("utf-8"), hashlib.sha256).hexdigest()

    def _sqlite_path(self) -> Path:
        raw = self.database_url.removeprefix("sqlite:///")
        path = Path(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _connect(self):
        if self.backend == "postgresql":
            try:
                import psycopg
            except ImportError as exc:
                raise RuntimeError("Install psycopg[binary] for PostgreSQL support") from exc
            return psycopg.connect(self.database_url)
        return sqlite3.connect(self._sqlite_path())

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        if self.backend == "postgresql":
            statements = (
                SCHEMA_SQL[0],
                SCHEMA_SQL[1].replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY"),
            )
        else:
            statements = SCHEMA_SQL
        with self._lock, self._connection() as connection:
            for statement in statements:
                connection.execute(statement)
            connection.commit()

    def session_started(self, session: dict) -> None:
        session_hash = self.hash_session(session["session_id"])
        params = (
            session_hash,
            datetime.now(timezone.utc).isoformat(),
            int(session["training_consent"]),
            session["locale"],
        )
        sql = (
            "INSERT INTO sessions(session_hash, started_at, training_consent, locale) "
            "VALUES (?, ?, ?, ?)"
        )
        if self.backend == "postgresql":
            sql = sql.replace("?", "%s") + " ON CONFLICT (session_hash) DO NOTHING"
        with self._lock, self._connection() as connection:
            connection.execute(sql, params)
            connection.commit()

    def risk_event(
        self,
        session_id: str,
        risk_level: str,
        is_emergency: bool,
        model_version: str,
        reason_codes: list[str],
    ) -> None:
        params = (
            self.hash_session(session_id),
            datetime.now(timezone.utc).isoformat(),
            risk_level,
            int(is_emergency),
            model_version,
            ",".join(reason_codes),
        )
        sql = (
            "INSERT INTO risk_events(session_hash, created_at, risk_level, is_emergency, "
            "model_version, reason_codes) VALUES (?, ?, ?, ?, ?, ?)"
        )
        if self.backend == "postgresql":
            sql = sql.replace("?", "%s")
        with self._lock, self._connection() as connection:
            connection.execute(sql, params)
            connection.commit()

    def session_ended(self, session_id: str) -> None:
        sql = "UPDATE sessions SET ended_at = ? WHERE session_hash = ?"
        params = (datetime.now(timezone.utc).isoformat(), self.hash_session(session_id))
        if self.backend == "postgresql":
            sql = sql.replace("?", "%s")
        with self._lock, self._connection() as connection:
            connection.execute(sql, params)
            connection.commit()

    def health(self) -> bool:
        try:
            with self._connection() as connection:
                connection.execute("SELECT 1")
            return True
        except Exception:
            return False
