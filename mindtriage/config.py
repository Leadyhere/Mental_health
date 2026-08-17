import os
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _path_setting(name: str, default: Path) -> Path:
    path = Path(os.getenv(name, str(default))).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def _database_url() -> str:
    value = os.getenv("DATABASE_URL", "sqlite:///data/mindtriage.db")
    if not value.startswith("sqlite:///"):
        return value
    path = Path(value.removeprefix("sqlite:///")).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return f"sqlite:///{path.as_posix()}"


@dataclass(frozen=True)
class Settings:
    app_name: str = "MindTriage"
    environment: str = field(
        default_factory=lambda: os.getenv("APP_ENV", "development").strip().lower()
    )
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    local_risk_model_dir: Path = field(
        default_factory=lambda: _path_setting(
            "LOCAL_MODEL_DIR", BASE_DIR / "models" / "inhouse_risk_classifier"
        )
    )
    local_nlp_model_dir: Path = field(
        default_factory=lambda: _path_setting(
            "LOCAL_NLP_MODEL_DIR", BASE_DIR / "models" / "inhouse_nlp"
        )
    )
    enable_local_dialogue_model: bool = field(
        default_factory=lambda: _as_bool("ENABLE_LOCAL_DIALOGUE_MODEL", False)
    )
    local_dialogue_model_dir: Path = field(
        default_factory=lambda: _path_setting(
            "LOCAL_DIALOGUE_MODEL_DIR", BASE_DIR / "models" / "inhouse_llama_dialogue"
        )
    )
    local_dialogue_base_model: str = field(
        default_factory=lambda: os.getenv(
            "LOCAL_DIALOGUE_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
    )
    enable_redis: bool = field(default_factory=lambda: _as_bool("ENABLE_REDIS", True))
    redis_host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    redis_password: str | None = field(default_factory=lambda: os.getenv("REDIS_PASSWORD") or None)
    session_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("SESSION_TTL_SECONDS", "3600")))
    database_url: str = field(default_factory=_database_url)
    analytics_hash_secret: str = field(
        default_factory=lambda: os.getenv("ANALYTICS_HASH_SECRET", "development-only-change-me")
    )
    dataset_path: Path = field(
        default_factory=lambda: _path_setting(
            "CONVERSATIONS_LOG_PATH", BASE_DIR / "data" / "conversations.jsonl"
        )
    )
    frontend_path: Path = field(default_factory=lambda: BASE_DIR / "index.html")
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            value.strip()
            for value in os.getenv(
                "CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
            ).split(",")
            if value.strip()
        )
    )
    max_message_chars: int = field(default_factory=lambda: int(os.getenv("MAX_MESSAGE_CHARS", "4000")))

    def validate(self) -> None:
        if self.environment not in {"development", "test", "production"}:
            raise RuntimeError("APP_ENV must be development, test, or production")
        if not 1 <= self.redis_port <= 65535:
            raise RuntimeError("REDIS_PORT must be between 1 and 65535")
        if self.session_ttl_seconds <= 0:
            raise RuntimeError("SESSION_TTL_SECONDS must be greater than zero")
        if not 1 <= self.max_message_chars <= 4000:
            raise RuntimeError("MAX_MESSAGE_CHARS must be between 1 and 4000")
        if not self.database_url.startswith(("sqlite:///", "postgresql://", "postgres://")):
            raise RuntimeError("DATABASE_URL must use SQLite or PostgreSQL")
        if "*" in self.cors_origins:
            raise RuntimeError("CORS_ORIGINS cannot use a wildcard when credentials are enabled")
        if self.environment != "production":
            return
        if len(self.analytics_hash_secret) < 32 or self.analytics_hash_secret in {
            "development-only-change-me",
            "replace-with-a-long-random-secret",
        }:
            raise RuntimeError("ANALYTICS_HASH_SECRET must be a unique 32+ character value")
        if not self.database_url.startswith(("postgresql://", "postgres://")):
            raise RuntimeError("Production requires a PostgreSQL DATABASE_URL")
        if not self.enable_redis:
            raise RuntimeError("Production requires Redis session storage")

    def validate_for_production(self) -> None:
        """Backward-compatible alias for older callers."""
        self.validate()


def get_settings() -> Settings:
    settings = Settings()
    settings.validate()
    return settings
