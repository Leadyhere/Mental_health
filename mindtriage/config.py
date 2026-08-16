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


@dataclass(frozen=True)
class Settings:
    app_name: str = "MindTriage"
    environment: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    local_risk_model_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("LOCAL_MODEL_DIR", BASE_DIR / "models" / "inhouse_risk_classifier")
        )
    )
    local_nlp_model_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("LOCAL_NLP_MODEL_DIR", BASE_DIR / "models" / "inhouse_nlp")
        )
    )
    enable_local_dialogue_model: bool = field(
        default_factory=lambda: _as_bool("ENABLE_LOCAL_DIALOGUE_MODEL", False)
    )
    local_dialogue_model_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("LOCAL_DIALOGUE_MODEL_DIR", BASE_DIR / "models" / "inhouse_llama_dialogue")
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
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", f"sqlite:///{(BASE_DIR / 'data' / 'mindtriage.db').as_posix()}"
        )
    )
    analytics_hash_secret: str = field(
        default_factory=lambda: os.getenv("ANALYTICS_HASH_SECRET", "development-only-change-me")
    )
    dataset_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("CONVERSATIONS_LOG_PATH", BASE_DIR / "data" / "conversations.jsonl")
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

    def validate_for_production(self) -> None:
        if self.environment != "production":
            return
        if self.analytics_hash_secret == "development-only-change-me":
            raise RuntimeError("ANALYTICS_HASH_SECRET must be changed in production")
        if not self.database_url.startswith(("postgresql://", "postgres://")):
            raise RuntimeError("Production requires a PostgreSQL DATABASE_URL")


def get_settings() -> Settings:
    settings = Settings()
    settings.validate_for_production()
    return settings
