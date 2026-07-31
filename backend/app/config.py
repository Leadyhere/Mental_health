from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8", extra="ignore")

    grok_api_key: str = ""
    grok_model: str = "grok-2-latest"
    grok_api_base: str = "https://api.x.ai/v1"

    redis_url: str = ""  # empty -> use in-process fakeredis
    database_url: str = "sqlite+aiosqlite:///./dev.db"

    jwt_secret: str = "dev-only-change-me"
    jwt_algorithm: str = "HS256"

    retrain_every_n_sessions: int = 50
    inhouse_model_artifact_dir: str = "../inhouse_model/artifacts"

    session_idle_timeout_minutes: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
