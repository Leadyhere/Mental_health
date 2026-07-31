from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import get_settings


class Base(DeclarativeBase):
    pass


class AnonymizedSessionRecord(Base):
    """Persisted post-teardown. Never stores raw conversation text."""

    __tablename__ = "anonymized_session_records"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String, index=True)
    risk_level: Mapped[str] = mapped_column(String)
    sub_labels: Mapped[dict] = mapped_column(JSON)
    confidence: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String)
    embeddings_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    reviewer_verdict: Mapped[str | None] = mapped_column(String, nullable=True)


class TrainingTriple(Base):
    """Stage 2 logging row: (narrative + structured answers) -> LLM decision.

    `is_reviewed`/`reviewer_verdict` exist per the deferred human-review
    decision -- no review UI yet, but the field is here so one can be added
    later without a schema migration surprise.
    """

    __tablename__ = "training_triples"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String, index=True)
    narrative_features: Mapped[dict] = mapped_column(JSON)
    structured_answers: Mapped[dict] = mapped_column(JSON)
    llm_risk_level: Mapped[str] = mapped_column(String)
    llm_sub_labels: Mapped[dict] = mapped_column(JSON)
    llm_rationale: Mapped[str | None] = mapped_column(String, nullable=True)
    inhouse_risk_level: Mapped[str | None] = mapped_column(String, nullable=True)
    inhouse_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    reviewer_verdict: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    """Tracks each in-house checkpoint version + its agreement rate vs. the
    LLM on held-out sessions, so Stage 3's 'flip once agreement is high'
    decision has an auditable trail."""

    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    version: Mapped[int] = mapped_column(Integer, unique=True)
    agreement_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    eval_sample_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RetrainCounter(Base):
    """Single-row counter of sessions torn down since the last retrain,
    used to trigger Stage 2 fine-tuning every N sessions."""

    __tablename__ = "retrain_counter"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    sessions_since_last_retrain: Mapped[int] = mapped_column(Integer, default=0)


_engine = None
_session_maker: async_sessionmaker[AsyncSession] | None = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(settings.database_url, echo=False)
    return _engine


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    global _session_maker
    if _session_maker is None:
        _session_maker = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_maker


async def init_db() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
