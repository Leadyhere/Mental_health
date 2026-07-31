"""Ephemeral Session Architecture: deletes the raw conversation from Redis
on report generation or idle timeout, persisting only the anonymized
fields (risk level, sub-labels, confidence, model version) -- never raw
text.
"""
from __future__ import annotations

from app.models.db import AnonymizedSessionRecord, get_session_maker
from app.models.schemas import SessionState
from app.services.session_store import get_session_store


async def teardown_session(state: SessionState) -> None:
    assessment = state.current_risk_assessment
    if assessment is not None:
        session_maker = get_session_maker()
        async with session_maker() as db_session:
            record = AnonymizedSessionRecord(
                session_id=str(state.session_id),
                risk_level=assessment.risk_level,
                sub_labels=assessment.sub_labels.model_dump(),
                confidence=assessment.confidence,
                model_version=assessment.source,
                is_reviewed=assessment.is_reviewed,
                reviewer_verdict=assessment.reviewer_verdict,
            )
            db_session.add(record)
            await db_session.commit()

    await get_session_store().delete(state.session_id)
