"""Stage 2 logging: persists the anonymized (narrative + structured answers)
-> LLM decision triple, and forwards the same triple to the in-house model's
`log_llm_decision` for the periodic retraining job. Never persists raw
narrative text -- only the already-summarized `narrative_analysis` fields
and the (short, categorical) structured answers.
"""
from __future__ import annotations

import logging

from app.models.db import RetrainCounter, TrainingTriple, get_session_maker
from app.models.schemas import RiskAssessment, SessionState

logger = logging.getLogger(__name__)


def _turns_for_inhouse_model(state: SessionState) -> list[str]:
    turns = []
    if state.narrative_analysis:
        turns.append(f"emotions: {', '.join(state.narrative_analysis.emotions)}")
        turns.append(f"coping_signals: {', '.join(state.narrative_analysis.coping_signals)}")
        turns.append(f"support_signals: {', '.join(state.narrative_analysis.support_signals)}")
        turns.append(f"functional_impact: {state.narrative_analysis.functional_impact or ''}")
    for a in state.structured_answers:
        turns.append(f"{a.question_id}: {a.chip_selected or a.answer_text or ''}")
    return turns


async def log_session_decision(state: SessionState, assessment: RiskAssessment) -> None:
    narrative_features = state.narrative_analysis.model_dump() if state.narrative_analysis else {}
    structured_answers = [a.model_dump() for a in state.structured_answers]

    session_maker = get_session_maker()
    async with session_maker() as db_session:
        triple = TrainingTriple(
            session_id=str(state.session_id),
            narrative_features=narrative_features,
            structured_answers=structured_answers,
            llm_risk_level=assessment.risk_level,
            llm_sub_labels=assessment.sub_labels.model_dump(),
            llm_rationale=assessment.rationale,
        )
        db_session.add(triple)

        counter = await db_session.get(RetrainCounter, 1)
        if counter is None:
            counter = RetrainCounter(id=1, sessions_since_last_retrain=0)
            db_session.add(counter)
        counter.sessions_since_last_retrain += 1

        await db_session.commit()

    try:
        from inhouse_model.model import InHouseModel

        InHouseModel().log_llm_decision(
            session=_turns_for_inhouse_model(state),
            llm_output={
                "risk_level": assessment.risk_level,
                "sub_labels": assessment.sub_labels.model_dump(),
                "rationale": assessment.rationale,
                "confidence": assessment.confidence,
            },
        )
    except Exception as exc:
        logger.warning("could not forward decision to in-house model logger: %s", exc)
