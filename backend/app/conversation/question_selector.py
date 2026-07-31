"""Phase 2: pick the next unanswered structured question, honoring the
fixed safety-first rank order, the narrative's answered_questions map, and
the 15-question cap."""
from __future__ import annotations

from app.conversation.question_bank import (
    MAX_STRUCTURED_QUESTIONS,
    QuestionBankItem,
    get_next_question,
)
from app.models.schemas import NarrativeAnalysis, StructuredAnswer


def select_next_question(
    narrative_analysis: NarrativeAnalysis | None,
    structured_answers: list[StructuredAnswer],
    structured_question_count: int,
) -> QuestionBankItem | None:
    if structured_question_count >= MAX_STRUCTURED_QUESTIONS:
        return None

    answered_ids = {a.question_id for a in structured_answers}
    prior_answers = {a.question_id: (a.chip_selected or a.answer_text) for a in structured_answers}
    narrative_answered = set()
    if narrative_analysis:
        narrative_answered = {k for k, v in narrative_analysis.answered_questions.items() if v}

    return get_next_question(answered_ids, narrative_answered, prior_answers)
