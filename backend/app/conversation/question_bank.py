"""Q1-Q15 question bank for Phase 2 (adaptive structured questions).

Order is fixed and safety-first: risk-sensing questions are NEVER moved
earlier, even if the narrative already answered everything else and the
risk questions are the only gap. The two risk-sensing questions use the
exact research-standard verbatim phrasing from the spec -- these strings
must never be paraphrased by the LLM. `chat_ws.py` renders them directly
from these constants, bypassing `llm/client.py` entirely for those turns.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class QuestionBankItem:
    id: str
    rank: int
    prompt: str
    chip_options: list[str]
    is_risk_gating: bool = False
    asked_verbatim: bool = False
    # narrative_analysis key this question can be pre-answered by, if present
    narrative_field: str | None = None
    # only asked if this predicate on prior answers returns True (else skipped)
    condition_on: str | None = None


PASSIVE_SI_PROMPT = (
    "Sometimes when people feel overwhelmed, they wish they could just stop "
    "existing or go to sleep and not wake up. Have you had thoughts like "
    "that recently?"
)
PASSIVE_SI_OPTIONS = ["No", "Occasionally", "Often", "prefer not to answer"]

ACTIVE_SI_PROMPT = "Have you had thoughts about hurting yourself or ending your life?"
ACTIVE_SI_OPTIONS = ["No", "Yes but wouldn't act", "Yes and it feels risky right now"]

QUESTION_BANK: list[QuestionBankItem] = [
    QuestionBankItem(
        id="emotional_clarification",
        rank=1,
        prompt="Which of these feels closest to what you've been experiencing? (pick one, or describe it in your own words)",
        chip_options=["Sad", "Anxious", "Empty / numb", "Irritable", "Overwhelmed", "Hopeless", "Confused", "Other"],
        narrative_field="emotions",
    ),
    QuestionBankItem(
        id="coping_behaviors",
        rank=2,
        prompt="What do you usually do to get through days like this?",
        chip_options=["Talk to someone", "Distract myself", "Sleep it off", "Avoid it", "Not sure"],
        narrative_field="coping_signals",
    ),
    QuestionBankItem(
        id="support_person",
        rank=3,
        prompt="Is there anyone you feel even a little comfortable talking to about this?",
        chip_options=["Yes, definitely", "Somewhat", "Not really", "No one right now"],
        narrative_field="support_signals",
    ),
    QuestionBankItem(
        id="distress_intensity",
        rank=4,
        prompt="How intense would you say this feels right now?",
        chip_options=["Mild", "Moderate", "Very intense", "Overwhelming"],
    ),
    QuestionBankItem(
        id="passive_si",
        rank=5,
        prompt=PASSIVE_SI_PROMPT,
        chip_options=PASSIVE_SI_OPTIONS,
        is_risk_gating=True,
        asked_verbatim=True,
    ),
    QuestionBankItem(
        id="active_si",
        rank=6,
        prompt=ACTIVE_SI_PROMPT,
        chip_options=ACTIVE_SI_OPTIONS,
        is_risk_gating=True,
        asked_verbatim=True,
        condition_on="passive_si_not_no",
    ),
    QuestionBankItem(
        id="feels_safe",
        rank=7,
        prompt="Right now, do you feel physically safe?",
        chip_options=["Yes", "I'm not fully sure", "No, I feel unsafe"],
        is_risk_gating=True,
    ),
    QuestionBankItem(
        id="meaning_readiness",
        rank=8,
        prompt="If things felt even a little better, what would be different?",
        chip_options=["Less pressure", "Better sleep", "Feeling less alone", "Not sure yet"],
    ),
    QuestionBankItem(
        id="open_to_support",
        rank=9,
        prompt="Would you be open to getting some support with this -- from someone you trust, or a professional?",
        chip_options=["Yes, I want help", "I'm open to suggestions", "Maybe later", "Not right now"],
        narrative_field=None,
    ),
]

MAX_STRUCTURED_QUESTIONS = 15


def get_next_question(answered_ids: set[str], narrative_analysis_answered: set[str], prior_answers: dict[str, str]) -> QuestionBankItem | None:
    """Returns the next unanswered question in fixed safety-first rank order,
    skipping ones the narrative already answered and honoring conditional
    questions (e.g. active_si only follows a non-"No" passive_si)."""
    for item in sorted(QUESTION_BANK, key=lambda q: q.rank):
        if item.id in answered_ids:
            continue
        if item.narrative_field and item.narrative_field in narrative_analysis_answered:
            continue
        if item.condition_on == "passive_si_not_no":
            passive_answer = prior_answers.get("passive_si")
            if passive_answer is None or passive_answer == "No":
                continue
        return item
    return None
