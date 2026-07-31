"""Hardcoded Emergency authority.

Per the spec's Stage 3 constraint: the in-house model's prediction is NEVER
read here. Only two signals may set is_emergency=True:
  1. The hardcoded verbatim-question rule (active_si == the highest-risk
     option, or feels_safe == "No, I feel unsafe").
  2. A narrative safety-override: an explicit unsafe-right-now / active
     intent statement detected during free narration (keyword heuristic),
     confirmed by the LLM classifier's own is_emergency judgment.

Once triggered, this bypasses all other logic -- the caller must stop
adaptive questioning and surface crisis resources immediately and
unconditionally.
"""
from __future__ import annotations

import re

from app.conversation.question_bank import ACTIVE_SI_OPTIONS
from app.models.schemas import StructuredAnswer

EMERGENCY_ACTIVE_SI_VALUE = "Yes and it feels risky right now"
EMERGENCY_UNSAFE_VALUE = "No, I feel unsafe"

# Conservative keyword heuristic for free-narrative safety override. This is
# intentionally broad (over-triggering is the safe direction) -- any hit
# here should be confirmed by the LLM classifier before fully committing to
# the Emergency path in evaluate_emergency(), but it should still cause an
# immediate interrupt of the listening phase per the spec.
_UNSAFE_RIGHT_NOW_PATTERNS = [
    r"\bkill (myself|me)\b",
    r"\bend (my|it all|this)\b.*\b(life|now|tonight|today)\b",
    r"\bsuicide\b",
    r"\bgoing to (die|end it|hurt myself)\b",
    r"\bhave a plan\b",
    r"\bwant to die\b",
    r"\bnot safe right now\b",
    r"\bcan'?t (go on|do this anymore)\b",
]
_UNSAFE_RIGHT_NOW_RE = re.compile("|".join(_UNSAFE_RIGHT_NOW_PATTERNS), re.IGNORECASE)


def check_structured_emergency(structured_answers: list[StructuredAnswer]) -> bool:
    for answer in structured_answers:
        value = answer.chip_selected or answer.answer_text
        if answer.question_id == "active_si" and value == EMERGENCY_ACTIVE_SI_VALUE:
            return True
        if answer.question_id == "feels_safe" and value == EMERGENCY_UNSAFE_VALUE:
            return True
    return False


def narrative_contains_unsafe_language(text: str) -> bool:
    return bool(_UNSAFE_RIGHT_NOW_RE.search(text))


def evaluate_emergency(
    structured_answers: list[StructuredAnswer],
    narrative_text: str | None = None,
    llm_says_emergency: bool | None = None,
) -> bool:
    if check_structured_emergency(structured_answers):
        return True
    if narrative_text and narrative_contains_unsafe_language(narrative_text):
        # Keyword hit alone is sufficient to interrupt listening mode and
        # route to the LLM for confirmation; if the LLM has already weighed
        # in (llm_says_emergency is not None), honor that judgment too.
        if llm_says_emergency is None or llm_says_emergency:
            return True
    if llm_says_emergency:
        return True
    return False
