"""Phase 0 (orientation) and Phase 1 (open narrative / listening) logic.

Listening mode never interrupts with structured questions and never asks
"why". It only: (a) emits minimal acknowledgment turns, (b) asks the single
ambiguous-completion check at most once, or (c) detects the story is done
and hands off to Phase 1.5 (narrative analysis) -- unless a safety override
fires, which bypasses all of this immediately.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from app.conversation.emergency_gate import narrative_contains_unsafe_language
from app.llm.client import complete
from app.llm.prompts.loader import render

_EXPLICIT_COMPLETION_RE = re.compile(
    r"\b(that'?s (it|all)|i'?m done|nothing else|not sure what else to (say|add))\b", re.IGNORECASE
)
_WHAT_NOW_RE = re.compile(r"\bwhat (now|next|do i do now)\b", re.IGNORECASE)
_SHORT_REPLY_MAX_CHARS = 20


class NarrativeAction(str, Enum):
    emergency = "emergency"
    ask_completion_check = "ask_completion_check"
    move_to_analysis = "move_to_analysis"
    ack = "ack"


@dataclass
class NarrativePhaseResult:
    action: NarrativeAction
    assistant_text: str | None = None


def _is_explicit_completion(text: str) -> bool:
    return bool(_EXPLICIT_COMPLETION_RE.search(text)) or bool(_WHAT_NOW_RE.search(text))


def _is_ambiguous_pause_signal(text: str, has_prior_substantive_turn: bool) -> bool:
    return has_prior_substantive_turn and len(text.strip()) <= _SHORT_REPLY_MAX_CHARS


async def process_narrative_turn(
    user_text: str,
    has_prior_substantive_turn: bool,
    completion_check_already_asked: bool,
) -> NarrativePhaseResult:
    # Safety override: never waits for the story to finish.
    if narrative_contains_unsafe_language(user_text):
        return NarrativePhaseResult(action=NarrativeAction.emergency)

    if _is_explicit_completion(user_text):
        return NarrativePhaseResult(action=NarrativeAction.move_to_analysis)

    if completion_check_already_asked:
        # The single check has already been asked once; treat any further
        # short/negative reply as completion, otherwise keep listening.
        if _is_ambiguous_pause_signal(user_text, has_prior_substantive_turn) or _is_explicit_completion(user_text):
            return NarrativePhaseResult(action=NarrativeAction.move_to_analysis)
        return NarrativePhaseResult(action=NarrativeAction.ack, assistant_text=await _generate_ack(user_text))

    if _is_ambiguous_pause_signal(user_text, has_prior_substantive_turn):
        return NarrativePhaseResult(
            action=NarrativeAction.ask_completion_check,
            assistant_text=render("completion_check"),
        )

    return NarrativePhaseResult(action=NarrativeAction.ack, assistant_text=await _generate_ack(user_text))


async def _generate_ack(user_text: str) -> str:
    # A listening-turn acknowledgment is low-stakes and must never crash the
    # conversation if the LLM is unreachable -- fall back to a static,
    # non-directive acknowledgment rather than propagating the error.
    try:
        return await complete(system=render("listening_ack", user_text=user_text), user="Acknowledge briefly.")
    except Exception:
        return "I hear you. Please go on whenever you're ready."


def orientation_message() -> str:
    return render("orientation")
