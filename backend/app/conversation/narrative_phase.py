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
from app.nlp_engine import NLPEngine


async def process_narrative_turn(
    user_text: str,
    has_prior_substantive_turn: bool,
    completion_check_already_asked: bool,
    explicit_done: bool = False,
) -> NarrativePhaseResult:
    # 2-Tier Crisis Override
    is_crisis, _rationale = NLPEngine.check_crisis_override(user_text)
    if is_crisis or narrative_contains_unsafe_language(user_text):
        return NarrativePhaseResult(action=NarrativeAction.emergency)

    if explicit_done or _is_explicit_completion(user_text):
        return NarrativePhaseResult(action=NarrativeAction.move_to_analysis)

    if completion_check_already_asked:
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
    try:
        grok_text = await complete(system=render("listening_ack", user_text=user_text), user="Acknowledge briefly.")
        NLPEngine.log_teacher_pair(user_text, {"type": "listening_ack", "output": grok_text})
        return grok_text
    except Exception:
        return NLPEngine.generate_offline_reflection([user_text])


def orientation_message() -> str:
    return render("orientation")
