"""Phase 1.5: run the completed narrative through the in-house nuance
encoder (Stage 3: in-house model runs first) and the LLM extraction prompt,
producing the internal NarrativeAnalysis object that drives Phase 2's
adaptive question selection. Never shown to the user.
"""
from __future__ import annotations

import json
import logging

from app.llm.client import complete_json
from app.llm.prompts.loader import render
from app.models.schemas import NarrativeAnalysis

logger = logging.getLogger(__name__)

_EMPTY_ANSWERED = {
    "emotional_clarification": False,
    "coping_behaviors": False,
    "support_person": False,
    "distress_intensity": False,
    "feels_safe": False,
    "meaning_readiness": False,
    "open_to_support": False,
}


def run_inhouse_nuance_pass(turns: list[str]) -> dict | None:
    """Runs the in-house model's nuance_encoder over the narrative turns for
    background signal/logging only -- per Stage 3, this never drives the
    narrative_analysis object shown to the rest of the pipeline; the LLM's
    extraction is the source of truth this phase."""
    try:
        from inhouse_model.model import InHouseModel

        model = InHouseModel()
        risk_level, sub_labels, confidence = model.predict(turns)
        return {"risk_level": risk_level, "sub_labels": sub_labels, "confidence": confidence}
    except Exception as exc:
        logger.warning("in-house nuance pass unavailable: %s", exc)
        return None


async def analyze_narrative(narrative_text: str, turns: list[str]) -> tuple[NarrativeAnalysis, dict | None]:
    inhouse_signal = run_inhouse_nuance_pass(turns)

    system = render("narrative_analysis", narrative_text=narrative_text)
    try:
        data = await complete_json(system=system, user="Analyze this narrative.")
        analysis = NarrativeAnalysis(
            key_events=data.get("key_events", []),
            timeline=data.get("timeline"),
            emotions=data.get("emotions", []),
            coping_signals=data.get("coping_signals", []),
            support_signals=data.get("support_signals", []),
            functional_impact=data.get("functional_impact"),
            risk_language_flags=data.get("risk_language_flags", []),
            answered_questions=data.get("answered_questions", dict(_EMPTY_ANSWERED)),
        )
    except Exception as exc:
        logger.warning("narrative analysis LLM call failed, treating story as unanalyzed: %s", exc)
        analysis = NarrativeAnalysis(answered_questions=dict(_EMPTY_ANSWERED))

    return analysis, inhouse_signal
