"""Every assistant-bound message and every risk classification passes
through here before being shown to the user or acted on.

- `apply_confidence_escalation`: below-threshold classifications escalate
  one risk level. Never de-escalates.
- `filter_outbound_message`: rejects messages that minimize risk, offer
  medical/treatment instructions, or sound dismissive -- a fast regex
  pre-check plus an LLM safety-review pass.
"""
from __future__ import annotations

import re

from app.llm.client import complete_json
from app.llm.prompts.loader import render
from app.models.schemas import RiskAssessment, RiskLevel

RISK_ORDER: list[RiskLevel] = ["N/A-Feeling Fine", "Low", "Mild", "Moderate", "Severe", "Emergency"]
CONFIDENCE_ESCALATION_THRESHOLD = 0.55

_DISMISSIVE_OR_MEDICAL_PATTERNS = [
    r"\byou should take\b.*\b(mg|milligrams|dose)\b",
    r"\bI (diagnose|am diagnosing)\b",
    r"\byou have (depression|anxiety|bipolar|ptsd|a disorder)\b",
    r"\bit'?s not (that|really) (serious|bad|a big deal)\b",
    r"\bjust (relax|calm down|cheer up|get over it)\b",
    r"\bthat makes sense\b.*\b(hurt yourself|end your life|suicide)\b",
]
_QUICK_REJECT_RE = re.compile("|".join(_DISMISSIVE_OR_MEDICAL_PATTERNS), re.IGNORECASE)

_SAFE_FALLBACK_MESSAGE = (
    "I hear you, and I want to make sure I respond carefully here. "
    "Could you tell me a little more about what's been going on?"
)


def escalate_one_level(risk_level: RiskLevel) -> RiskLevel:
    idx = RISK_ORDER.index(risk_level)
    return RISK_ORDER[min(idx + 1, len(RISK_ORDER) - 1)]


def apply_confidence_escalation(assessment: RiskAssessment, threshold: float = CONFIDENCE_ESCALATION_THRESHOLD) -> RiskAssessment:
    if assessment.confidence >= threshold:
        return assessment
    escalated_level = escalate_one_level(assessment.risk_level)
    if escalated_level == assessment.risk_level:
        return assessment
    return assessment.model_copy(
        update={
            "risk_level": escalated_level,
            "rationale": f"{assessment.rationale or ''} [Escalated one level: confidence {assessment.confidence:.2f} below threshold.]".strip(),
        }
    )


async def check_message_safety(message_text: str) -> tuple[bool, str]:
    if _QUICK_REJECT_RE.search(message_text):
        return False, "matched a minimization/medical-instruction/dismissiveness pattern"
    system = render("safety_filter_check", message_text=message_text)
    try:
        result = await complete_json(system=system, user="Review this message.")
        return bool(result.get("safe", False)), result.get("reason", "")
    except Exception:
        # Unparseable safety-check response -- treat conservatively as unsafe.
        return False, "safety-check response was unparseable"


async def filter_outbound_message(message_text: str) -> str:
    is_safe, _reason = await check_message_safety(message_text)
    if is_safe:
        return message_text
    return _SAFE_FALLBACK_MESSAGE
