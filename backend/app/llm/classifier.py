from __future__ import annotations

from app.llm.client import complete_json
from app.llm.prompts.loader import render
from app.models.schemas import RiskAssessment, RiskLevel, SubLabels

VALID_RISK_LEVELS: set[str] = {
    "N/A-Feeling Fine", "Low", "Mild", "Moderate", "Severe", "Emergency",
}


async def classify_risk(session_context: str) -> tuple[RiskAssessment, bool]:
    """Calls the LLM to produce the risk classification. Returns
    (RiskAssessment, is_emergency). Malformed/unparseable JSON or an
    out-of-rubric risk_level is treated as a hard escalation to Severe with
    zero confidence, per the spec's "low confidence escalates, never
    de-escalates" rule -- never silently defaults to a lower level.
    """
    system = render("risk_classifier", session_context=session_context)
    try:
        data = await complete_json(system=system, user="Classify this session.")
        risk_level = data["risk_level"]
        if risk_level not in VALID_RISK_LEVELS:
            raise ValueError(f"invalid risk_level from LLM: {risk_level}")
        assessment = RiskAssessment(
            risk_level=risk_level,
            sub_labels=SubLabels(**data["sub_labels"]),
            confidence=float(data.get("confidence", 0.5)),
            source="llm",
            rationale=data.get("rationale"),
        )
        is_emergency = bool(data.get("is_emergency", risk_level == "Emergency"))
        return assessment, is_emergency
    except Exception as exc:
        fallback = RiskAssessment(
            risk_level="Severe",
            sub_labels=SubLabels(
                suicide_risk="unknown (LLM response unparseable)",
                functional_impairment="marked",
                support_level="unknown",
                help_readiness="unknown",
            ),
            confidence=0.0,
            source="llm",
            rationale=f"Escalated conservatively: LLM classification failed to parse ({exc}).",
        )
        return fallback, False
