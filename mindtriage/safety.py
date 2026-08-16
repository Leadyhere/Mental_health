import re
from dataclasses import dataclass

from .nlp import ExtractedFeatures
from .risk import RISK_LEVELS, highest_risk


@dataclass(frozen=True)
class SafetyDecision:
    final_risk: str
    is_emergency: bool
    contacts: dict[str, str]
    reason_codes: list[str]


class SafetyEngine:
    """Independent emergency fail-safe; it is not the normal risk classifier."""

    active_patterns = (
        re.compile(r"\b(?:i\s+)?(?:want|plan|going|intend)\s+to\s+(?:kill|hurt|harm)\s+myself\b", re.I),
        re.compile(r"\b(?:kill myself|end my life|take my life)\b", re.I),
        re.compile(r"\bi have (?:a )?plan to (?:die|kill myself|hurt myself)\b", re.I),
    )
    passive_patterns = (
        re.compile(r"\b(?:wish i (?:was|were) dead|do not want to wake up|don't want to wake up)\b", re.I),
        re.compile(r"\b(?:everyone would be better off without me|want to disappear)\b", re.I),
    )
    immediate_patterns = (
        re.compile(r"\b(?:right now|tonight|today)\b", re.I),
        re.compile(r"\b(?:already took (?:the )?pills|have (?:the )?pills|have a weapon)\b", re.I),
    )
    negated_patterns = (
        re.compile(r"\b(?:not|never) (?:suicidal|going to hurt myself|going to kill myself)\b", re.I),
        re.compile(r"\bno (?:thoughts|plan|intention) (?:of|to) (?:self harm|suicide|hurt myself)\b", re.I),
    )

    contacts = {
        "Emergency services": "112",
        "Tele-MANAS": "14416 / 1800-891-4416",
    }

    def evaluate(
        self, text: str, predicted_risk: str, features: ExtractedFeatures
    ) -> SafetyDecision:
        if any(pattern.search(text) for pattern in self.negated_patterns):
            return SafetyDecision(predicted_risk, False, {}, ["explicit_negation"])

        active = features.active_self_harm_signal or any(
            pattern.search(text) for pattern in self.active_patterns
        )
        passive = features.passive_self_harm_signal or any(
            pattern.search(text) for pattern in self.passive_patterns
        )
        immediate = any(pattern.search(text) for pattern in self.immediate_patterns)

        if active:
            reasons = ["active_self_harm_intent"]
            if immediate:
                reasons.append("immediacy_or_means")
            return SafetyDecision(RISK_LEVELS[4], True, self.contacts, reasons)
        if passive:
            return SafetyDecision(
                highest_risk(predicted_risk, RISK_LEVELS[3]),
                False,
                {},
                ["passive_self_harm_signal"],
            )
        return SafetyDecision(predicted_risk, False, {}, [])

    @staticmethod
    def emergency_response() -> str:
        return (
            "I'm concerned about your immediate safety. Please call emergency services at 112 "
            "or ask a trusted person to stay with you now. You can also call Tele-MANAS at "
            "14416 or 1800-891-4416. This chat cannot provide emergency help."
        )
