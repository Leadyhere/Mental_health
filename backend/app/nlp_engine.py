"""Unified NLP Engine for Mental Health Conversational Triage Platform.
Consolidates Tier 1/2 crisis evaluation, dynamic reflective NLG, free-text slot filling,
teacher-student logging, offline fallback synthesis, and edge-case hardening into a single module.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.models.schemas import NarrativeAnalysis, RiskAssessment, SubLabels

logger = logging.getLogger(__name__)

# --- Directory setup ---
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEACHER_LOGS_DIR = REPO_ROOT / "data" / "teacher_logs"
TEACHER_LOGS_PATH = TEACHER_LOGS_DIR / "dataset.jsonl"

# --- Tier 1 Crisis Patterns & Keywords ---
CRISIS_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "end my life", "take my life",
    "overdose", "overdosing", "cut my wrist", "cutting myself", "hang myself",
    "poison", "want to die", "die", "hanging myself"
]

CRISIS_FUZZY_PATTERNS = [
    r"\bsuic[ia]?d[ee]?\b",
    r"\bposion[s]?\b",
    r"\bkill\s*my\s*self\b",
    r"\boverdos(e|ing)?\b",
    r"\bdieing\b",
    r"\bhang\s*my\s*self\b",
    r"\bcut\s*my\s*self\b",
    r"\bpeison\b",
    r"\bkll\s*myself\b",
]

_CRISIS_REGEX = re.compile("|".join([r"\b" + re.escape(k) + r"\b" for k in CRISIS_KEYWORDS] + CRISIS_FUZZY_PATTERNS), re.IGNORECASE)

# --- Negation & Safety Overrides ---
NEGATION_PATTERNS = [
    r"\bnot\s+(suicidal|trying to die|going to kill|wanting to die)\b",
    r"\bdon'?t\s+want\s+to\s+(die|kill myself|end my life)\b",
    r"\bno\s+suicid(al|e)\b",
    r"\bnever\s+(suicidal|thought of suicide)\b",
    r"\bnot\s+looking\s+to\s+(end|take)\s+my\s+life\b",
    r"\bam\s+not\s+suicidal\b",
]
_NEGATION_REGEX = re.compile("|".join(NEGATION_PATTERNS), re.IGNORECASE)

# --- Tier 2 Implicit Despair Signals ---
IMPLICIT_DESPAIR_PATTERNS = [
    r"\beveryone\s+would\s+be\s+better\s+off\s+without\s+me\b",
    r"\bno\s+reason\s+to\s+live\b",
    r"\bcan'?t\s+go\s+on\s+any\s*more\b",
    r"\bworld\s+would\s+be\s+better\s+if\s+i\s+(wasn'?t|were not)\s+here\b",
    r"\btired\s+of\s+(living|existing|life)\b",
    r"\bwish\s+i\s+was\s+never\s+born\b",
    r"\bno\s+way\s+out\s+of\s+this\s+pain\b",
]
_DESPAIR_REGEX = re.compile("|".join(IMPLICIT_DESPAIR_PATTERNS), re.IGNORECASE)

# --- Gibberish & Short Text Patterns ---
_GIBBERISH_REGEX = re.compile(r"^(asdfghj|qwerty|zxcvbnm|12345|test|aaa+|zzz+)$", re.IGNORECASE)
_SHORT_TEXT_SET = {"ok", "okay", "idk", "hm", "hmm", "fine", "yes", "no", "meh", "k"}

# --- Jailbreak & Prompt Injection Patterns ---
JAILBREAK_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"system\s+prompt",
    r"you\s+are\s+now\s+dan",
    r"forget\s+all\s+rules",
    r"override\s+safety",
]
_JAILBREAK_REGEX = re.compile("|".join(JAILBREAK_PATTERNS), re.IGNORECASE)


def sanitize_input(text: str) -> str:
    """Sanitizes user input against prompt injection attempts."""
    if _JAILBREAK_REGEX.search(text):
        logger.warning("Sanitizing potential prompt injection attempt: %s", text)
        return _JAILBREAK_REGEX.sub("[sanitized]", text)
    return text


def Levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return Levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


class NLPEngine:
    """Unified NLP Engine providing 2-tier crisis detection, free-text slot filling,
    dynamic listening NLG, teacher logging, and offline fallback.
    """

    @staticmethod
    def is_gibberish_or_vague(text: str) -> bool:
        cleaned = text.strip()
        if len(cleaned) < 3 or cleaned.lower() in _SHORT_TEXT_SET:
            return True
        if _GIBBERISH_REGEX.match(cleaned):
            return True
        return False

    @staticmethod
    def check_crisis_override(user_text: str) -> tuple[bool, str]:
        """2-Tier Crisis Override with Sarcasm/Negation protection.
        Returns (is_crisis, rationale).
        """
        sanitized = sanitize_input(user_text)

        # Check explicit negation first: "I'm NOT suicidal"
        if _NEGATION_REGEX.search(sanitized):
            return False, "negation detected (explicit non-suicidal statement)"

        # Tier 1: Regex & Levenshtein Fuzzy Guard
        if _CRISIS_REGEX.search(sanitized):
            return True, "Tier 1: High-precision crisis keyword or typo match detected"

        # Tier 1 Fuzzy fallback for individual words
        words = re.findall(r"\b[a-zA-Z]{4,}\b", sanitized.lower())
        for word in words:
            for kw in ["suicide", "poison", "overdose", "hanging"]:
                if Levenshtein_distance(word, kw) <= 1:
                    return True, f"Tier 1: Fuzzy match '{word}' close to '{kw}'"

        # Tier 2: Implicit Despair & Suicidal Intent Analyzer
        if _DESPAIR_REGEX.search(sanitized):
            return True, "Tier 2: High despair / implicit suicidal intent score detected"

        return False, "safe"

    @staticmethod
    def extract_clinical_slots(text: str) -> dict[str, Any]:
        """Free-text slot filling to extract clinical indicators from unstructured text."""
        slots: dict[str, Any] = {}
        lower = text.lower()

        # Sleep quality
        if any(w in lower for w in ["insomnia", "can't sleep", "cant sleep", "barely sleep", "poor sleep"]):
            slots["sleep_quality"] = "poor"
        elif any(w in lower for w in ["sleep fine", "sleeping well", "good sleep", "8 hours"]):
            slots["sleep_quality"] = "good"

        # Support system
        if any(w in lower for w in ["alone", "no friends", "nobody to talk", "no support", "isolated"]):
            slots["support_system"] = "isolated"
        elif any(w in lower for w in ["family", "friend", "friends", "partner", "therapist", "counselor"]):
            slots["support_system"] = "supported"

        # Coping mechanisms
        if any(w in lower for w in ["alcohol", "drinking", "drugs", "substances", "smoke", "vape"]):
            slots["coping_mechanisms"] = "substances"
        elif any(w in lower for w in ["walk", "walking", "music", "journal", "exercise", "meditation"]):
            slots["coping_mechanisms"] = "healthy"

        # Distress intensity
        intensity_match = re.search(r"\b(10|[1-9])/10\b", lower)
        if intensity_match:
            slots["distress_intensity"] = intensity_match.group(0)
        elif any(w in lower for w in ["overwhelming", "unbearable", "extremely high", "severe"]):
            slots["distress_intensity"] = "high"
        elif any(w in lower for w in ["mild", "manageable", "low"]):
            slots["distress_intensity"] = "low"

        # Feels safe
        if any(w in lower for w in ["feel unsafe", "not safe", "unsafe"]):
            slots["feels_safe"] = "no"
        elif any(w in lower for w in ["feel safe", "i am safe", "im safe"]):
            slots["feels_safe"] = "yes"

        return slots

    @staticmethod
    def evaluate_narrative_completeness(turns: list[str], explicit_done: bool = False) -> tuple[bool, float]:
        """Evaluates whether narrative phase is complete based on explicit button or implicit score."""
        if explicit_done:
            return True, 1.0

        full_text = " ".join(turns)
        word_count = len(full_text.split())
        num_turns = len(turns)

        # Base score on length and turn count
        score = min(1.0, (word_count / 40.0) * 0.6 + (num_turns / 3.0) * 0.4)
        is_complete = num_turns >= 2 and word_count >= 30
        return is_complete, score

    @staticmethod
    def generate_offline_reflection(turns: list[str]) -> str:
        """Local CPU-based Dynamic Response Synthesizer (NLG Engine) when Grok LLM is offline."""
        combined = " ".join(turns).lower()
        themes = []
        if any(w in combined for w in ["sad", "grief", "crying", "depressed", "loss", "pain"]):
            themes.append("the deep emotional pain you're carrying")
        if any(w in combined for w in ["stress", "overwhelmed", "work", "school", "pressure", "exhausted"]):
            themes.append("how overwhelming things feel right now")
        if any(w in combined for w in ["alone", "lonely", "isolated", "nobody"]):
            themes.append("the sense of isolation you've been experiencing")

        if themes:
            theme_str = " and ".join(themes)
            return (
                f"Thank you for sharing that with me. I hear {theme_str}. "
                "I'm here to listen and support you. Is there anything else you'd like to express?"
            )
        return (
            "Thank you for sharing your experience. I'm listening closely to everything you're saying. "
            "Please feel free to share more, or click 'Done sharing' when you're ready."
        )

    @staticmethod
    def log_teacher_pair(user_input: str, grok_output: dict[str, Any]) -> None:
        """Logs (User Input) -> (Grok Analysis) into data/teacher_logs/dataset.jsonl."""
        try:
            TEACHER_LOGS_DIR.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_input": user_input,
                "grok_output": grok_output,
            }
            with open(TEACHER_LOGS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning("Failed to log teacher pair: %s", exc)


class MultiTurnSafetyMemory:
    """Preserves historical high-risk flags across conversation turns."""
    def __init__(self):
        self.highest_risk_level: str = "N/A-Feeling Fine"
        self.risk_flags: set[str] = set()

    def update(self, risk_level: str, flags: list[str] | None = None):
        risk_order = ["N/A-Feeling Fine", "Low", "Mild", "Moderate", "Severe", "Emergency"]
        curr_idx = risk_order.index(self.highest_risk_level) if self.highest_risk_level in risk_order else 0
        new_idx = risk_order.index(risk_level) if risk_level in risk_order else 0

        if new_idx > curr_idx:
            self.highest_risk_level = risk_level

        if flags:
            self.risk_flags.update(flags)

    def apply_memory(self, assessment: RiskAssessment) -> RiskAssessment:
        risk_order = ["N/A-Feeling Fine", "Low", "Mild", "Moderate", "Severe", "Emergency"]
        curr_idx = risk_order.index(assessment.risk_level) if assessment.risk_level in risk_order else 0
        mem_idx = risk_order.index(self.highest_risk_level) if self.highest_risk_level in risk_order else 0

        if mem_idx > curr_idx:
            return assessment.model_copy(
                update={
                    "risk_level": self.highest_risk_level,
                    "rationale": f"{assessment.rationale or ''} [Retained historical high-risk memory: {self.highest_risk_level}]".strip(),
                }
            )
        return assessment
