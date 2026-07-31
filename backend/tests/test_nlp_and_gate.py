"""Pytest suite for NLP Engine, 2-Tier Crisis Override, Sarcasm/Negation, Slot Filling, and Fallback NLG.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.schemas import RiskAssessment, SubLabels
from app.nlp_engine import MultiTurnSafetyMemory, NLPEngine


def test_tier1_crisis_exact_and_fuzzy():
    # Exact keyword match
    is_crisis, rationale = NLPEngine.check_crisis_override("I want to commit suicide today")
    assert is_crisis is True
    assert "Tier 1" in rationale

    # Fuzzy typo match: posion, kill my self
    is_crisis_typo, rationale_typo = NLPEngine.check_crisis_override("I took posion and kill my self")
    assert is_crisis_typo is True
    assert "Tier 1" in rationale_typo


def test_negation_protection():
    # Negation should prevent false alarm
    is_crisis, rationale = NLPEngine.check_crisis_override("I am NOT suicidal at all, just stressed")
    assert is_crisis is False
    assert "negation" in rationale

    is_crisis_2, _ = NLPEngine.check_crisis_override("No, I don't want to die")
    assert is_crisis_2 is False


def test_tier2_implicit_despair():
    is_crisis, rationale = NLPEngine.check_crisis_override("Everyone would be better off without me")
    assert is_crisis is True
    assert "Tier 2" in rationale


def test_clinical_slot_filling():
    slots = NLPEngine.extract_clinical_slots("I have severe insomnia, live alone with no friends, and my distress is 8/10")
    assert slots.get("sleep_quality") == "poor"
    assert slots.get("support_system") == "isolated"
    assert slots.get("distress_intensity") == "8/10"


def test_offline_fallback_nlg():
    turns = ["I have been feeling so sad and overwhelmed with work recently"]
    reflection = NLPEngine.generate_offline_reflection(turns)
    assert "deep emotional pain" in reflection or "overwhelming" in reflection
    assert len(reflection) > 20


def test_multi_turn_safety_memory():
    mem = MultiTurnSafetyMemory()
    mem.update("Severe", ["si_flag"])

    curr_assessment = RiskAssessment(
        risk_level="Low",
        sub_labels=SubLabels(
            suicide_risk="none",
            functional_impairment="none",
            support_level="good",
            help_readiness="high",
        ),
        confidence=0.9,
        source="llm",
        rationale="User says fine now",
    )

    retained = mem.apply_memory(curr_assessment)
    assert retained.risk_level == "Severe"
    assert "Retained historical high-risk memory" in (retained.rationale or "")


if __name__ == "__main__":
    test_tier1_crisis_exact_and_fuzzy()
    test_negation_protection()
    test_tier2_implicit_despair()
    test_clinical_slot_filling()
    test_offline_fallback_nlg()
    test_multi_turn_safety_memory()
    print("ALL NLP AND SAFETY GATE TESTS PASSED 100%!")
