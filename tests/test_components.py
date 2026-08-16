import unittest

from mindtriage.dialogue import DialogueStateTracker
from mindtriage.nlp import ExtractedFeatures, NLPModelUnavailable, UnavailableFeatureExtractor
from mindtriage.risk import RISK_LEVELS, highest_risk
from mindtriage.safety import SafetyEngine


class ComponentTests(unittest.TestCase):
    def setUp(self):
        self.safety = SafetyEngine()

    def features(self):
        return ExtractedFeatures(
            emotions=["neutral"], category="general", narrative_depth=0.2,
            trigger=None, duration=None, impact=None, coping=None, support=None,
        )

    def test_missing_neural_model_fails_closed(self):
        with self.assertRaises(NLPModelUnavailable):
            UnavailableFeatureExtractor("weights missing").extract("hello")

    def test_dialogue_slot_priority_and_cooldown(self):
        tracker = DialogueStateTracker()
        slots = {
            "trigger": "work",
            "duration": None,
            "impact": None,
            "emotions": [],
            "coping": None,
            "support": None,
        }
        self.assertEqual(tracker.decide(slots, []).target_slot, "duration")
        self.assertEqual(tracker.decide(slots, ["duration"]).target_slot, "impact")

    def test_safety_active_and_passive(self):
        active_text = "I intend to hurt myself right now"
        active = self.safety.evaluate(
            active_text, RISK_LEVELS[0], self.features()
        )
        self.assertTrue(active.is_emergency)
        passive_text = "I wish I was dead"
        passive = self.safety.evaluate(
            passive_text, RISK_LEVELS[0], self.features()
        )
        self.assertFalse(passive.is_emergency)
        self.assertEqual(passive.final_risk, RISK_LEVELS[3])

    def test_highest_risk(self):
        self.assertEqual(highest_risk(RISK_LEVELS[3], RISK_LEVELS[1]), RISK_LEVELS[3])


if __name__ == "__main__":
    unittest.main()
