import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from mindtriage.dialogue import DialogueStateTracker
from mindtriage.nlp import ExtractedFeatures, NLPModelUnavailable, UnavailableFeatureExtractor
from mindtriage.risk import RISK_LEVELS, highest_risk
from mindtriage.safety import SafetyEngine
from mindtriage.config import Settings
from mindtriage.sessions import MemorySessionStore, build_session_store
from mindtriage.training import TrainingMonitor
from mindtriage.generation import ConversationGenerator
from scripts.generate_dialogue_scenarios import post_message_with_retry


class FakeProgressBar:
    def __init__(self):
        self.postfix = None

    def set_postfix(self, **kwargs):
        self.postfix = kwargs


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

    def test_safety_phrase_matrix(self):
        cases = {
            "I want to kill myself": True,
            "I am going to hurt myself tonight": True,
            "I will end my life today": True,
            "I am thinking about harming myself": True,
            "I have a plan to die": True,
            "I was not suicidal before, but I have a plan to kill myself tonight": True,
            "I am not going to hurt myself": False,
            "I don't want to kill myself": False,
            "My friend told me to kill myself": False,
            "An article used the phrase kill myself": False,
            "I never plan to harm myself": False,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                decision = self.safety.evaluate(text, RISK_LEVELS[0], self.features())
                self.assertEqual(decision.is_emergency, expected)

    def test_highest_risk(self):
        self.assertEqual(highest_risk(RISK_LEVELS[3], RISK_LEVELS[1]), RISK_LEVELS[3])

    def test_unknown_risk_level_is_rejected(self):
        with self.assertRaises(ValueError):
            highest_risk(RISK_LEVELS[0], "unknown")

    def test_production_configuration_requires_secure_services(self):
        settings = Settings(
            environment="production",
            enable_redis=False,
            database_url="postgresql://example.invalid/mindtriage",
            analytics_hash_secret="x" * 32,
        )
        with self.assertRaisesRegex(RuntimeError, "Redis"):
            settings.validate()

    def test_unknown_environment_cannot_bypass_production_checks(self):
        with self.assertRaisesRegex(RuntimeError, "APP_ENV"):
            Settings(environment="prodution", enable_redis=False).validate()

    def test_groq_generation_uses_low_reasoning_effort(self):
        settings = Settings(groq_model="openai/gpt-oss-120b")
        generator = ConversationGenerator(settings)
        response = Mock()
        response.choices = [Mock(message=Mock(content="I hear you."))]
        generator.client = Mock()
        generator.client.chat.completions.create.return_value = response
        features = ExtractedFeatures(
            emotions=["sad"], category="general", narrative_depth=0.2,
            trigger=None, duration=None, impact=None, coping=None, support=None,
        )
        generator.generate(
            "I feel low", features, DialogueStateTracker().decide({}, []), [], "Mild", []
        )
        self.assertEqual(
            generator.client.chat.completions.create.call_args.kwargs["reasoning_effort"],
            "low",
        )

    @patch("scripts.generate_dialogue_scenarios.time.sleep")
    def test_generation_retries_transient_service_failure(self, sleep):
        unavailable = Mock(status_code=503)
        success = Mock(status_code=200)
        client = Mock()
        client.post.side_effect = [unavailable, success]

        response = post_message_with_retry(
            client, {"session_id": "session", "user_input": "hello"}, max_retries=2
        )

        self.assertIs(response, success)
        self.assertEqual(client.post.call_count, 2)
        sleep.assert_called_once_with(2)

    def test_deleted_memory_session_cannot_be_resurrected_by_a_late_save(self):
        store = MemorySessionStore(ttl_seconds=60)
        session = store.create(False, "en-IN")
        self.assertTrue(store.delete(session["session_id"]))
        session["turn_count"] = 1
        self.assertFalse(store.save(session))
        self.assertIsNone(store.get(session["session_id"]))

    def test_production_redis_failure_never_falls_back_to_memory(self):
        settings = Settings(
            environment="production",
            enable_redis=True,
            database_url="postgresql://example.invalid/mindtriage",
            analytics_hash_secret="x" * 32,
        )
        with patch("redis.Redis", side_effect=ConnectionError("unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Redis is required"):
                build_session_store(settings)

    def test_training_monitor_reports_percentage_and_writes_tensorboard_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = TrainingMonitor(
                Path(temp_dir), "test-task", epochs=2, steps_per_epoch=2
            )
            progress = FakeProgressBar()
            percentages = [
                monitor.log_batch(progress, 1, 1, 0.8),
                monitor.log_batch(progress, 1, 2, 0.6),
                monitor.log_batch(progress, 2, 1, 0.4),
                monitor.log_batch(progress, 2, 2, 0.2),
            ]
            monitor.log_epoch(1, {"train_loss": 0.7, "validation_loss": 0.5})
            monitor.finish(2)
            monitor.close()
            self.assertEqual(percentages, [25.0, 50.0, 75.0, 100.0])
            self.assertEqual(progress.postfix["overall"], "100.00%")
            self.assertTrue(list(Path(temp_dir).glob("events.out.tfevents.*")))


if __name__ == "__main__":
    unittest.main()
