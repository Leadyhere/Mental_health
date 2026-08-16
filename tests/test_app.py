import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from mindtriage.api import create_app
from mindtriage.config import Settings
from tests.fakes import FakeConversationGenerator, FakeFeatureExtractor, FakeRiskClassifier


class ChatApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.dataset_path = root / "conversations.jsonl"
        self.database_path = root / "analytics.db"
        settings = Settings(
            enable_redis=False,
            database_url=f"sqlite:///{self.database_path.as_posix()}",
            analytics_hash_secret="test-secret",
            dataset_path=self.dataset_path,
            frontend_path=Path(__file__).resolve().parent.parent / "index.html",
        )
        self.app = create_app(
            settings,
            feature_extractor=FakeFeatureExtractor(),
            risk_model=FakeRiskClassifier(),
            conversation_model=FakeConversationGenerator(),
        )
        self.client = TestClient(self.app)

    def tearDown(self):
        self.temp_dir.cleanup()

    def start(self, consent=False):
        response = self.client.post(
            "/chat/start", json={"training_consent": consent, "locale": "en-IN"}
        )
        self.assertEqual(response.status_code, 200)
        return response.json()["session_id"]

    def message(self, session_id, text):
        return self.client.post(
            "/chat/message", json={"session_id": session_id, "user_input": text}
        )

    def test_health_and_frontend(self):
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["database"], "sqlite")
        frontend = self.client.get("/")
        self.assertEqual(frontend.status_code, 200)
        self.assertIn("MindTriage", frontend.text)

    def test_production_path_never_uses_rule_fallbacks(self):
        root = Path(self.temp_dir.name)
        strict_settings = Settings(
            groq_api_key="",
            enable_local_dialogue_model=False,
            enable_redis=False,
            local_nlp_model_dir=root / "missing-nlp",
            local_risk_model_dir=root / "missing-risk",
            database_url=f"sqlite:///{(root / 'strict.db').as_posix()}",
            analytics_hash_secret="test-secret",
            frontend_path=Path(__file__).resolve().parent.parent / "index.html",
        )
        strict_client = TestClient(create_app(strict_settings))
        self.assertEqual(strict_client.get("/health").json()["status"], "degraded")
        session_id = strict_client.post("/chat/start", json={}).json()["session_id"]
        response = strict_client.post(
            "/chat/message", json={"session_id": session_id, "user_input": "hello"}
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("NLP model", response.json()["detail"])

    def test_normal_conversation_and_summary(self):
        session_id = self.start()
        response = self.message(
            session_id, "Work has made me anxious for two weeks and it affects my sleep"
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("anxious", body["extracted_emotions"])
        self.assertFalse(body["is_emergency"])
        summary = self.client.get("/chat/summary", params={"session_id": session_id})
        self.assertEqual(summary.status_code, 200)
        self.assertIn("two weeks", summary.json()["report"])

    def test_no_dataset_without_consent(self):
        session_id = self.start(False)
        self.assertEqual(self.message(session_id, "I feel stressed").status_code, 200)
        self.assertFalse(self.dataset_path.exists())

    def test_consented_jsonl_is_redacted(self):
        session_id = self.start(True)
        response = self.message(
            session_id, "Email me at person@example.com or +91 99999 88888 about work"
        )
        self.assertEqual(response.status_code, 200)
        records = [
            json.loads(line)
            for line in self.dataset_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["schema_version"], "2.0")
        self.assertIn("[EMAIL]", records[0]["user_text"])
        self.assertIn("[PHONE]", records[0]["user_text"])
        self.assertNotIn("session_id", records[0])

    def test_emergency_bypasses_conversation_model(self):
        session_id = self.start()
        response = self.message(session_id, "I have a plan to kill myself tonight")
        body = response.json()
        self.assertTrue(body["is_emergency"])
        self.assertIn("Level 5", body["risk_level"])
        self.assertEqual(body["model_source"], "deterministic-safety-v2")
        self.assertIn("112", body["bot_reply"])

    def test_explicit_negation_does_not_trigger_emergency(self):
        session_id = self.start()
        response = self.message(session_id, "I am not going to hurt myself")
        self.assertFalse(response.json()["is_emergency"])

    def test_highest_risk_does_not_drop(self):
        session_id = self.start()
        first = self.message(session_id, "I wish I was dead").json()
        self.assertIn("Level 4", first["highest_risk_level"])
        second = self.message(session_id, "Work was okay today").json()
        self.assertIn("Level 4", second["highest_risk_level"])

    def test_analytics_contains_no_raw_text(self):
        session_id = self.start()
        secret_phrase = "private unique phrase 98765"
        self.message(session_id, secret_phrase)
        connection = sqlite3.connect(self.database_path)
        dump = "\n".join(connection.iterdump())
        connection.close()
        self.assertNotIn(secret_phrase, dump)

    def test_end_session_is_idempotent(self):
        session_id = self.start()
        first = self.client.post("/chat/end", json={"session_id": session_id})
        second = self.client.post("/chat/end", json={"session_id": session_id})
        self.assertEqual(first.json()["status"], "deleted")
        self.assertEqual(second.json()["status"], "already_deleted")
        self.assertEqual(
            self.client.get("/chat/summary", params={"session_id": session_id}).status_code,
            404,
        )


if __name__ == "__main__":
    unittest.main()
