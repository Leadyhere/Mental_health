import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path


class ConsentedDatasetLogger:
    schema_version = "2.0"

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    @staticmethod
    def redact(text: str) -> str:
        text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
        text = re.sub(r"(?<!\w)(?:\+?\d[\d\s()-]{7,}\d)(?!\w)", "[PHONE]", text)
        text = re.sub(r"https?://\S+", "[URL]", text)
        return text

    def log(
        self,
        conversation_id: str,
        turn_number: int,
        user_text: str,
        bot_reply: str,
        features: dict,
        risk_level: str,
        target_slot: str,
        policy: str,
        conversation_model: str,
    ) -> None:
        record = {
            "schema_version": self.schema_version,
            "conversation_id": conversation_id,
            "turn_number": turn_number,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_text": self.redact(user_text),
            "assistant_text": self.redact(bot_reply),
            "extracted_features": features,
            "risk_level": risk_level,
            "target_slot": target_slot,
            "mi_policy": policy,
            "conversation_model": conversation_model,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
