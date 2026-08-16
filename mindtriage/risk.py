import json
from dataclasses import dataclass

from .config import Settings
from .nlp import ExtractedFeatures


RISK_LEVELS = (
    "Level 1 - Low (Watchful)",
    "Level 2 - Mild (Self-Help)",
    "Level 3 - Moderate (Structured)",
    "Level 4 - Severe (Priority)",
    "Level 5 - Emergency (Immediate)",
)


class RiskModelUnavailable(RuntimeError):
    pass


def risk_index(level: str) -> int:
    try:
        return next(index for index, value in enumerate(RISK_LEVELS) if value == level)
    except StopIteration:
        return 0


def highest_risk(*levels: str) -> str:
    return max(levels, key=risk_index)


@dataclass(frozen=True)
class RiskPrediction:
    level: str
    probabilities: list[float]
    source: str


class RiskClassifier:
    def __init__(self, settings: Settings):
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.model_version = "unavailable"
        if settings.local_risk_model_dir.exists():
            self._load(settings.local_risk_model_dir)

    def _load(self, model_dir) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.model.eval()
            metadata_path = model_dir / "model_metadata.json"
            if metadata_path.exists():
                self.model_version = json.loads(metadata_path.read_text())["model_version"]
            else:
                self.model_version = model_dir.name
        except Exception as exc:
            print(f"[WARNING] Local risk model could not be loaded: {exc}")
            self.model = None
            self.tokenizer = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def predict(self, text: str, features: ExtractedFeatures, history: list[dict]) -> RiskPrediction:
        if self.model is None or self.tokenizer is None:
            raise RiskModelUnavailable(
                "The trained MentalBERT risk classifier is unavailable. Run train_classifier.py."
            )
        conversation = " ".join(item["user"] for item in history[-4:] + [{"user": text}])
        fields = [f"Reason: {features.trigger or features.category or conversation}"]
        if features.duration:
            fields.append(f"Duration: {features.duration}")
        if features.emotions:
            fields.append(f"Emotions: {', '.join(features.emotions)}")
        if features.impact:
            fields.append(f"Daily impact: {features.impact}")
        if features.coping:
            fields.append(f"Coping: {features.coping}")
        if features.support:
            fields.append(f"Feels supported: {features.support}")
        fields.append(f"Conversation: {conversation}")
        model_input = " | ".join(fields)
        inputs = self.tokenizer(model_input, return_tensors="pt", truncation=True, max_length=256)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            probabilities = self.torch.softmax(logits, dim=1).squeeze().tolist()
        predicted = int(max(range(len(probabilities)), key=probabilities.__getitem__))
        return RiskPrediction(RISK_LEVELS[predicted], probabilities, self.model_version)
