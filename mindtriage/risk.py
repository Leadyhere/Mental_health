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
        return RISK_LEVELS.index(level)
    except ValueError as exc:
        raise ValueError(f"Unknown risk level: {level}") from exc


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
        self.validated = False
        self.validation_status = "unavailable"
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
            expected_labels = [level.split(" - ", 1)[1].split(" (", 1)[0] for level in RISK_LEVELS]
            actual_labels = [
                self.model.config.id2label[index]
                for index in range(self.model.config.num_labels)
            ]
            if actual_labels != expected_labels:
                raise ValueError(
                    f"Risk label order mismatch: expected {expected_labels}, got {actual_labels}"
                )
            metadata_path = model_dir / "model_metadata.json"
            if metadata_path.exists():
                self.model_version = json.loads(metadata_path.read_text())["model_version"]
            else:
                self.model_version = model_dir.name
            metrics_path = model_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.validated = (
                metrics.get("data_quality", {}).get("cross_split_exact_text_overlap") == 0
            )
            self.validation_status = (
                "deduplicated-questionnaire-holdout"
                if self.validated
                else "legacy-metrics-retrain-required"
            )
        except Exception as exc:
            print(f"[WARNING] Local risk model could not be loaded: {exc}")
            self.model = None
            self.tokenizer = None
            self.validated = False
            self.validation_status = "unavailable"

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
