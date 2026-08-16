from dataclasses import dataclass
from pathlib import Path

from .config import Settings


class NLPModelUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ExtractedFeatures:
    emotions: list[str]
    category: str | None
    narrative_depth: float
    trigger: str | None
    duration: str | None
    impact: str | None
    coping: str | None
    support: str | None
    passive_self_harm_signal: bool = False
    active_self_harm_signal: bool = False

    def as_dict(self) -> dict:
        return {
            "primary_emotions": self.emotions,
            "problem_category": self.category,
            "narrative_depth": self.narrative_depth,
            "trigger": self.trigger,
            "duration": self.duration,
            "impact": self.impact,
            "coping": self.coping,
            "support": self.support,
        }


class NeuralFeatureExtractor:
    """Runs trained transformer models for emotion, topic and BIO slot extraction."""

    def __init__(self, model_root: Path):
        self.ready = True
        task_dirs = {task: model_root / task for task in ("emotion", "topic", "slots")}
        missing = [str(path) for path in task_dirs.values() if not path.exists()]
        if missing:
            raise NLPModelUnavailable(f"Missing trained NLP model(s): {', '.join(missing)}")
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoModelForTokenClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise NLPModelUnavailable("PyTorch and Transformers are required") from exc

        self.torch = torch
        required = {
            "emotion": AutoModelForSequenceClassification,
            "topic": AutoModelForSequenceClassification,
            "slots": AutoModelForTokenClassification,
        }
        self.models = {}
        for task, model_class in required.items():
            task_dir = task_dirs[task]
            tokenizer = AutoTokenizer.from_pretrained(task_dir, use_fast=True)
            model = model_class.from_pretrained(task_dir)
            model.eval()
            labels = [model.config.id2label[index] for index in range(model.config.num_labels)]
            self.models[task] = (tokenizer, model, labels)

    def _sequence_prediction(self, task: str, text: str) -> list[str]:
        tokenizer, model, labels = self.models[task]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with self.torch.no_grad():
            logits = model(**inputs).logits.squeeze(0)
        if task == "topic":
            return [labels[int(self.torch.softmax(logits, dim=0).argmax().item())]]
        probabilities = self.torch.sigmoid(logits)
        selected = [
            label for label, probability in zip(labels, probabilities)
            if float(probability) >= 0.5
        ]
        return selected or [labels[int(probabilities.argmax().item())]]

    def _slot_spans(self, text: str) -> dict[str, str]:
        tokenizer, model, labels = self.models["slots"]
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping").squeeze(0).tolist()
        with self.torch.no_grad():
            predicted_ids = model(**encoded).logits.argmax(dim=-1).squeeze(0).tolist()

        pieces: dict[str, list[tuple[int, int]]] = {}
        active_slot = None
        for predicted_id, (start, end) in zip(predicted_ids, offsets):
            if start == end:
                continue
            label = labels[predicted_id]
            if label == "O":
                active_slot = None
                continue
            prefix, slot = label.split("-", 1)
            if prefix == "B" or slot != active_slot:
                pieces.setdefault(slot, []).append((start, end))
            else:
                previous_start, _ = pieces[slot][-1]
                pieces[slot][-1] = (previous_start, end)
            active_slot = slot
        return {
            slot: " | ".join(text[start:end].strip() for start, end in spans if text[start:end].strip())
            for slot, spans in pieces.items()
        }

    def extract(self, text: str) -> ExtractedFeatures:
        emotions = self._sequence_prediction("emotion", text)
        topic = self._sequence_prediction("topic", text)[0]
        spans = self._slot_spans(text)
        return ExtractedFeatures(
            emotions=emotions,
            category=topic,
            narrative_depth=min(1.0, round(len(text.split()) / 50.0, 2)),
            trigger=spans.get("trigger"),
            duration=spans.get("duration"),
            impact=spans.get("impact"),
            coping=spans.get("coping"),
            support=spans.get("support"),
        )


class UnavailableFeatureExtractor:
    def __init__(self, reason: str):
        self.reason = reason
        self.ready = False

    def extract(self, text: str) -> ExtractedFeatures:
        raise NLPModelUnavailable(self.reason)


def build_feature_extractor(settings: Settings):
    try:
        return NeuralFeatureExtractor(settings.local_nlp_model_dir)
    except Exception as exc:
        return UnavailableFeatureExtractor(str(exc))
