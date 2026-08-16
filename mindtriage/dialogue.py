from dataclasses import dataclass

from .nlp import ExtractedFeatures


@dataclass(frozen=True)
class DialogueDecision:
    target_slot: str
    policy: str


class DialogueStateTracker:
    slots = ("trigger", "duration", "impact", "coping", "support")
    weights = {
        "trigger": 0.35,
        "duration": 0.25,
        "impact": 0.20,
        "coping": 0.10,
        "support": 0.10,
    }
    policies = {
        "trigger": "Open narrative reflection",
        "duration": "Complex reflection + gentle temporal probe",
        "impact": "Validation + daily functioning probe",
        "coping": "Affirmation + coping exploration",
        "support": "Affirmation + social support probe",
        "general_reflection": "Summary reflection + user-led next step",
    }

    def update(self, slots: dict, features: ExtractedFeatures, text: str) -> dict:
        updated = dict(slots)
        if not updated.get("trigger") and (features.trigger or features.category):
            updated["trigger"] = features.trigger or features.category
        for key in ("duration", "impact", "coping", "support"):
            value = getattr(features, key)
            if value and not updated.get(key):
                updated[key] = value
        current_emotions = list(updated.get("emotions", []))
        for emotion in features.emotions:
            if emotion not in current_emotions:
                current_emotions.append(emotion)
        updated["emotions"] = current_emotions
        return updated

    def decide(self, slots: dict, turn_history: list[str]) -> DialogueDecision:
        recent = set(turn_history[-2:])
        candidates = [slot for slot in self.slots if not slots.get(slot) and slot not in recent]
        target = max(candidates, key=self.weights.get) if candidates else "general_reflection"
        return DialogueDecision(target, self.policies[target])
