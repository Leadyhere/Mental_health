from mindtriage.nlp import ExtractedFeatures
from mindtriage.risk import RISK_LEVELS, RiskPrediction


class FakeFeatureExtractor:
    """Deterministic test double; never used by the production application."""

    ready = True

    def extract(self, text: str) -> ExtractedFeatures:
        lowered = text.lower()
        emotions = [name for name in ("anxious", "stressed", "sad") if name in lowered]
        return ExtractedFeatures(
            emotions=emotions or ["neutral"],
            category="study_or_work_stress" if "work" in lowered or "college" in lowered else "general",
            narrative_depth=min(1.0, len(text.split()) / 50),
            trigger="work" if "work" in lowered else None,
            duration="two weeks" if "two weeks" in lowered else None,
            impact="sleep" if "sleep" in lowered else None,
            coping=None,
            support=None,
        )


class FakeRiskClassifier:
    """Stable test prediction provider; production uses the transformer classifier."""

    ready = True
    model_version = "test-risk-model"

    def predict(self, text, features, history):
        lowered = text.lower()
        if "wish i was dead" in lowered:
            index = 3
        elif "kill myself" in lowered or "hurt myself" in lowered:
            index = 4
        else:
            index = 0
        probabilities = [0.0] * len(RISK_LEVELS)
        probabilities[index] = 1.0
        return RiskPrediction(RISK_LEVELS[index], probabilities, self.model_version)


class FakeConversationGenerator:
    ready = True
    source = "test-neural-dialogue-model"

    def generate(self, user_text, features, decision, history):
        return "It sounds like this has been difficult. What feels most important right now?"
