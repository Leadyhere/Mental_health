"""Generate project-specific synthetic dialogue turns by replaying scenario cards."""

import argparse
import itertools
import json
import random
import sys
import tempfile
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mindtriage.api import create_app
from mindtriage.config import Settings


TOPICS = [
    "work deadlines", "college exams", "family expectations", "a relationship conflict",
    "money pressure", "feeling isolated", "health worries", "grief after a loss",
]
EMOTIONS = ["anxious", "sad", "overwhelmed", "irritable", "lonely", "numb", "tired"]
DURATIONS = ["for three days", "for two weeks", "for about a month", "for several months"]
IMPACTS = [
    "My sleep is getting worse.",
    "It is becoming hard to focus on work or studies.",
    "I am avoiding people and normal activities.",
    "I can still manage my routine, but it takes more effort.",
]
COPING = [
    "I usually talk to a friend.",
    "I distract myself by scrolling on my phone.",
    "I go for a walk when I can.",
    "I mostly push through and ignore it.",
]
SUPPORT = [
    "I have someone I trust.",
    "I am not sure who I can speak to.",
    "My family is nearby but I do not feel understood.",
    "I would consider speaking with a counsellor.",
]

RISK_TURNS = {
    "Low": ["I am mainly checking in and I feel safe."],
    "Mild": ["It is uncomfortable, but I am still functioning and I feel safe."],
    "Moderate": ["It is affecting several parts of my daily routine, but I feel safe."],
    "Severe": ["Sometimes I wish I was dead, but I do not have a plan to hurt myself."],
    "Emergency": ["I want to hurt myself right now."],
}


def scenario_stream(seed: int):
    combinations = list(itertools.product(TOPICS, EMOTIONS, DURATIONS, IMPACTS, COPING, SUPPORT))
    random.Random(seed).shuffle(combinations)
    risk_levels = list(RISK_TURNS)
    for index, values in enumerate(itertools.cycle(combinations)):
        topic, emotion, duration, impact, coping, support = values
        risk = risk_levels[index % len(risk_levels)]
        first = f"I have been feeling {emotion} because of {topic} {duration}."
        yield risk, [first, impact, coping, support, RISK_TURNS[risk][0]]


def generate(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        settings = Settings(
            enable_local_dialogue_model=False,
            enable_redis=False,
            database_url=f"sqlite:///{(temp / 'analytics.db').as_posix()}",
            analytics_hash_secret="synthetic-generation-only",
            dataset_path=temp / "unused.jsonl",
        )
        client = TestClient(create_app(settings))
        records = []
        for scenario_index, (expected_risk, turns) in zip(
            range(args.conversations), scenario_stream(args.seed)
        ):
            session = client.post(
                "/chat/start", json={"training_consent": False, "locale": "en-IN"}
            ).json()
            conversation_id = f"synthetic_{args.seed}_{scenario_index}_{uuid.uuid4().hex[:8]}"
            for turn_number, user_text in enumerate(turns, start=1):
                response = client.post(
                    "/chat/message",
                    json={"session_id": session["session_id"], "user_input": user_text},
                )
                response.raise_for_status()
                body = response.json()
                records.append(
                    {
                        "schema_version": "2.0",
                        "data_origin": "groq_synthetic_scenario_replay",
                        "conversation_id": conversation_id,
                        "turn_number": turn_number,
                        "expected_risk": expected_risk,
                        "observed_risk": body["risk_level"],
                        "user_text": user_text,
                        "assistant_text": body["bot_reply"],
                        "target_slot": body["target_slot"],
                        "conversation_model": body["model_source"],
                    }
                )
            client.post("/chat/end", json={"session_id": session["session_id"]})

    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "conversations": args.conversations,
        "turns": len(records),
        "output": str(output),
        "note": "Synthetic data must be reviewed before training and kept separate from holdout safety tests.",
    }
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=Path("data/groq_dialogues.jsonl")
    )
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
