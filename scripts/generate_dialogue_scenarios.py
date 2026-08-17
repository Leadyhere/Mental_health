"""Generate project-specific synthetic dialogue turns by replaying scenario cards."""

import argparse
import itertools
import json
import random
import sys
import tempfile
import time
import uuid
from collections import Counter
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


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def post_message_with_retry(client, payload: dict, max_retries: int):
    """Retry transient hosted-model failures without mutating a chat session."""
    for attempt in range(max_retries + 1):
        response = client.post("/chat/message", json=payload)
        if response.status_code not in RETRYABLE_STATUS_CODES:
            response.raise_for_status()
            return response
        if attempt == max_retries:
            response.raise_for_status()
        delay_seconds = min(60, 2 ** (attempt + 1))
        print(
            json.dumps(
                {
                    "event": "transient_generation_failure",
                    "status": response.status_code,
                    "retry": attempt + 1,
                    "max_retries": max_retries,
                    "retry_in_seconds": delay_seconds,
                }
            ),
            flush=True,
        )
        time.sleep(delay_seconds)
    raise RuntimeError("Unreachable retry state")


def load_checkpoint(path: Path) -> tuple[list[dict], set[int]]:
    if not path.exists():
        return [], set()
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[int, list[dict]] = {}
    for record in records:
        scenario_index = record.get("scenario_index")
        if not isinstance(scenario_index, int):
            raise ValueError(
                f"Checkpoint {path} was created by an older generator and cannot be resumed. "
                "Delete it and start again."
            )
        grouped.setdefault(scenario_index, []).append(record)
    completed = {
        index
        for index, conversation_records in grouped.items()
        if len(conversation_records) == len(RISK_TURNS) and {
            record["turn_number"] for record in conversation_records
        } == set(range(1, len(RISK_TURNS) + 1))
    }
    return records, completed


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
    if args.conversations <= 0:
        raise ValueError("--conversations must be greater than zero")
    if args.max_retries < 0:
        raise ValueError("--max-retries cannot be negative")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be greater than zero")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_name(f"{output.name}.partial")
    if checkpoint.exists() and not args.resume:
        raise FileExistsError(
            f"Checkpoint exists at {checkpoint}. Re-run with --resume to continue it, "
            "or delete that checkpoint to start over."
        )
    records, completed_scenarios = load_checkpoint(checkpoint) if args.resume else ([], set())
    if completed_scenarios:
        print(
            json.dumps(
                {
                    "event": "resuming_generation",
                    "completed_conversations": len(completed_scenarios),
                    "requested_conversations": args.conversations,
                }
            ),
            flush=True,
        )
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
        # ``expected_risk`` is a scenario-card target, not a clinical label.  The
        # label that conditions dialogue training must be the risk produced by the
        # same NLP + MentalBERT + safety pipeline used at runtime.
        risk_mismatches = []
        for scenario_index, (expected_risk, turns) in zip(
            range(args.conversations), scenario_stream(args.seed)
        ):
            if scenario_index in completed_scenarios:
                continue
            session = client.post(
                "/chat/start", json={"training_consent": False, "locale": "en-IN"}
            ).json()
            conversation_id = f"synthetic_{args.seed}_{scenario_index}_{uuid.uuid4().hex[:8]}"
            conversation_records = []
            for turn_number, user_text in enumerate(turns, start=1):
                response = post_message_with_retry(
                    client,
                    {"session_id": session["session_id"], "user_input": user_text},
                    args.max_retries,
                )
                body = response.json()
                turn_expected_risk = expected_risk if turn_number == len(turns) else None
                if turn_expected_risk and turn_expected_risk not in body["risk_level"]:
                    risk_mismatches.append(
                        {
                            "conversation": scenario_index,
                            "turn": turn_number,
                            "expected": turn_expected_risk,
                            "observed": body["risk_level"],
                        }
                    )
                conversation_records.append(
                    {
                        "schema_version": "2.0",
                        "data_origin": "groq_synthetic_scenario_replay",
                        "conversation_id": conversation_id,
                        "scenario_index": scenario_index,
                        "turn_number": turn_number,
                        "expected_risk": turn_expected_risk,
                        "observed_risk": body["risk_level"],
                        "user_text": user_text,
                        "assistant_text": body["bot_reply"],
                        "target_slot": body["target_slot"],
                        "conversation_model": body["model_source"],
                    }
                )
            client.post("/chat/end", json={"session_id": session["session_id"]})
            with checkpoint.open("a", encoding="utf-8") as handle:
                for record in conversation_records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.extend(conversation_records)
            if (scenario_index + 1) % args.progress_every == 0 or scenario_index + 1 == args.conversations:
                print(
                    json.dumps(
                        {
                            "event": "generation_progress",
                            "completed_conversations": scenario_index + 1,
                            "requested_conversations": args.conversations,
                            "completion_percent": round(
                                (scenario_index + 1) * 100 / args.conversations, 2
                            ),
                        }
                    ),
                    flush=True,
                )

    checkpoint.replace(output)
    trainable_records = [
        record
        for record in records
        if not record["conversation_model"].startswith("deterministic-safety")
    ]
    observed_distribution = Counter(
        record["observed_risk"] for record in trainable_records
    )
    scenario_mismatch_count = sum(
        1
        for record in records
        if record["expected_risk"] and record["expected_risk"] not in record["observed_risk"]
    )
    summary = {
        "conversations": args.conversations,
        "turns": len(records),
        "trainable_turns": len(trainable_records),
        "scenario_risk_mismatches": scenario_mismatch_count,
        "observed_risk_distribution": dict(sorted(observed_distribution.items())),
        "output": str(output),
        "note": (
            "observed_risk is the runtime-pipeline label used by train_dialogue.py; "
            "expected_risk is retained only to audit synthetic scenario cards. "
            "Synthetic data must be reviewed before training and kept separate from holdout safety tests."
        ),
    }
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=Path("data/groq_dialogues.jsonl")
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
