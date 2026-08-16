"""Run the complete model-training pipeline in a reproducible order."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MENTALBERT = ROOT / "models" / "base" / "mental-bert-base-uncased"
SYNTHETIC_RISK = ROOT / "outputs" / "dataset-v2" / "mental_health_synthetic_v2_100k_report.csv"
GROQ_DIALOGUES = ROOT / "data" / "groq_dialogues.jsonl"


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def main(args) -> None:
    if not MENTALBERT.exists():
        raise FileNotFoundError("MentalBERT is missing; run scripts/download_mentalbert.py first")

    run("train_nlp.py", "--base-model", str(MENTALBERT))
    risk_command = ["train_classifier.py", "--base-model", str(MENTALBERT)]
    if args.with_synthetic_risk:
        risk_command.extend(["--synthetic-dataset", str(SYNTHETIC_RISK)])
    run(*risk_command)

    if args.with_benchmark:
        benchmark_command = ["benchmark_risk_models.py"]
        if args.with_synthetic_risk:
            benchmark_command.extend(["--synthetic-dataset", str(SYNTHETIC_RISK)])
        run(*benchmark_command)

    if args.with_dialogue:
        if not GROQ_DIALOGUES.exists():
            raise FileNotFoundError(
                "Groq dialogue data is missing; run scripts/generate_dialogue_scenarios.py first"
            )
        run("train_dialogue.py", "--dataset", str(GROQ_DIALOGUES))

    run("-m", "unittest", "discover", "-s", "tests", "-v")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-synthetic-risk", action="store_true")
    parser.add_argument("--with-benchmark", action="store_true")
    parser.add_argument("--with-dialogue", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
