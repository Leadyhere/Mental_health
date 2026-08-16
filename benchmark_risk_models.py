"""Train MentalBERT and DistilBERT on identical splits and compare results."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODELS = {
    "mentalbert": str(BASE_DIR / "models" / "base" / "mental-bert-base-uncased"),
    "distilbert": "distilbert-base-uncased",
}


def main(args):
    results = {}
    for name, model_id in MODELS.items():
        output = args.output / name
        command = [
            sys.executable,
            str(BASE_DIR / "train_classifier.py"),
            "--base-model", model_id,
            "--output", str(output),
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
        ]
        if args.synthetic_dataset:
            command.extend(["--synthetic-dataset", str(args.synthetic_dataset)])
        subprocess.run(command, check=True)
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        results[name] = {
            "base_model": model_id,
            "emergency_recall": metrics["emergency_recall"],
            "macro_f1": metrics["classification_report"]["macro avg"]["f1-score"],
            "weighted_f1": metrics["classification_report"]["weighted avg"]["f1-score"],
        }
    ranked = sorted(
        results,
        key=lambda name: (results[name]["emergency_recall"], results[name]["macro_f1"]),
        reverse=True,
    )
    report = {"models": results, "recommended_for_review": ranked[0]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=BASE_DIR / "models" / "risk_benchmarks")
    parser.add_argument("--synthetic-dataset", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
