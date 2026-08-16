"""Train and evaluate the five-level MindTriage risk classifier."""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "mental_health_final_USER_TONE_dataset (1).xlsx"
DEFAULT_SYNTHETIC_DATASET = BASE_DIR / "outputs" / "dataset-v2" / "mental_health_synthetic_v2_100k_report.csv"
DEFAULT_OUTPUT = BASE_DIR / "models" / "inhouse_risk_classifier"
PRIMARY_MODEL = "mental/mental-bert-base-uncased"
LABELS = ["Low", "Mild", "Moderate", "Severe", "Emergency"]
LABEL_MAP = {label: index for index, label in enumerate(LABELS)}

QUESTION_COLUMNS = [
    "Q1: Reason for Opening Form",
    "Q2: When Does It Feel Hardest",
    "Q3: How Is It Affecting Daily Life",
    "Q4: How Long Has This Been Going On",
    "Q5: Current Emotions",
    "Q6: Coping Behaviors",
    "Q7: Feels Supported",
    "Q8: Intensity (0-10)",
    "Q9: Passive SI",
    "Q10: Active SI / SH Thoughts",
    "Q11: Feels Safe",
    "Q12: Self-Label",
    "Q13: Impact on Sleep / Appetite / Energy",
    "Q14: Support Person Available",
    "Q15: Open to Getting Support",
]

QUESTION_PREFIXES = {
    "Q1: Reason for Opening Form": "Reason",
    "Q2: When Does It Feel Hardest": "Hardest time",
    "Q3: How Is It Affecting Daily Life": "Daily impact",
    "Q4: How Long Has This Been Going On": "Duration",
    "Q5: Current Emotions": "Emotions",
    "Q6: Coping Behaviors": "Coping",
    "Q7: Feels Supported": "Feels supported",
    "Q8: Intensity (0-10)": "Intensity",
    "Q9: Passive SI": "Passive safety signal",
    "Q10: Active SI / SH Thoughts": "Active safety signal",
    "Q11: Feels Safe": "Feels safe",
    "Q12: Self-Label": "Self-description",
    "Q13: Impact on Sleep / Appetite / Energy": "Sleep appetite energy impact",
    "Q14: Support Person Available": "Support person",
    "Q15: Open to Getting Support": "Open to support",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_text(row: pd.Series) -> str:
    parts = []
    for column in QUESTION_COLUMNS:
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            parts.append(f"{QUESTION_PREFIXES[column]}: {str(value).strip()}")
    return " | ".join(parts)


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="Combined Data")
    missing = [column for column in QUESTION_COLUMNS + ["Risk Level"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    frame = frame.copy()
    frame["Risk Level"] = frame["Risk Level"].astype(str).str.strip()
    frame = frame[frame["Risk Level"].isin(LABEL_MAP)].copy()
    frame["text"] = frame.apply(build_text, axis=1)
    frame = frame[frame["text"].str.len() > 0].copy()
    frame["label"] = frame["Risk Level"].map(LABEL_MAP)
    if frame.empty:
        raise ValueError("No valid labelled rows were found")
    return frame[["text", "label", "Risk Level"]].reset_index(drop=True)


def load_synthetic_training_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Combined Text", "Risk Level"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Synthetic dataset must contain {sorted(required)}")
    frame["Risk Level"] = frame["Risk Level"].astype(str).str.strip()
    frame = frame[frame["Risk Level"].isin(LABEL_MAP)].copy()
    frame["text"] = frame["Combined Text"].astype(str).str.strip()
    frame["label"] = frame["Risk Level"].map(LABEL_MAP)
    return frame[["text", "label", "Risk Level"]].reset_index(drop=True)


class RiskDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        self.texts = frame["text"].tolist()
        self.labels = frame["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def evaluate(model, loader, device) -> tuple[float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    predictions, targets = [], []
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            total_loss += loss_fn(logits, labels).item()
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return total_loss / max(len(loader), 1), predictions, targets


def train(args) -> None:
    seed_everything(args.seed)
    frame = load_dataset(args.dataset)
    train_frame, temp_frame = train_test_split(
        frame, test_size=0.30, stratify=frame["label"], random_state=args.seed
    )
    validation_frame, test_frame = train_test_split(
        temp_frame, test_size=0.50, stratify=temp_frame["label"], random_state=args.seed
    )
    primary_train_size = len(train_frame)
    synthetic_train_size = 0
    if args.synthetic_dataset:
        synthetic_frame = load_synthetic_training_data(args.synthetic_dataset)
        if args.max_synthetic_rows:
            synthetic_frame = synthetic_frame.sample(
                min(args.max_synthetic_rows, len(synthetic_frame)), random_state=args.seed
            )
        synthetic_train_size = len(synthetic_frame)
        train_frame = pd.concat([train_frame, synthetic_frame], ignore_index=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model, num_labels=len(LABELS)
        )
        selected_model = args.base_model
    except Exception as exc:
        raise RuntimeError(
            f"Base transformer could not be loaded: {args.base_model}. If it is gated, "
            "accept its Hugging Face terms and run `hf auth login`."
        ) from exc

    model.config.id2label = {index: label for index, label in enumerate(LABELS)}
    model.config.label2id = LABEL_MAP

    train_loader = DataLoader(
        RiskDataset(train_frame, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        RiskDataset(validation_frame, tokenizer, args.max_length),
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(
        RiskDataset(test_frame, tokenizer, args.max_length),
        batch_size=args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(LABELS)), y=train_frame["label"]
    )
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    best_validation_loss = float("inf")
    best_state = None
    patience_remaining = args.patience
    history = []
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        validation_loss, _, _ = evaluate(model, validation_loader, device)
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(len(train_loader), 1),
            "validation_loss": validation_loss,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result))
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_remaining = args.patience
        else:
            patience_remaining -= 1
            if patience_remaining == 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    test_loss, predictions, targets = evaluate(model, test_loader, device)
    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "test_loss": test_loss,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=list(range(len(LABELS)))
        ).tolist(),
        "emergency_recall": report["Emergency"]["recall"],
        "split_sizes": {
            "train": len(train_frame),
            "primary_train": primary_train_size,
            "synthetic_train": synthetic_train_size,
            "validation": len(validation_frame),
            "test": len(test_frame),
        },
        "class_counts": frame["Risk Level"].value_counts().to_dict(),
        "history": history,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata = {
        "model_version": f"risk-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "base_model": selected_model,
        "labels": LABELS,
        "dataset": args.dataset.name,
        "max_length": args.max_length,
    }
    (args.output / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output": str(args.output), **metrics["split_sizes"]}))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--synthetic-dataset", type=Path)
    parser.add_argument("--max-synthetic-rows", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-model", default=PRIMARY_MODEL)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
