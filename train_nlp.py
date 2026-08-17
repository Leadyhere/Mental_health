"""Train deep-learning NLP heads for emotion, topic and semantic-slot detection."""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

from mindtriage.training import TrainingMonitor, timestamped_run_dir


BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "mental_health_final_USER_TONE_dataset (1).xlsx"
OUTPUT = BASE_DIR / "models" / "inhouse_nlp"
BASE_MODEL = "mental/mental-bert-base-uncased"

FIELD_MAP = {
    "trigger": ["Q1: Reason for Opening Form"],
    "duration": ["Q4: How Long Has This Been Going On"],
    "impact": ["Q3: How Is It Affecting Daily Life", "Q13: Impact on Sleep / Appetite / Energy"],
    "coping": ["Q6: Coping Behaviors"],
    "support": ["Q7: Feels Supported", "Q14: Support Person Available"],
}


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, multilabel):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_length, self.multilabel = tokenizer, max_length, multilabel

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        dtype = torch.float32 if self.multilabel else torch.long
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=dtype),
        }


class SlotTokenDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length, label_map):
        self.samples, self.tokenizer = samples, tokenizer
        self.max_length, self.label_map = max_length, label_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text, spans = self.samples[index]
        encoded = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.max_length,
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping").squeeze(0).tolist()
        labels = []
        previous_slot = None
        for start, end in offsets:
            if start == end:
                labels.append(-100)
                previous_slot = None
                continue
            matching = next((slot for span_start, span_end, slot in spans if start < span_end and end > span_start), None)
            if matching is None:
                labels.append(self.label_map["O"])
                previous_slot = None
            else:
                prefix = "I" if previous_slot == matching else "B"
                labels.append(self.label_map[f"{prefix}-{matching}"])
                previous_slot = matching
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def clean(value) -> str | None:
    if pd.isna(value) or not str(value).strip():
        return None
    return str(value).strip()


def combined_text(row) -> str:
    parts = []
    for columns in FIELD_MAP.values():
        for column in columns:
            value = clean(row.get(column))
            if value:
                parts.append(value)
    emotion = clean(row.get("Q5: Current Emotions"))
    if emotion:
        parts.append(emotion)
    return " ".join(parts)


def prepare_tasks(frame: pd.DataFrame, seed: int):
    topic_rows, emotion_rows, slot_rows = [], [], []
    rng = random.Random(seed)
    for _, row in frame.iterrows():
        text = combined_text(row)
        topic = clean(row.get("Q1: Reason for Opening Form"))
        emotions = clean(row.get("Q5: Current Emotions"))
        if text and topic:
            topic_rows.append((text, topic))
        if text and emotions:
            labels = [item.strip().lower().replace(" ", "_") for item in re.split(r"[,;/]", emotions) if item.strip()]
            emotion_rows.append((text, labels))

        fragments = []
        for slot, columns in FIELD_MAP.items():
            values = [clean(row.get(column)) for column in columns]
            values = [value for value in values if value]
            if values:
                fragment = " ".join(values)
                slot_rows.append((fragment, [(0, len(fragment), slot)]))
                fragments.append((slot, fragment))
        if len(fragments) >= 2:
            rng.shuffle(fragments)
            chosen = fragments[: rng.randint(2, len(fragments))]
            text_parts, spans, cursor = [], [], 0
            for slot, value in chosen:
                if text_parts:
                    cursor += 2
                start = cursor
                text_parts.append(value)
                cursor += len(value)
                spans.append((start, cursor, slot))
            slot_rows.append((". ".join(text_parts), spans))
    return {
        "topic": deduplicate_task_rows(topic_rows, "topic"),
        "emotion": deduplicate_task_rows(emotion_rows, "emotion"),
        "slots": deduplicate_task_rows(slot_rows, "slots"),
    }


def deduplicate_task_rows(rows, task: str):
    unique = {}
    for text, target in rows:
        normalized_target = json.dumps(target, sort_keys=True)
        if text in unique and unique[text][1] != normalized_target:
            raise ValueError(f"Conflicting {task} labels found for duplicate text")
        unique.setdefault(text, (target, normalized_target))
    return [(text, target_and_key[0]) for text, target_and_key in unique.items()]


def encode_labels(task, rows):
    texts = [text for text, _ in rows]
    if task == "topic":
        labels = sorted({label for _, label in rows})
        label_map = {label: index for index, label in enumerate(labels)}
        return texts, [label_map[label] for _, label in rows], labels, False
    encoder = MultiLabelBinarizer()
    encoded = encoder.fit_transform([labels for _, labels in rows]).astype(np.float32)
    return texts, encoded.tolist(), encoder.classes_.tolist(), True


def split_single_label_indices(labels, seed: int, minimum_for_holdout: int = 10):
    """Create an 80/10/10 split without leaking underrepresented labels.

    Labels with too few unique examples to place at least one example in both
    validation and test remain in training. Their holdout performance therefore
    cannot be claimed and is recorded in the model metrics.
    """
    counts = Counter(labels)
    train_only_labels = {
        label for label, count in counts.items() if count < minimum_for_holdout
    }
    holdout_ids = [
        index for index, label in enumerate(labels) if label not in train_only_labels
    ]
    train_only_ids = [
        index for index, label in enumerate(labels) if label in train_only_labels
    ]
    if not holdout_ids:
        raise ValueError(
            "No label has enough unique examples for validation and test splits"
        )

    holdout_labels = [labels[index] for index in holdout_ids]
    train_ids, temp_ids = train_test_split(
        holdout_ids,
        test_size=0.2,
        random_state=seed,
        stratify=holdout_labels,
    )
    temp_labels = [labels[index] for index in temp_ids]
    validation_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.5,
        random_state=seed,
        stratify=temp_labels,
    )
    train_ids.extend(train_only_ids)
    random.Random(seed).shuffle(train_ids)
    return train_ids, validation_ids, test_ids, {
        label: counts[label] for label in sorted(train_only_labels)
    }


def evaluate_slots(
    model, loader, device, label_count, monitor: TrainingMonitor | None = None,
    phase: str = "evaluation", epoch: int | None = None,
):
    model.eval()
    targets, predictions, total_loss = [], [], 0.0
    batches = monitor.evaluation_batches(loader, phase, epoch) if monitor else loader
    with torch.no_grad():
        for batch in batches:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            total_loss += output.loss.item()
            predicted = output.logits.argmax(dim=-1)
            mask = batch["labels"] != -100
            targets.extend(batch["labels"][mask].cpu().tolist())
            predictions.extend(predicted[mask].cpu().tolist())
    labels = list(range(1, label_count))
    return (
        total_loss / max(len(loader), 1),
        f1_score(targets, predictions, labels=labels, average="macro", zero_division=0),
    )


def train_slots(samples, args, device):
    slot_names = list(FIELD_MAP)
    labels = ["O"] + [f"{prefix}-{slot}" for slot in slot_names for prefix in ("B", "I")]
    label_map = {label: index for index, label in enumerate(labels)}
    indices = list(range(len(samples)))
    train_ids, temp_ids = train_test_split(indices, test_size=0.2, random_state=args.seed)
    validation_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.base_model, num_labels=len(labels))
    model.config.id2label = {index: label for index, label in enumerate(labels)}
    model.config.label2id = label_map
    model.to(device)

    def loader(ids, shuffle=False):
        subset = [samples[index] for index in ids]
        return DataLoader(
            SlotTokenDataset(subset, tokenizer, args.max_length, label_map),
            batch_size=args.batch_size, shuffle=shuffle,
        )

    train_loader, validation_loader, test_loader = loader(train_ids, True), loader(validation_ids), loader(test_ids)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    best_loss, best_state, history = float("inf"), None, []
    run_dir = args.tensorboard_run_dir / "slots"
    monitor = TrainingMonitor(
        run_dir=run_dir,
        task_name="nlp-slots",
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        log_every_steps=args.log_every_steps,
        tensorboard_enabled=not args.no_tensorboard,
    )
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        progress = monitor.training_batches(train_loader, epoch + 1)
        for batch_number, batch in enumerate(progress, start=1):
            optimizer.zero_grad()
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            monitor.log_batch(progress, epoch + 1, batch_number, float(loss.item()))
        validation_loss, validation_f1 = evaluate_slots(
            model, validation_loader, device, len(labels), monitor,
            "validation", epoch + 1,
        )
        epoch_result = {"epoch": epoch + 1, "train_loss": total / len(train_loader), "validation_loss": validation_loss, "validation_slot_macro_f1": validation_f1}
        history.append(epoch_result)
        monitor.log_epoch(
            epoch + 1,
            {
                "train_loss": epoch_result["train_loss"],
                "validation_loss": validation_loss,
                "validation_slot_macro_f1": validation_f1,
            },
        )
        print(json.dumps({"task": "slots", **epoch_result}), flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(device)
    test_loss, test_f1 = evaluate_slots(
        model, test_loader, device, len(labels), monitor, "test"
    )
    monitor.log_test({"loss": test_loss, "slot_macro_f1": test_f1})
    monitor.finish(len(history))
    monitor.close()
    output = args.output / "slots"
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    metrics = {
        "task": "BIO semantic slot extraction",
        "algorithm": "fine-tuned transformer token classification",
        "labels": labels,
        "rows": len(samples),
        "split_sizes": {"train": len(train_ids), "validation": len(validation_ids), "test": len(test_ids)},
        "cross_split_exact_text_overlap": 0,
        "test_loss": test_loss,
        "test_slot_macro_f1_excluding_o": test_f1,
        "history": history,
        "evaluation_scope": "Synthetic slot spans derived from questionnaire fields; validate on separately annotated free-form chat before deployment.",
        "tensorboard_run": str(run_dir) if not args.no_tensorboard else None,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"task": "slots", "rows": len(samples), "test_slot_macro_f1_excluding_o": test_f1}))


def evaluate(
    model, loader, device, multilabel, monitor: TrainingMonitor | None = None,
    phase: str = "evaluation", epoch: int | None = None,
):
    model.eval()
    targets, predictions = [], []
    total_loss = 0.0
    batches = monitor.evaluation_batches(loader, phase, epoch) if monitor else loader
    with torch.no_grad():
        for batch in batches:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            total_loss += output.loss.item()
            if multilabel:
                predictions.extend((torch.sigmoid(output.logits) >= 0.5).int().cpu().tolist())
                targets.extend(batch["labels"].int().cpu().tolist())
            else:
                predictions.extend(output.logits.argmax(dim=1).cpu().tolist())
                targets.extend(batch["labels"].cpu().tolist())
    score = f1_score(targets, predictions, average="micro", zero_division=0)
    return total_loss / max(len(loader), 1), score, targets, predictions


def train_task(task, rows, args, device):
    texts, labels, names, multilabel = encode_labels(task, rows)
    indices = list(range(len(texts)))
    train_only_label_counts = {}
    if multilabel:
        train_ids, temp_ids = train_test_split(
            indices, test_size=0.2, random_state=args.seed
        )
        validation_ids, test_ids = train_test_split(
            temp_ids, test_size=0.5, random_state=args.seed
        )
    else:
        train_ids, validation_ids, test_ids, train_only_label_counts = (
            split_single_label_indices(labels, args.seed)
        )
        if train_only_label_counts:
            print(json.dumps({
                "task": task,
                "warning": "rare labels are training-only and excluded from holdout metrics",
                "train_only_labels": {
                    names[label]: count
                    for label, count in train_only_label_counts.items()
                },
            }), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(names),
        problem_type="multi_label_classification" if multilabel else "single_label_classification",
    )
    model.config.id2label = {index: label for index, label in enumerate(names)}
    model.config.label2id = {label: index for index, label in enumerate(names)}
    model.to(device)

    def loader(ids, shuffle=False):
        return DataLoader(
            TextDataset([texts[i] for i in ids], [labels[i] for i in ids], tokenizer, args.max_length, multilabel),
            batch_size=args.batch_size, shuffle=shuffle,
        )

    train_loader, validation_loader, test_loader = loader(train_ids, True), loader(validation_ids), loader(test_ids)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    best_loss, best_state, history = float("inf"), None, []
    run_dir = args.tensorboard_run_dir / task
    monitor = TrainingMonitor(
        run_dir=run_dir,
        task_name=f"nlp-{task}",
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        log_every_steps=args.log_every_steps,
        tensorboard_enabled=not args.no_tensorboard,
    )
    for epoch in range(args.epochs):
        model.train()
        training_loss = 0.0
        progress = monitor.training_batches(train_loader, epoch + 1)
        for batch_number, batch in enumerate(progress, start=1):
            optimizer.zero_grad()
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            training_loss += loss.item()
            monitor.log_batch(progress, epoch + 1, batch_number, float(loss.item()))
        validation_loss, validation_f1, _, _ = evaluate(
            model, validation_loader, device, multilabel, monitor,
            "validation", epoch + 1,
        )
        epoch_result = {"epoch": epoch + 1, "train_loss": training_loss / len(train_loader), "validation_loss": validation_loss, "validation_micro_f1": validation_f1}
        history.append(epoch_result)
        monitor.log_epoch(
            epoch + 1,
            {
                "train_loss": epoch_result["train_loss"],
                "validation_loss": validation_loss,
                "validation_micro_f1": validation_f1,
            },
        )
        print(json.dumps({"task": task, **epoch_result}), flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(device)
    test_loss, test_f1, targets, predictions = evaluate(
        model, test_loader, device, multilabel, monitor, "test"
    )
    monitor.log_test({"loss": test_loss, "micro_f1": test_f1})
    monitor.finish(len(history))
    monitor.close()
    output = args.output / task
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    metrics = {
        "task": task, "algorithm": "fine-tuned transformer sequence classification",
        "labels": names, "rows": len(rows), "test_loss": test_loss,
        "test_micro_f1": test_f1, "history": history,
        "split_sizes": {"train": len(train_ids), "validation": len(validation_ids), "test": len(test_ids)},
        "train_only_rare_labels": {
            names[label]: count for label, count in train_only_label_counts.items()
        },
        "cross_split_exact_text_overlap": 0,
        "evaluation_scope": "Questionnaire-derived in-distribution holdout; validate on separately labelled free-form chat before deployment.",
        "tensorboard_run": str(run_dir) if not args.no_tensorboard else None,
        "classification_report": classification_report(targets, predictions, output_dict=True, zero_division=0),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"task": task, "rows": len(rows), "test_micro_f1": test_f1}))


def main(args):
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be greater than zero")
    if args.learning_rate <= 0 or args.max_length <= 0:
        raise ValueError("learning rate and max length must be greater than zero")
    if args.log_every_steps <= 0:
        raise ValueError("log-every-steps must be greater than zero")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    frame = pd.read_excel(args.dataset, sheet_name="Combined Data")
    if args.max_rows:
        frame = frame.sample(min(args.max_rows, len(frame)), random_state=args.seed)
    tasks = prepare_tasks(frame, args.seed)
    for task in args.tasks:
        if len(tasks[task]) < 20:
            raise ValueError(f"Task {task} has too few unique examples for an 80/10/10 split")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.tensorboard_run_dir = args.tensorboard_dir or timestamped_run_dir(BASE_DIR, "nlp")
    selected_tasks = set(args.tasks)
    for task in ("emotion", "topic"):
        if task in selected_tasks:
            train_task(task, tasks[task], args, device)
    if "slots" in selected_tasks:
        train_slots(tasks["slots"], args, device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("emotion", "topic", "slots"),
        default=("emotion", "topic", "slots"),
        help="Train only the selected NLP heads; defaults to all three.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--log-every-steps", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
