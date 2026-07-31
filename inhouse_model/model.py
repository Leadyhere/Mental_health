"""In-house model package's public contract:

    model = InHouseModel()
    model.train(dataset)                        # Stage 1 bootstrap / Stage 2 fine-tune
    risk_level, sub_labels, confidence = model.predict(session)
    model.log_llm_decision(session, llm_output)  # Stage 2 teacher-label logging

`dataset` is a pandas DataFrame (or a path to one) shaped like
data/processed/train.parquet. `session` is either a dict/Series keyed by
inhouse_model.config.QUESTION_COLUMNS (Stage 1 / eval-style), or a plain
list[str] of already-ordered turns (live conversation, Phase 1.5+, where
narrative + structured answers have already been flattened into turns).

Emergency-gating is NOT decided here: per the spec, this model's risk_level
output is logged for agreement-rate tracking only. The backend's
`emergency_gate.py` is the sole authority for triggering the Emergency path
and must never read this model's prediction to make that decision.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from inhouse_model import config, metrics
from inhouse_model.backend import get_device
from inhouse_model.classifier_head import ClassifierHead
from inhouse_model.data import LabelEncoders, TriageDataset, Vocab, stringify_cell
from inhouse_model.nuance_encoder import NuanceEncoder


class TorchTriageModel(nn.Module):
    def __init__(self, vocab_size: int, task_num_classes: dict[str, int]):
        super().__init__()
        self.encoder = NuanceEncoder(vocab_size)
        self.head = ClassifierHead(self.encoder.output_dim, task_num_classes)

    def forward(self, turn_ids, turn_mask=None):
        per_turn_features, summary_vector = self.encoder(turn_ids, turn_mask)
        logits = self.head(summary_vector)
        return logits, per_turn_features, summary_vector


def _load_label_maps() -> dict:
    with open(config.LABEL_MAPS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _next_version_dir() -> tuple[Path, int]:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [
        int(p.name[1:]) for p in config.ARTIFACTS_DIR.iterdir()
        if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
    ]
    next_n = (max(existing) + 1) if existing else 1
    return config.ARTIFACTS_DIR / f"v{next_n}", next_n


def _latest_version_dir() -> Path | None:
    if not config.ARTIFACTS_DIR.exists():
        return None
    dirs = [
        p for p in config.ARTIFACTS_DIR.iterdir()
        if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
    ]
    if not dirs:
        return None
    return max(dirs, key=lambda p: int(p.name[1:]))


class InHouseModel:
    def __init__(self):
        self.device = get_device()
        self.model: TorchTriageModel | None = None
        self.vocab: Vocab | None = None
        self.label_encoders: LabelEncoders | None = None
        self.version: int | None = None

    # ------------------------------------------------------------------ #
    # Stage 1 bootstrap / Stage 2 fine-tune
    # ------------------------------------------------------------------ #
    def train(self, dataset, eval_dataset=None, epochs: int | None = None) -> dict:
        df = self._coerce_df(dataset)
        eval_df = self._coerce_df(eval_dataset) if eval_dataset is not None else None

        texts = [df[col] for col in config.QUESTION_COLUMNS]
        flat_texts = [stringify_cell(v) for col_series in texts for v in col_series]
        vocab = Vocab.build(flat_texts)

        label_maps_json = _load_label_maps()
        label_encoders = LabelEncoders.build(label_maps_json)

        task_num_classes = {name: label_encoders.num_classes(name) for name in config.LABEL_COLUMNS}
        model = TorchTriageModel(vocab_size=len(vocab), task_num_classes=task_num_classes).to(self.device)

        train_ds = TriageDataset(df, vocab, label_encoders)
        loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        num_epochs = epochs or config.NUM_EPOCHS

        model.train()
        epoch_losses = []
        for epoch in range(num_epochs):
            total_loss = 0.0
            for turn_ids, labels in loader:
                turn_ids = turn_ids.to(self.device)
                optimizer.zero_grad()
                logits, _, _ = model(turn_ids)
                loss = torch.tensor(0.0, device=self.device)
                for name, weight in config.TASK_LOSS_WEIGHTS.items():
                    target = labels[name].to(self.device)
                    loss = loss + weight * F.cross_entropy(logits[name], target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * turn_ids.size(0)
            avg_loss = total_loss / len(train_ds)
            epoch_losses.append(avg_loss)
            print(f"[train_stage1] epoch {epoch + 1}/{num_epochs} avg_loss={avg_loss:.4f}")

        self.model, self.vocab, self.label_encoders = model, vocab, label_encoders

        eval_report = None
        if eval_df is not None:
            eval_report = self._evaluate_df(eval_df)
            print(eval_report["formatted"])

        version_dir, version_n = _next_version_dir()
        version_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), version_dir / "model.pt")
        vocab.save(version_dir / "vocab.json")
        (version_dir / "task_num_classes.json").write_text(json.dumps(task_num_classes), encoding="utf-8")
        metadata = {
            "version": version_n,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_train_rows": len(df),
            "num_epochs": num_epochs,
            "epoch_losses": epoch_losses,
            "eval_metrics": eval_report["metrics"] if eval_report else None,
        }
        (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.version = version_n
        print(f"[train_stage1] saved checkpoint to {version_dir}")
        return metadata

    def _evaluate_df(self, eval_df: pd.DataFrame) -> dict:
        preds_by_task: dict[str, list[str]] = {name: [] for name in config.LABEL_COLUMNS}
        labels_by_task: dict[str, list[str]] = {name: [] for name in config.LABEL_COLUMNS}
        for _, row in eval_df.iterrows():
            risk_level, sub_labels, _confidence = self.predict(row)
            preds_by_task["risk_level"].append(risk_level)
            labels_by_task["risk_level"].append(row["risk_level"])
            for name in ("suicide_risk", "functional_impairment", "support_level", "help_readiness"):
                preds_by_task[name].append(sub_labels[name])
                labels_by_task[name].append(row[name])

        reports = []
        numeric_metrics = {}
        for name in config.LABEL_COLUMNS:
            classes = self.label_encoders.classes(name)
            reports.append(metrics.format_report(name, preds_by_task[name], labels_by_task[name], classes))
            numeric_metrics[name] = {
                "accuracy": metrics.accuracy(preds_by_task[name], labels_by_task[name]),
                "per_class_recall": metrics.per_class_recall(preds_by_task[name], labels_by_task[name], classes),
            }
        numeric_metrics["risk_level"]["severe_emergency_boundary"] = metrics.severe_emergency_boundary_report(
            preds_by_task["risk_level"], labels_by_task["risk_level"]
        )
        return {"formatted": "\n\n".join(reports), "metrics": numeric_metrics}

    # ------------------------------------------------------------------ #
    # Stage 3 runtime inference
    # ------------------------------------------------------------------ #
    def predict(self, session) -> tuple[str, dict, float]:
        if self.model is None:
            self.load_latest()
        turns = self._session_to_turns(session)
        turn_ids = torch.tensor(
            [[self.vocab.encode_turn(t) for t in turns]], dtype=torch.long, device=self.device
        )  # (1, num_turns, max_tokens)

        self.model.eval()
        with torch.no_grad():
            logits, _per_turn_features, _summary_vector = self.model(turn_ids)

        result = {}
        risk_probs = None
        for name, task_logits in logits.items():
            probs = F.softmax(task_logits, dim=-1)[0]
            idx = int(torch.argmax(probs).item())
            classes = self.label_encoders.classes(name)
            result[name] = classes[idx]
            if name == "risk_level":
                risk_probs = probs

        risk_level = result.pop("risk_level")
        confidence = float(torch.max(risk_probs).item())
        return risk_level, result, confidence

    # ------------------------------------------------------------------ #
    # Stage 2 logging: teacher (LLM) decisions accumulate here for the
    # periodic retrain job. Never stores raw identifying text -- callers
    # are expected to pass an already-anonymized session/turns.
    # ------------------------------------------------------------------ #
    def log_llm_decision(self, session, llm_output: dict) -> None:
        turns = self._session_to_turns(session)
        inhouse_risk_level, inhouse_sub_labels, inhouse_confidence = (None, None, None)
        if self.model is not None or _latest_version_dir() is not None:
            try:
                inhouse_risk_level, inhouse_sub_labels, inhouse_confidence = self.predict(session)
            except Exception:
                pass

        record = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "turns": turns,
            "llm_risk_level": llm_output.get("risk_level"),
            "llm_sub_labels": llm_output.get("sub_labels"),
            "llm_rationale": llm_output.get("rationale"),
            "llm_confidence": llm_output.get("confidence"),
            "inhouse_risk_level": inhouse_risk_level,
            "inhouse_sub_labels": inhouse_sub_labels,
            "inhouse_confidence": inhouse_confidence,
            "is_reviewed": False,
            "reviewer_verdict": None,
        }
        config.LLM_DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(config.LLM_DECISIONS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------ #
    # Checkpoint loading
    # ------------------------------------------------------------------ #
    def load_latest(self) -> "InHouseModel":
        version_dir = _latest_version_dir()
        if version_dir is None:
            raise FileNotFoundError(
                "No trained checkpoint found in inhouse_model/artifacts/. Run scripts/train_stage1.py first."
            )
        return self.load_version(version_dir)

    def load_version(self, version_dir: Path) -> "InHouseModel":
        vocab = Vocab.load(version_dir / "vocab.json")
        label_maps_json = _load_label_maps()
        label_encoders = LabelEncoders.build(label_maps_json)
        task_num_classes = json.loads((version_dir / "task_num_classes.json").read_text(encoding="utf-8"))

        model = TorchTriageModel(vocab_size=len(vocab), task_num_classes=task_num_classes).to(self.device)
        model.load_state_dict(torch.load(version_dir / "model.pt", map_location=self.device))
        model.eval()

        self.model, self.vocab, self.label_encoders = model, vocab, label_encoders
        self.version = int(version_dir.name[1:])
        return self

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce_df(dataset) -> pd.DataFrame:
        if isinstance(dataset, pd.DataFrame):
            return dataset
        return pd.read_parquet(dataset)

    @staticmethod
    def _session_to_turns(session) -> list[str]:
        if isinstance(session, (dict, pd.Series)):
            return [stringify_cell(session.get(col)) for col in config.QUESTION_COLUMNS]
        return [str(t) for t in session]
