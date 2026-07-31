"""Evaluation metrics for Stage 1/Stage 2, with explicit emphasis on the
Severe/Emergency boundary -- the single highest-stakes distinction per the
product spec (passive SI without intent = Severe, never Emergency).
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def accuracy(preds: list[str], labels: list[str]) -> float:
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    return correct / len(labels) if labels else 0.0


def per_class_recall(preds: list[str], labels: list[str], classes: list[str]) -> dict[str, float]:
    result = {}
    for c in classes:
        total = sum(1 for l in labels if l == c)
        if total == 0:
            result[c] = float("nan")
            continue
        correct = sum(1 for p, l in zip(preds, labels) if l == c and p == c)
        result[c] = correct / total
    return result


def confusion_matrix(preds: list[str], labels: list[str], classes: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    mat = np.zeros((len(classes), len(classes)), dtype=int)
    for p, l in zip(preds, labels):
        mat[idx[l], idx[p]] += 1
    return mat


def severe_emergency_boundary_report(preds: list[str], labels: list[str]) -> dict:
    """Treats {Severe, Emergency} as the safety-critical binary distinction.

    Reports, among rows whose true label is Severe or Emergency:
      - how often the model correctly keeps Severe vs Emergency separate
      - false-Severe-as-Emergency rate (over-escalation -- costly but safer)
      - false-Emergency-as-Severe rate (under-escalation -- the dangerous
        direction; per spec, low-confidence cases must escalate, never
        de-escalate, so this is the number to watch most closely)
    """
    pairs = [(p, l) for p, l in zip(preds, labels) if l in ("Severe", "Emergency")]
    if not pairs:
        return {"n": 0}
    n = len(pairs)
    severe_total = sum(1 for _, l in pairs if l == "Severe")
    emergency_total = sum(1 for _, l in pairs if l == "Emergency")
    severe_correct = sum(1 for p, l in pairs if l == "Severe" and p == "Severe")
    emergency_correct = sum(1 for p, l in pairs if l == "Emergency" and p == "Emergency")
    emergency_predicted_as_severe = sum(1 for p, l in pairs if l == "Emergency" and p == "Severe")
    severe_predicted_as_emergency = sum(1 for p, l in pairs if l == "Severe" and p == "Emergency")

    return {
        "n": n,
        "severe_recall": severe_correct / severe_total if severe_total else float("nan"),
        "emergency_recall": emergency_correct / emergency_total if emergency_total else float("nan"),
        "emergency_under_escalated_as_severe_rate": (
            emergency_predicted_as_severe / emergency_total if emergency_total else float("nan")
        ),
        "severe_over_escalated_as_emergency_rate": (
            severe_predicted_as_emergency / severe_total if severe_total else float("nan")
        ),
    }


def agreement_rate(inhouse_preds: list[str], llm_preds: list[str]) -> float:
    """Stage 2's core metric: how often the in-house model's risk_level
    matches the LLM's on held-out sessions."""
    return accuracy(inhouse_preds, llm_preds)


def format_report(task_name: str, preds: list[str], labels: list[str], classes: list[str]) -> str:
    lines = [f"=== {task_name} ===", f"accuracy: {accuracy(preds, labels):.4f}"]
    recalls = per_class_recall(preds, labels, classes)
    lines.append("per-class recall:")
    for c, r in recalls.items():
        lines.append(f"  {c}: {r:.4f}" if not np.isnan(r) else f"  {c}: n/a (0 support)")
    if task_name == "risk_level":
        boundary = severe_emergency_boundary_report(preds, labels)
        lines.append("Severe/Emergency boundary report:")
        for k, v in boundary.items():
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)
