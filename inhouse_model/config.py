"""Hyperparameters, paths, and the fixed Q1-Q15 turn ordering used to turn a
row of the dataset (or a live session's structured answers) into the ordered
list of "turns" the nuance_encoder consumes.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
LLM_DECISIONS_PATH = PROCESSED_DIR / "llm_decisions.jsonl"

TRAIN_PARQUET = PROCESSED_DIR / "train.parquet"
EVAL_PARQUET = PROCESSED_DIR / "eval.parquet"
LABEL_MAPS_PATH = PROCESSED_DIR / "label_maps.json"

# Fixed order in which Q1-Q15 answers are turned into "turns" fed to the
# nuance encoder. Q8 (numeric intensity) is stringified so it flows through
# the same text pipeline as the other questions.
QUESTION_COLUMNS = [
    "q1_reason",
    "q2_when_hardest",
    "q3_daily_impact",
    "q4_duration",
    "q5_emotions",
    "q6_coping",
    "q7_supported",
    "q8_intensity",
    "q9_passive_si",
    "q10_active_si",
    "q11_feels_safe",
    "q12_self_label",
    "q13_sleep_appetite_energy",
    "q14_support_person",
    "q15_open_to_support",
]

LABEL_COLUMNS = {
    "risk_level": "risk_level",
    "suicide_risk": "suicide_risk",
    "functional_impairment": "functional_impairment",
    "support_level": "support_level",
    "help_readiness": "help_readiness",
}

# --- model hyperparameters ---
MAX_VOCAB_SIZE = 6000
EMBED_DIM = 64
HIDDEN_DIM = 96
ATTENTION_DIM = 64
DROPOUT = 0.2
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 8
SEED = 42

# Loss weights per task -- risk_level is the primary/highest-stakes output.
TASK_LOSS_WEIGHTS = {
    "risk_level": 2.0,
    "suicide_risk": 1.0,
    "functional_impairment": 0.5,
    "support_level": 0.5,
    "help_readiness": 0.5,
}

RETRAIN_EVERY_N_SESSIONS_DEFAULT = 50
