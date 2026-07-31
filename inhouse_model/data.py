"""Vocabulary, tokenization, and Dataset/DataLoader for Stage 1 training.

Each dataset row becomes a fixed sequence of 15 "turns" (one per Q1-Q15,
in order, per inhouse_model.config.QUESTION_COLUMNS). A null/skipped answer
(the N/A-Feeling Fine cohort) becomes a turn containing only the special
<empty> token, rather than a masked-out position -- this lets the model
learn "this question was never reached" as a signal in its own right. Live
conversation turns at runtime go through the same `tokenize`/`encode_turn`
functions via `inhouse_model.model.InHouseModel.predict`.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset

from inhouse_model import config

PAD, UNK, EMPTY = "<pad>", "<unk>", "<empty>"
SPECIALS = [PAD, UNK, EMPTY]
MAX_TOKENS_PER_TURN = 12

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text) -> list[str]:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    return _TOKEN_RE.findall(str(text).lower())


def stringify_cell(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (int, float)):
        return f"intensity_{int(value)}"
    return str(value)


@dataclass
class Vocab:
    stoi: dict
    itos: list

    @classmethod
    def build(cls, texts: list[str], max_size: int = config.MAX_VOCAB_SIZE) -> "Vocab":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))
        most_common = [tok for tok, _ in counter.most_common(max_size - len(SPECIALS))]
        itos = list(SPECIALS) + most_common
        stoi = {tok: i for i, tok in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode_turn(self, text: str, max_tokens: int = MAX_TOKENS_PER_TURN) -> list[int]:
        tokens = tokenize(text)
        if not tokens:
            return [self.stoi[EMPTY]] + [self.stoi[PAD]] * (max_tokens - 1)
        ids = [self.stoi.get(tok, self.stoi[UNK]) for tok in tokens[:max_tokens]]
        ids += [self.stoi[PAD]] * (max_tokens - len(ids))
        return ids

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f)

    @classmethod
    def load(cls, path) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        itos = data["itos"]
        stoi = {tok: i for i, tok in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def __len__(self) -> int:
        return len(self.itos)


@dataclass
class LabelEncoders:
    maps: dict  # label_name -> {class_name: idx}

    @classmethod
    def build(cls, label_maps_json: dict) -> "LabelEncoders":
        maps = {}
        for label_name in config.LABEL_COLUMNS:
            classes = label_maps_json[label_name]
            maps[label_name] = {c: i for i, c in enumerate(classes)}
        return cls(maps=maps)

    def encode(self, label_name: str, value) -> int:
        mapping = self.maps[label_name]
        if value not in mapping:
            return 0
        return mapping[value]

    def num_classes(self, label_name: str) -> int:
        return len(self.maps[label_name])

    def classes(self, label_name: str) -> list[str]:
        inv = {i: c for c, i in self.maps[label_name].items()}
        return [inv[i] for i in range(len(inv))]


def row_to_turns(row: pd.Series) -> list[str]:
    return [stringify_cell(row[col]) for col in config.QUESTION_COLUMNS]


class TriageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Vocab, label_encoders: LabelEncoders):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.label_encoders = label_encoders

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        turns = row_to_turns(row)
        turn_ids = torch.tensor(
            [self.vocab.encode_turn(t) for t in turns], dtype=torch.long
        )  # (num_turns, max_tokens)
        labels = {
            name: torch.tensor(self.label_encoders.encode(name, row[col]), dtype=torch.long)
            for name, col in config.LABEL_COLUMNS.items()
        }
        return turn_ids, labels
