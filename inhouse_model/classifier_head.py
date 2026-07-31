"""Multi-task classifier head consuming the nuance_encoder's summary vector.

Kept as its own module (separate from NuanceEncoder) per the spec's
requirement that the encoder be independently swappable later.
"""
from __future__ import annotations

import torch.nn as nn

from inhouse_model import config


class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int, task_num_classes: dict[str, int], hidden: int = 128, dropout: float = config.DROPOUT):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({name: nn.Linear(hidden, n) for name, n in task_num_classes.items()})

    def forward(self, summary_vector):
        h = self.trunk(summary_vector)
        return {name: head(h) for name, head in self.heads.items()}
