"""Sequence-of-turns nuance encoder: BiGRU + additive attention over
per-turn embeddings. Kept independent of the classifier head so it can be
reused (e.g. fed directly into the LLM's context for accurate reflection)
or swapped for a transformer encoder later without touching the rest of
the pipeline.

Interface: forward(turn_ids, turn_mask) -> (per_turn_features, summary_vector)
  turn_ids:  (batch, num_turns, max_tokens_per_turn) long tensor of token ids
  turn_mask: (batch, num_turns) 1/0 mask, None if every session has the same
             number of turns (true for Stage 1 -- always 15 Q1-Q15 turns).
  per_turn_features: (batch, num_turns, hidden_dim*2) -- per-turn signal,
             usable for emotional-trajectory / contradiction / intensity
             drift analysis downstream.
  summary_vector: (batch, hidden_dim*2) -- attention-pooled session summary,
             fed into the classifier head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from inhouse_model import config


class NuanceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = config.EMBED_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        attention_dim: int = config.ATTENTION_DIM,
        dropout: float = config.DROPOUT,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pad_idx = pad_idx
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn_proj = nn.Linear(hidden_dim * 2, attention_dim)
        self.attn_context = nn.Linear(attention_dim, 1, bias=False)

    def _pool_turns(self, turn_ids: torch.Tensor) -> torch.Tensor:
        # turn_ids: (batch, num_turns, max_tokens) -> (batch, num_turns, embed_dim)
        embedded = self.embedding(turn_ids)
        token_mask = (turn_ids != self.pad_idx).float().unsqueeze(-1)
        summed = (embedded * token_mask).sum(dim=2)
        counts = token_mask.sum(dim=2).clamp(min=1.0)
        return summed / counts

    def forward(self, turn_ids: torch.Tensor, turn_mask: torch.Tensor | None = None):
        turn_vectors = self.dropout(self._pool_turns(turn_ids))
        gru_out, _ = self.gru(turn_vectors)  # (batch, num_turns, hidden*2)

        energy = torch.tanh(self.attn_proj(gru_out))
        scores = self.attn_context(energy).squeeze(-1)  # (batch, num_turns)
        if turn_mask is not None:
            scores = scores.masked_fill(turn_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, num_turns, 1)
        summary_vector = (weights * gru_out).sum(dim=1)  # (batch, hidden*2)

        return gru_out, summary_vector

    @property
    def output_dim(self) -> int:
        return self.gru.hidden_size * 2
