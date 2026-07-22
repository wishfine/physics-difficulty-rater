"""Shared single-question scalar scorer for Bradley-Terry training."""
from __future__ import annotations

import torch
import torch.nn as nn


class QwenPairwiseRater(nn.Module):
    def __init__(self, backbone: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        config = getattr(backbone.config, "text_config", backbone.config)
        hidden_size = getattr(config, "hidden_size", getattr(config, "hidden_dim", 2560))
        backbone_dtype = next(backbone.parameters()).dtype
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.score_head = nn.Linear(hidden_size, 1)
        self.norm.to(dtype=backbone_dtype)
        self.score_head.to(dtype=backbone_dtype)

    @staticmethod
    def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask is None or torch.eq(attention_mask, 0).all(dim=-1).any():
            raise ValueError("Each question must contain at least one non-padding token")
        positions = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(0).expand_as(attention_mask)
        last_index = torch.where(attention_mask.bool(), positions, torch.full_like(positions, -1)).max(dim=-1).values
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last_index]

    def score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(self.norm(self._pool(output.last_hidden_state, attention_mask)))
        return self.score_head(pooled).squeeze(-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pair_count: int) -> dict[str, torch.Tensor]:
        scores = self.score(input_ids, attention_mask)
        if scores.numel() != pair_count * 2:
            raise ValueError("combined batch must contain all A questions followed by all B questions")
        score_a, score_b = scores[:pair_count], scores[pair_count:]
        return {"score_a": score_a, "score_b": score_b, "pair_logits": score_a - score_b}
