"""Shared single-question scalar scorer for Bradley-Terry training."""
from __future__ import annotations

import torch
import torch.nn as nn

from physics_difficulty.schema import FEATURE_VALUES


class QwenPairwiseRater(nn.Module):
    def __init__(self, backbone: nn.Module, dropout: float = 0.1, auxiliary_features: bool = False):
        super().__init__()
        self.backbone = backbone
        config = getattr(backbone.config, "text_config", backbone.config)
        hidden_size = getattr(config, "hidden_size", getattr(config, "hidden_dim", 2560))
        backbone_dtype = next(backbone.parameters()).dtype
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.score_head = nn.Linear(hidden_size, 1)
        self.auxiliary_features = auxiliary_features
        # Do not let optional-head initialization advance the global RNG. This
        # keeps V1/V2 dropout streams matched under the same seed.
        with torch.random.fork_rng(devices=[]):
            self.auxiliary_heads = nn.ModuleDict({
                name: nn.Linear(hidden_size, len(values)) for name, values in FEATURE_VALUES.items()
            }) if auxiliary_features else nn.ModuleDict()
        self.norm.to(dtype=backbone_dtype)
        self.score_head.to(dtype=backbone_dtype)
        self.auxiliary_heads.to(dtype=backbone_dtype)

    @staticmethod
    def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask is None or torch.eq(attention_mask, 0).all(dim=-1).any():
            raise ValueError("Each question must contain at least one non-padding token")
        positions = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(0).expand_as(attention_mask)
        last_index = torch.where(attention_mask.bool(), positions, torch.full_like(positions, -1)).max(dim=-1).values
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last_index]

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(self.norm(self._pool(output.last_hidden_state, attention_mask)))

    def score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.score_head(self.encode(input_ids, attention_mask)).squeeze(-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pair_count: int) -> dict[str, torch.Tensor]:
        representation = self.encode(input_ids, attention_mask)
        scores = self.score_head(representation).squeeze(-1)
        if scores.numel() != pair_count * 2:
            raise ValueError("combined batch must contain all A questions followed by all B questions")
        score_a, score_b = scores[:pair_count], scores[pair_count:]
        result = {"score_a": score_a, "score_b": score_b, "pair_logits": score_a - score_b}
        if self.auxiliary_features:
            logits = {name: head(representation) for name, head in self.auxiliary_heads.items()}
            result["auxiliary_logits_a"] = {name: value[:pair_count] for name, value in logits.items()}
            result["auxiliary_logits_b"] = {name: value[pair_count:] for name, value in logits.items()}
        return result
