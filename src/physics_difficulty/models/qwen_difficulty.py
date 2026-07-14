from __future__ import annotations
import torch
import torch.nn as nn
from physics_difficulty.schema import FEATURE_VALUES

class QwenDifficultyRater(nn.Module):
    """Qwen encoder with one ordinal difficulty head and ten V2 feature heads."""
    def __init__(self, backbone: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        config = getattr(backbone.config, "text_config", backbone.config)
        hidden_size = getattr(config, "hidden_size", getattr(config, "hidden_dim", 2560))
        backbone_dtype = next(backbone.parameters()).dtype
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.difficulty_head = nn.Linear(hidden_size, 5)
        self.feature_heads = nn.ModuleDict({name: nn.Linear(hidden_size, len(values)) for name, values in FEATURE_VALUES.items()})
        # Qwen is normally loaded in bf16.  Heads must use the same dtype,
        # otherwise the first linear layer receives bf16 activations and fp32 weights.
        self.norm.to(dtype=backbone_dtype)
        self.difficulty_head.to(dtype=backbone_dtype)
        self.feature_heads.to(dtype=backbone_dtype)

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask is None or torch.eq(attention_mask, 0).all(dim=-1).any():
            raise ValueError("Each batch item must contain at least one non-padding token")
        positions = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(0).expand_as(attention_mask)
        last_index = torch.where(attention_mask.bool(), positions, torch.full_like(positions, -1)).max(dim=-1).values.clamp_min(0)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last_index]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(self.norm(self._pool(output.last_hidden_state, attention_mask)))
        return {"difficulty_logits": self.difficulty_head(pooled), "feature_logits": {name: head(pooled) for name, head in self.feature_heads.items()}}
