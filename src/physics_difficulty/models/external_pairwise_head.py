"""Inference-only task heads for representations produced outside Transformers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from physics_difficulty.models.pairwise_loading import checkpoint_feature_values
from physics_difficulty.schema import FEATURE_VALUES


class ExternalPairwiseHead(nn.Module):
    """Apply the trained LayerNorm, scalar head, and optional auxiliary heads."""

    def __init__(
        self,
        hidden_size: int,
        auxiliary_features: bool,
        feature_values: Mapping[str, Sequence[str]] | None = None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.score_head = nn.Linear(hidden_size, 1)
        self.auxiliary_features = auxiliary_features
        self.feature_values = {
            name: list(values) for name, values in (feature_values or FEATURE_VALUES).items()
        }
        self.auxiliary_heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_size, len(values))
                for name, values in self.feature_values.items()
            }
            if auxiliary_features
            else {}
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "ExternalPairwiseHead":
        checkpoint = Path(checkpoint_dir)
        head_path = checkpoint / "pairwise_head.pt"
        if not head_path.is_file():
            raise ValueError(f"checkpoint is missing {head_path}")
        config_path = checkpoint / "pairwise_config.json"
        config: dict[str, Any] = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else {}
        )
        state = torch.load(head_path, map_location="cpu")
        if "norm" not in state or "score_head" not in state:
            raise ValueError("pairwise_head.pt must contain norm and score_head")
        hidden_size = int(state["norm"]["weight"].numel())
        auxiliary_features = bool(config.get("auxiliary_features", False))
        if auxiliary_features and "auxiliary_heads" not in state:
            raise ValueError("auxiliary checkpoint is missing auxiliary_heads")
        feature_values = checkpoint_feature_values(config, state)
        head = cls(hidden_size, auxiliary_features, feature_values)
        head.norm.load_state_dict(state["norm"])
        head.score_head.load_state_dict(state["score_head"])
        if auxiliary_features:
            head.auxiliary_heads.load_state_dict(state["auxiliary_heads"])
        return head.to(device=device, dtype=dtype).eval()

    def forward(self, last_hidden_state: torch.Tensor) -> dict[str, Any]:
        if last_hidden_state.ndim != 2:
            raise ValueError(
                "external head expects [batch, hidden_size] LAST-token representations"
            )
        representation = self.norm(last_hidden_state)
        result: dict[str, Any] = {
            "representation": representation,
            "scores": self.score_head(representation).squeeze(-1),
        }
        if self.auxiliary_features:
            result["auxiliary_logits"] = {
                name: head(representation)
                for name, head in self.auxiliary_heads.items()
            }
        return result
