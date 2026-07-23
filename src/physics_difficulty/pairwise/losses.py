"""Loss helpers for Bradley-Terry training with optional auxiliary supervision."""
from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F


def auxiliary_loss_weight(step: int, total_steps: int, maximum: float = 0.1, warmup_ratio: float = 0.1) -> float:
    if maximum < 0 or not 0 <= warmup_ratio <= 1:
        raise ValueError("invalid auxiliary loss schedule")
    warmup_steps = max(1, math.ceil(total_steps * warmup_ratio))
    return maximum * min(max(step, 0) / warmup_steps, 1.0)


def _weighted_feature_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    example_weights: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor | None:
    valid = targets.ge(0) & example_weights.gt(0)
    if not bool(valid.any()):
        return None
    losses = F.cross_entropy(logits[valid].float(), targets[valid], weight=class_weights.float(), reduction="none")
    weights = example_weights[valid].float()
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def normalized_auxiliary_loss(
    logits_a: Mapping[str, torch.Tensor],
    logits_b: Mapping[str, torch.Tensor],
    targets_a: Mapping[str, torch.Tensor],
    targets_b: Mapping[str, torch.Tensor],
    weights_a: torch.Tensor,
    weights_b: torch.Tensor,
    class_weights: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    normalized = []
    for name in logits_a:
        class_count = logits_a[name].shape[-1]
        logits = torch.cat((logits_a[name], logits_b[name]), dim=0)
        targets = torch.cat((targets_a[name], targets_b[name]), dim=0)
        weights = torch.cat((weights_a, weights_b), dim=0)
        loss = _weighted_feature_ce(logits, targets, weights, class_weights[name].to(logits.device))
        if loss is not None:
            normalized.append(loss / math.log(class_count))
    if not normalized:
        raise ValueError("batch contains no valid auxiliary labels")
    return torch.stack(normalized).mean()
