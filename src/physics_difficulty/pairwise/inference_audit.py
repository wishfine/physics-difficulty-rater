"""Deterministic helpers for single-question score comparison and calibration audits."""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Iterable, Sequence

from physics_difficulty.pairwise.calibration import linear_quantile, validate_distribution


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = rank
        cursor = end
    return ranks


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("correlation inputs must be non-empty and equally sized")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    return numerator / (left_scale * right_scale) if left_scale and right_scale else None


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return pearson(average_ranks(left), average_ranks(right))


def thresholds(values: Sequence[float], distribution: Sequence[float]) -> list[float]:
    distribution = validate_distribution(distribution)
    ordered = sorted(float(value) for value in values)
    boundaries = []
    cumulative = 0.0
    for probability in distribution[:-1]:
        cumulative += probability
        boundaries.append(linear_quantile(ordered, cumulative))
    return boundaries


def level_id(score: float, boundaries: Sequence[float]) -> int:
    return sum(float(score) >= float(boundary) for boundary in boundaries)


def bootstrap_thresholds(
    values: Sequence[float],
    distribution: Sequence[float],
    *,
    repetitions: int,
    seed: int,
) -> list[list[float]]:
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    if not values:
        raise ValueError("cannot bootstrap an empty score collection")
    generator = random.Random(seed)
    return [
        thresholds([values[generator.randrange(len(values))] for _ in values], distribution)
        for _ in range(repetitions)
    ]


def percentile_interval(values: Iterable[float], low: float = 0.025, high: float = 0.975) -> list[float]:
    ordered = sorted(float(value) for value in values)
    return [linear_quantile(ordered, low), linear_quantile(ordered, high)]


def migration_summary(
    scores: Sequence[float], reference_thresholds: Sequence[float], candidate_thresholds: Sequence[float]
) -> dict[str, object]:
    pairs = [
        (level_id(score, reference_thresholds), level_id(score, candidate_thresholds))
        for score in scores
    ]
    matrix = [[0 for _ in range(5)] for _ in range(5)]
    for source, target in pairs:
        matrix[source][target] += 1
    changed = sum(source != target for source, target in pairs)
    return {
        "records": len(pairs),
        "agreement": 1.0 - changed / len(pairs) if pairs else None,
        "changed_records": changed,
        "confusion_matrix": matrix,
    }


def top_bottom_overlap(left: Sequence[float], right: Sequence[float], fraction: float = 0.10) -> dict[str, float | int]:
    if len(left) != len(right) or not left:
        raise ValueError("overlap inputs must be non-empty and equally sized")
    count = max(1, round(len(left) * fraction))
    left_order = sorted(range(len(left)), key=lambda index: (left[index], index))
    right_order = sorted(range(len(right)), key=lambda index: (right[index], index))
    bottom = len(set(left_order[:count]) & set(right_order[:count])) / count
    top = len(set(left_order[-count:]) & set(right_order[-count:])) / count
    return {"set_size": count, "bottom_overlap": bottom, "top_overlap": top}
