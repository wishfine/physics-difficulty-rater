"""Dependency-light metrics for soft pairwise labels and comparison graphs."""
from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from typing import Any, Dict, Iterable, Sequence


def soft_pairwise_metrics(predictions: Sequence[float], targets: Sequence[float]) -> Dict[str, Any]:
    if len(predictions) != len(targets) or not targets:
        raise ValueError("predictions and non-empty targets must have equal length")
    eps = 1e-12
    log_loss = 0.0
    brier = 0.0
    hard_correct = 0
    hard_count = 0
    decisive_correct = 0
    decisive_count = 0
    for prediction, target in zip(predictions, targets):
        if not 0 <= prediction <= 1 or not 0 <= target <= 1:
            raise ValueError("pairwise probabilities must be in [0, 1]")
        clipped = min(1 - eps, max(eps, prediction))
        log_loss -= target * math.log(clipped) + (1 - target) * math.log(1 - clipped)
        brier += (prediction - target) ** 2
        if target != 0.5:
            hard_count += 1
            hard_correct += (prediction >= 0.5) == (target > 0.5)
        if abs(target - 0.5) >= 0.2:
            decisive_count += 1
            decisive_correct += (prediction >= 0.5) == (target >= 0.5)
    count = len(targets)
    hard_labels = [int(target > 0.5) for target in targets if target != 0.5]
    hard_scores = [prediction for prediction, target in zip(predictions, targets) if target != 0.5]
    positives = sum(hard_labels)
    negatives = len(hard_labels) - positives
    auc = 0.0
    if positives and negatives:
        order = sorted(range(len(hard_scores)), key=lambda index: hard_scores[index])
        ranks = [0.0] * len(order)
        cursor = 0
        while cursor < len(order):
            end = cursor + 1
            while end < len(order) and hard_scores[order[end]] == hard_scores[order[cursor]]:
                end += 1
            average_rank = (cursor + 1 + end) / 2
            for position in range(cursor, end):
                ranks[order[position]] = average_rank
            cursor = end
        positive_rank_sum = sum(rank for rank, label in zip(ranks, hard_labels) if label)
        auc = (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return {
        "soft_pairwise_log_loss": log_loss / count,
        "brier_score": brier / count,
        "pairwise_accuracy": hard_correct / hard_count if hard_count else 0.0,
        "non_tied_pair_count": hard_count,
        "pairwise_auc": auc,
        "auc_status": "OK" if positives and negatives else "UNDEFINED_SINGLE_CLASS",
        "decisive_pairwise_accuracy": decisive_correct / decisive_count if decisive_count else 0.0,
        "decisive_pair_count": decisive_count,
    }


def graph_metrics(pairs: Iterable[Dict[str, Any]], expected_nodes: Iterable[str] | None = None) -> Dict[str, Any]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    edges: set[tuple[str, str]] = set()
    for pair in pairs:
        left, right = str(pair["question_a_id"]), str(pair["question_b_id"])
        if left == right:
            continue
        edge = tuple(sorted((left, right)))
        edges.add(edge)
        adjacency[left].add(right)
        adjacency[right].add(left)
    nodes = set(adjacency)
    expected = set(map(str, expected_nodes)) if expected_nodes is not None else nodes
    all_nodes = nodes | expected
    components = []
    unseen = set(nodes)
    while unseen:
        root = next(iter(unseen))
        queue = deque([root])
        unseen.remove(root)
        size = 0
        while queue:
            node = queue.popleft()
            size += 1
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(size)
    degrees = sorted((len(adjacency.get(node, set())) for node in all_nodes))

    def percentile(values: list[int], fraction: float) -> float:
        if not values:
            return 0.0
        position = fraction * (len(values) - 1)
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return float(values[lower])
        return values[lower] * (upper - position) + values[upper] * (position - lower)

    largest = max(components, default=0)
    return {
        "expected_nodes": len(expected),
        "covered_nodes": len(nodes & expected),
        "node_coverage": len(nodes & expected) / max(1, len(expected)),
        "unique_edges": len(edges),
        "connected_components": len(components) + len(expected - nodes),
        "largest_component_nodes": largest,
        "largest_component_ratio": largest / max(1, len(expected)),
        "degree": {
            "minimum": min(degrees, default=0),
            "p10": percentile(degrees, 0.10),
            "median": percentile(degrees, 0.50),
            "mean": sum(degrees) / max(1, len(degrees)),
            "p90": percentile(degrees, 0.90),
            "maximum": max(degrees, default=0),
            "zero_degree_nodes": sum(value == 0 for value in degrees),
        },
    }


def soft_target_distribution(values: Iterable[float]) -> Dict[str, Any]:
    buckets = Counter()
    entropies = []
    gaps = []
    for value in values:
        if not 0 <= value <= 1:
            raise ValueError("soft target must be in [0, 1]")
        if value < 0.1:
            bucket = "very_clear_b"
        elif value < 0.3:
            bucket = "clear_b"
        elif value <= 0.7:
            bucket = "uncertain"
        elif value <= 0.9:
            bucket = "clear_a"
        else:
            bucket = "very_clear_a"
        buckets[bucket] += 1
        clipped = min(1 - 1e-12, max(1e-12, value))
        entropies.append(-(clipped * math.log(clipped) + (1 - clipped) * math.log(1 - clipped)))
        gaps.append(abs(value - 0.5))
    return {
        "buckets": dict(buckets),
        "mean_entropy": sum(entropies) / max(1, len(entropies)),
        "mean_distance_from_half": sum(gaps) / max(1, len(gaps)),
    }
