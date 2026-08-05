"""Offline Bradley--Terry auditing for labeled comparison graphs.

This module deliberately ignores question text and auxiliary features.  It
fits one scalar per question and asks whether held-out pair labels can be
explained by the global score differences learned from the remaining edges.
"""
from __future__ import annotations

import hashlib
import math
import random
import sys
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

from physics_difficulty.pairwise.metrics import graph_metrics, soft_pairwise_metrics


def _validated_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen_pair_ids: set[str] = set()
    for index, source in enumerate(rows):
        row = dict(source)
        pair_id = str(row.get("pair_id") or f"pair-{index}")
        left = str(row.get("question_a_id") or "").strip()
        right = str(row.get("question_b_id") or "").strip()
        if not left or not right or left == right:
            raise ValueError(f"pair {pair_id} must contain two different question IDs")
        if pair_id in seen_pair_ids:
            raise ValueError(f"duplicate pair ID: {pair_id}")
        target = float(row["soft_target"])
        weight = float(row.get("sample_weight", 1.0))
        if not 0.0 <= target <= 1.0:
            raise ValueError(f"pair {pair_id} has soft_target outside [0, 1]")
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"pair {pair_id} must have a positive finite sample_weight")
        row.update(
            pair_id=pair_id,
            question_a_id=left,
            question_b_id=right,
            soft_target=target,
            sample_weight=weight,
        )
        seen_pair_ids.add(pair_id)
        validated.append(row)
    if not validated:
        raise ValueError("offline BT audit requires at least one pair")
    return validated


def _question_index(
    rows: Sequence[dict[str, Any]], question_ids: Iterable[str] | None = None
) -> tuple[list[str], dict[str, int]]:
    nodes = {str(value) for value in question_ids or ()}
    for row in rows:
        nodes.add(str(row["question_a_id"]))
        nodes.add(str(row["question_b_id"]))
    ordered = sorted(nodes)
    return ordered, {node: index for index, node in enumerate(ordered)}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _arrays(
    rows: Sequence[dict[str, Any]], index: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left = np.fromiter((index[row["question_a_id"]] for row in rows), dtype=np.int64)
    right = np.fromiter((index[row["question_b_id"]] for row in rows), dtype=np.int64)
    targets = np.fromiter((row["soft_target"] for row in rows), dtype=np.float64)
    weights = np.fromiter((row["sample_weight"] for row in rows), dtype=np.float64)
    return left, right, targets, weights


def fit_bradley_terry(
    rows: Iterable[dict[str, Any]],
    *,
    question_ids: Iterable[str] | None = None,
    max_iterations: int = 600,
    learning_rate: float = 0.08,
    l2: float = 1e-4,
    tolerance: float = 1e-9,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit soft-label Bradley--Terry scores with deterministic full-batch Adam."""
    data = _validated_rows(rows)
    if max_iterations <= 0 or learning_rate <= 0 or l2 < 0:
        raise ValueError("invalid Bradley-Terry optimization settings")
    nodes, index = _question_index(data, question_ids)
    left, right, targets, weights = _arrays(data, index)
    total_weight = float(weights.sum())
    node_count = len(nodes)
    rng = np.random.default_rng(seed)
    scores = rng.normal(0.0, 1e-6, size=node_count)
    first_moment = np.zeros(node_count, dtype=np.float64)
    second_moment = np.zeros(node_count, dtype=np.float64)
    beta1, beta2 = 0.9, 0.999
    epsilon = 1e-8
    previous_loss = math.inf
    stable_iterations = 0
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        predictions = _sigmoid(scores[left] - scores[right])
        edge_gradient = weights * (predictions - targets) / total_weight
        gradient = np.bincount(left, weights=edge_gradient, minlength=node_count)
        gradient -= np.bincount(right, weights=edge_gradient, minlength=node_count)
        gradient += l2 * scores / max(1, node_count)

        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
        corrected_first = first_moment / (1.0 - beta1**iteration)
        corrected_second = second_moment / (1.0 - beta2**iteration)
        scores -= learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)
        scores -= scores.mean()

        clipped = np.clip(predictions, 1e-12, 1.0 - 1e-12)
        loss = float(
            np.sum(
                weights
                * (-(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped)))
            )
            / total_weight
            + 0.5 * l2 * float(np.mean(scores * scores))
        )
        iterations = iteration
        if previous_loss - loss >= 0 and previous_loss - loss < tolerance:
            stable_iterations += 1
            if stable_iterations >= 20:
                break
        else:
            stable_iterations = 0
        previous_loss = loss

    predictions = _sigmoid(scores[left] - scores[right])
    metrics = soft_pairwise_metrics(predictions.tolist(), targets.tolist())
    weighted_metrics = soft_pairwise_metrics(
        predictions.tolist(), targets.tolist(), weights.tolist()
    )
    degrees = np.bincount(left, minlength=node_count) + np.bincount(
        right, minlength=node_count
    )
    weighted_degrees = np.bincount(
        left, weights=weights, minlength=node_count
    ) + np.bincount(right, weights=weights, minlength=node_count)
    information = np.bincount(
        left,
        weights=weights * predictions * (1.0 - predictions),
        minlength=node_count,
    )
    information += np.bincount(
        right,
        weights=weights * predictions * (1.0 - predictions),
        minlength=node_count,
    )
    standard_errors = 1.0 / np.sqrt(np.maximum(information + l2, 1e-12))
    return {
        "scores": {node: float(scores[index[node]]) for node in nodes},
        "standard_errors": {
            node: float(standard_errors[index[node]]) for node in nodes
        },
        "information": {node: float(information[index[node]]) for node in nodes},
        "degrees": {node: int(degrees[index[node]]) for node in nodes},
        "weighted_degrees": {
            node: float(weighted_degrees[index[node]]) for node in nodes
        },
        "predictions": predictions.tolist(),
        "targets": targets.tolist(),
        "metrics": metrics,
        "weighted_metrics": weighted_metrics,
        "final_log_loss": weighted_metrics["soft_pairwise_log_loss"],
        "iterations": iterations,
        "converged": iterations < max_iterations,
        "score_constraint": "mean_zero",
        "uncertainty_method": "diagonal_observed_fisher_approximation",
    }


class _DisjointSet:
    def __init__(self, nodes: Iterable[str]) -> None:
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}

    def find(self, node: str) -> str:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: str, right: str) -> bool:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def _connectivity_backbone(
    data: Sequence[dict[str, Any]], *, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes, _ = _question_index(data)
    disjoint = _DisjointSet(nodes)

    def key(row: dict[str, Any]) -> str:
        payload = f"{seed}:{row['pair_id']}:{row['question_a_id']}:{row['question_b_id']}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    backbone: list[dict[str, Any]] = []
    redundant: list[dict[str, Any]] = []
    for row in sorted(data, key=key):
        if disjoint.union(row["question_a_id"], row["question_b_id"]):
            backbone.append(row)
        else:
            redundant.append(row)
    roots = {disjoint.find(node) for node in nodes}
    if len(roots) != 1:
        raise ValueError(
            f"comparison graph must be connected for offline BT audit; found {len(roots)} components"
        )
    return backbone, redundant


def pair_graph_integrity(
    rows: Iterable[dict[str, Any]],
    *,
    expected_question_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Report graph integrity without hiding missing expected question nodes."""
    data = list(rows)
    expected = (
        {str(question_id) for question_id in expected_question_ids}
        if expected_question_ids is not None
        else {
            str(row.get(key) or "")
            for row in data
            for key in ("question_a_id", "question_b_id")
            if str(row.get(key) or "")
        }
    )
    seen_pair_ids: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()
    duplicate_pair_ids = 0
    duplicate_edges = 0
    self_loops = 0
    endpoints: set[str] = set()
    for index, row in enumerate(data):
        pair_id = str(row.get("pair_id") or f"pair-{index}")
        left = str(row.get("question_a_id") or "").strip()
        right = str(row.get("question_b_id") or "").strip()
        if pair_id in seen_pair_ids:
            duplicate_pair_ids += 1
        seen_pair_ids.add(pair_id)
        if not left or not right:
            continue
        endpoints.update((left, right))
        if left == right:
            self_loops += 1
            continue
        edge = tuple(sorted((left, right)))
        if edge in seen_edges:
            duplicate_edges += 1
        seen_edges.add(edge)
    topology = graph_metrics(data, expected)
    topology.update(
        {
            "pair_records": len(data),
            "unique_pair_ids": len(seen_pair_ids),
            "duplicate_pair_ids": duplicate_pair_ids,
            "duplicate_undirected_edges": duplicate_edges,
            "self_loops": self_loops,
            "missing_expected_nodes": len(expected - endpoints),
            "unknown_question_endpoints": len(endpoints - expected),
        }
    )
    return topology


def graph_connectivity_risks(
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Find articulation nodes and bridge edges in the final simple graph."""
    data = _validated_rows(rows)
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in data:
        left, right = row["question_a_id"], row["question_b_id"]
        adjacency[left].add(right)
        adjacency[right].add(left)

    discovery: dict[str, int] = {}
    low: dict[str, int] = {}
    parent: dict[str, str | None] = {}
    articulation_nodes: set[str] = set()
    bridges: set[tuple[str, str]] = set()
    clock = 0

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 2 * len(adjacency) + 100))

    def visit(node: str) -> None:
        nonlocal clock
        discovery[node] = low[node] = clock
        clock += 1
        children = 0
        for neighbor in sorted(adjacency[node]):
            if neighbor not in discovery:
                parent[neighbor] = node
                children += 1
                visit(neighbor)
                low[node] = min(low[node], low[neighbor])
                if parent[node] is None and children > 1:
                    articulation_nodes.add(node)
                if parent[node] is not None and low[neighbor] >= discovery[node]:
                    articulation_nodes.add(node)
                if low[neighbor] > discovery[node]:
                    bridges.add(tuple(sorted((node, neighbor))))
            elif neighbor != parent[node]:
                low[node] = min(low[node], discovery[neighbor])

    try:
        for root in sorted(adjacency):
            if root not in discovery:
                parent[root] = None
                visit(root)
    finally:
        sys.setrecursionlimit(old_limit)

    ordered_bridges = [list(edge) for edge in sorted(bridges)]
    incident_bridge_counts: dict[str, int] = defaultdict(int)
    for left, right in bridges:
        incident_bridge_counts[left] += 1
        incident_bridge_counts[right] += 1
    return {
        "articulation_node_count": len(articulation_nodes),
        "articulation_nodes": sorted(articulation_nodes),
        "bridge_edge_count": len(bridges),
        "bridge_edges": ordered_bridges,
        "incident_bridge_counts": dict(sorted(incident_bridge_counts.items())),
    }


def connectivity_preserving_bootstrap_sample(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    backbone_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Bootstrap redundant edges while retaining a fixed spanning-tree backbone."""
    data = _validated_rows(rows)
    backbone, redundant = _connectivity_backbone(
        data, seed=seed if backbone_seed is None else backbone_seed
    )
    rng = random.Random(seed)
    sampled: list[tuple[dict[str, Any], bool]] = [
        (row, True) for row in backbone
    ]
    if redundant:
        sampled.extend(
            (redundant[rng.randrange(len(redundant))], False)
            for _ in range(len(redundant))
        )
    return [
        {
            **row,
            "pair_id": f"bootstrap-{seed}-{index}",
            "bootstrap_source_pair_id": row["pair_id"],
            "bootstrap_is_backbone": is_backbone,
        }
        for index, (row, is_backbone) in enumerate(sampled)
    ]


def connectivity_preserving_folds(
    rows: Iterable[dict[str, Any]], *, folds: int = 5, seed: int = 42
) -> dict[str, int]:
    """Return pair_id -> fold, using -1 for a spanning-forest backbone."""
    data = _validated_rows(rows)
    if folds < 2:
        raise ValueError("folds must be at least 2")
    assignment: dict[str, int] = {}
    backbone, redundant = _connectivity_backbone(data, seed=seed)
    for row in backbone:
        assignment[row["pair_id"]] = -1
    if len(redundant) < folds:
        raise ValueError(
            f"comparison graph has only {len(redundant)} redundant edges; cannot create {folds} held-out folds"
        )
    for index, row in enumerate(redundant):
        assignment[row["pair_id"]] = index % folds
    return assignment


def cross_validate_bradley_terry(
    rows: Iterable[dict[str, Any]],
    *,
    folds: int = 5,
    max_iterations: int = 600,
    learning_rate: float = 0.08,
    l2: float = 1e-4,
    seed: int = 42,
) -> dict[str, Any]:
    data = _validated_rows(rows)
    assignment = connectivity_preserving_folds(data, folds=folds, seed=seed)
    all_nodes, _ = _question_index(data)
    heldout_predictions: list[float] = []
    heldout_targets: list[float] = []
    heldout_weights: list[float] = []
    baseline_predictions: list[float] = []
    heldout_records: list[dict[str, Any]] = []
    fold_reports = []
    backbone_edges = sum(value == -1 for value in assignment.values())
    for fold in range(folds):
        fit_rows = [row for row in data if assignment[row["pair_id"]] != fold]
        heldout_rows = [row for row in data if assignment[row["pair_id"]] == fold]
        if not heldout_rows:
            continue
        fitted = fit_bradley_terry(
            fit_rows,
            question_ids=all_nodes,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed + fold,
        )
        scores = fitted["scores"]
        predictions = [
            1.0
            / (
                1.0
                + math.exp(
                    -max(
                        -40.0,
                        min(
                            40.0,
                            scores[row["question_a_id"]]
                            - scores[row["question_b_id"]],
                        ),
                    )
                )
            )
            for row in heldout_rows
        ]
        targets = [row["soft_target"] for row in heldout_rows]
        weights = [row["sample_weight"] for row in heldout_rows]
        fit_weight = sum(row["sample_weight"] for row in fit_rows)
        constant = (
            sum(row["sample_weight"] * row["soft_target"] for row in fit_rows)
            / fit_weight
        )
        heldout_predictions.extend(predictions)
        heldout_targets.extend(targets)
        heldout_weights.extend(weights)
        baseline_predictions.extend([constant] * len(targets))
        heldout_records.extend(
            {
                "pair_id": row["pair_id"],
                "prediction": prediction,
                "baseline_prediction": constant,
                "soft_target": row["soft_target"],
                "sample_weight": row["sample_weight"],
                "pair_source": (row.get("metadata") or {}).get("pair_source")
                or row.get("pair_source"),
                "label_source": row.get("label_source"),
                **_provenance_slice_values(row),
            }
            for row, prediction in zip(heldout_rows, predictions)
        )
        fold_reports.append(
            {
                "fold": fold,
                "fit_pairs": len(fit_rows),
                "heldout_pairs": len(heldout_rows),
                "fit_iterations": fitted["iterations"],
                "heldout_metrics": soft_pairwise_metrics(predictions, targets),
                "heldout_weighted_metrics": soft_pairwise_metrics(
                    predictions, targets, weights
                ),
                "constant_probability": constant,
            }
        )
    return {
        "folds_requested": folds,
        "completed_folds": len(fold_reports),
        "connectivity_backbone_edges": backbone_edges,
        "heldout_pairs": len(heldout_targets),
        "heldout_metrics": soft_pairwise_metrics(
            heldout_predictions, heldout_targets
        ),
        "heldout_weighted_metrics": soft_pairwise_metrics(
            heldout_predictions, heldout_targets, heldout_weights
        ),
        "constant_baseline_metrics": soft_pairwise_metrics(
            baseline_predictions, heldout_targets
        ),
        "constant_baseline_weighted_metrics": soft_pairwise_metrics(
            baseline_predictions, heldout_targets, heldout_weights
        ),
        "heldout_slice_metrics": summarize_prediction_slices(heldout_records),
        "fold_reports": fold_reports,
    }


def summarize_prediction_slices(
    records: Iterable[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Report held-out metrics by provenance without using in-sample predictions."""
    data = list(records)
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for field in (
        "pair_source",
        "label_source",
        "route_reason",
        "reliability_status",
        "position_bias_bucket",
        "feature_distance_bucket",
        "target_confidence_bucket",
    ):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in data:
            groups[str(row.get(field) or "unknown")].append(row)
        output[field] = {}
        for value, group in sorted(groups.items()):
            predictions = [float(row["prediction"]) for row in group]
            baseline = [float(row["baseline_prediction"]) for row in group]
            targets = [float(row["soft_target"]) for row in group]
            weights = [float(row["sample_weight"]) for row in group]
            output[field][value] = {
                "records": len(group),
                "metrics": soft_pairwise_metrics(predictions, targets),
                "weighted_metrics": soft_pairwise_metrics(
                    predictions, targets, weights
                ),
                "constant_baseline_metrics": soft_pairwise_metrics(
                    baseline, targets
                ),
                "constant_baseline_weighted_metrics": soft_pairwise_metrics(
                    baseline, targets, weights
                ),
            }
    return output


def run_negative_controls(
    rows: Iterable[dict[str, Any]],
    *,
    folds: int = 5,
    max_iterations: int = 600,
    learning_rate: float = 0.08,
    l2: float = 1e-4,
    seed: int = 42,
) -> dict[str, Any]:
    """Run deterministic label controls to calibrate whether the audit detects noise."""
    data = _validated_rows(rows)
    rng = random.Random(seed)
    targets = [row["soft_target"] for row in data]
    shuffled_targets = list(targets)
    rng.shuffle(shuffled_targets)
    shuffled = [
        {**row, "soft_target": target}
        for row, target in zip(data, shuffled_targets)
    ]

    flip_count = max(1, round(0.10 * len(data)))
    flipped_indices = set(rng.sample(range(len(data)), k=flip_count))
    flipped = [
        {
            **row,
            "soft_target": 1.0 - row["soft_target"] if index in flipped_indices else row["soft_target"],
        }
        for index, row in enumerate(data)
    ]
    controls = {
        "shuffled_soft_targets": shuffled,
        "flipped_direction_10_percent": flipped,
    }
    return {
        name: {
            "records": len(control_rows),
            "cross_validation": cross_validate_bradley_terry(
                control_rows,
                folds=folds,
                max_iterations=max_iterations,
                learning_rate=learning_rate,
                l2=l2,
                seed=seed + offset + 101,
            ),
        }
        for offset, (name, control_rows) in enumerate(controls.items())
    }


def _ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.arange(len(array), dtype=np.float64)
    return ranks


def bootstrap_rank_stability(
    rows: Iterable[dict[str, Any]],
    reference_scores: dict[str, float],
    *,
    runs: int = 20,
    max_iterations: int = 400,
    learning_rate: float = 0.08,
    l2: float = 1e-4,
    seed: int = 42,
) -> dict[str, Any]:
    data = _validated_rows(rows)
    if runs < 0:
        raise ValueError("bootstrap runs cannot be negative")
    nodes = sorted(reference_scores)
    if not runs:
        return {
            "runs": 0,
            "mean_spearman": None,
            "minimum_spearman": None,
            "mean_top_10_percent_overlap": None,
            "mean_bottom_10_percent_overlap": None,
            "sampling_method": "fixed_spanning_tree_plus_redundant_edge_bootstrap",
            "node_stability": {},
        }
    reference = [reference_scores[node] for node in nodes]
    reference_ranks = _ranks(reference)
    tail_size = max(1, math.ceil(len(nodes) * 0.10))
    reference_order = sorted(nodes, key=reference_scores.get)
    reference_bottom = set(reference_order[:tail_size])
    reference_top = set(reference_order[-tail_size:])
    correlations, top_overlaps, bottom_overlaps = [], [], []
    sampled_scores: list[list[float]] = []
    sampled_ranks: list[list[float]] = []
    sampled_top_sets: list[set[str]] = []
    sampled_bottom_sets: list[set[str]] = []
    for run in range(runs):
        sampled = connectivity_preserving_bootstrap_sample(
            data,
            seed=seed + run + 1,
            backbone_seed=seed,
        )
        fitted = fit_bradley_terry(
            sampled,
            question_ids=nodes,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed + run + 1,
        )
        current = [fitted["scores"][node] for node in nodes]
        current_ranks = _ranks(current)
        correlation = float(np.corrcoef(reference_ranks, current_ranks)[0, 1])
        current_order = sorted(nodes, key=fitted["scores"].get)
        sampled_scores.append(current)
        sampled_ranks.append(current_ranks.tolist())
        sampled_bottom_sets.append(set(current_order[:tail_size]))
        sampled_top_sets.append(set(current_order[-tail_size:]))
        correlations.append(correlation)
        bottom_overlaps.append(
            len(reference_bottom & set(current_order[:tail_size])) / tail_size
        )
        top_overlaps.append(
            len(reference_top & set(current_order[-tail_size:])) / tail_size
        )
    score_samples = np.asarray(sampled_scores, dtype=np.float64)
    rank_samples = np.asarray(sampled_ranks, dtype=np.float64)
    node_stability = {
        node: {
            "score_ci95_low": float(np.quantile(score_samples[:, index], 0.025)),
            "score_ci95_high": float(np.quantile(score_samples[:, index], 0.975)),
            "rank_mean": float(np.mean(rank_samples[:, index]) + 1.0),
            "rank_standard_deviation": float(np.std(rank_samples[:, index])),
            "top_10_percent_frequency": float(
                np.mean([node in current for current in sampled_top_sets])
            ),
            "bottom_10_percent_frequency": float(
                np.mean([node in current for current in sampled_bottom_sets])
            ),
        }
        for index, node in enumerate(nodes)
    }
    return {
        "runs": runs,
        "mean_spearman": float(np.mean(correlations)),
        "minimum_spearman": float(np.min(correlations)),
        "mean_top_10_percent_overlap": float(np.mean(top_overlaps)),
        "mean_bottom_10_percent_overlap": float(np.mean(bottom_overlaps)),
        "sampling_method": "fixed_spanning_tree_plus_redundant_edge_bootstrap",
        "node_stability": node_stability,
    }


def residual_rows(
    rows: Iterable[dict[str, Any]], scores: dict[str, float]
) -> list[dict[str, Any]]:
    data = _validated_rows(rows)
    output = []
    for row in data:
        difference = max(
            -40.0,
            min(
                40.0,
                scores[row["question_a_id"]] - scores[row["question_b_id"]],
            ),
        )
        prediction = 1.0 / (1.0 + math.exp(-difference))
        output.append(
            {
                "pair_id": row["pair_id"],
                "question_a_id": row["question_a_id"],
                "question_b_id": row["question_b_id"],
                "teacher_soft_target": row["soft_target"],
                "bt_probability_a_harder": prediction,
                "absolute_residual": abs(row["soft_target"] - prediction),
                "sample_weight": row["sample_weight"],
                "pair_source": (row.get("metadata") or {}).get("pair_source")
                or row.get("pair_source"),
                "label_source": row.get("label_source"),
                **_provenance_slice_values(row),
            }
        )
    return sorted(output, key=lambda item: item["absolute_residual"], reverse=True)


def summarize_residual_slices(
    residuals: Iterable[dict[str, Any]], *, severe_threshold: float
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Summarize full-fit residual diagnostics by edge provenance."""
    data = list(residuals)
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for field in (
        "pair_source",
        "label_source",
        "route_reason",
        "reliability_status",
        "position_bias_bucket",
        "feature_distance_bucket",
        "target_confidence_bucket",
    ):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in data:
            groups[str(row.get(field) or "unknown")].append(row)
        output[field] = {}
        for value, group in sorted(groups.items()):
            residual_values = np.asarray(
                [float(row["absolute_residual"]) for row in group],
                dtype=np.float64,
            )
            targets = np.clip(
                np.asarray(
                    [float(row["teacher_soft_target"]) for row in group],
                    dtype=np.float64,
                ),
                1e-12,
                1.0 - 1e-12,
            )
            entropies = -(targets * np.log(targets) + (1 - targets) * np.log(1 - targets))
            output[field][value] = {
                "records": len(group),
                "mean_absolute_residual": float(np.mean(residual_values)),
                "p95_absolute_residual": float(np.quantile(residual_values, 0.95)),
                "severe_residual_count": int(
                    np.sum(residual_values >= severe_threshold)
                ),
                "severe_residual_rate": float(
                    np.mean(residual_values >= severe_threshold)
                ),
                "mean_teacher_target_entropy": float(np.mean(entropies)),
                "mean_sample_weight": float(
                    np.mean([float(row["sample_weight"]) for row in group])
                ),
            }
    return output


def _provenance_slice_values(row: dict[str, Any]) -> dict[str, str]:
    metadata = row.get("metadata") or {}
    route = row.get("cascade_route") or {}
    reliability = row.get("reliability") or {}
    vote_stats = row.get("vote_stats") or {}

    route_reason = (
        route.get("route_reason")
        or route.get("reason")
        or route.get("action")
        or "unknown"
    )
    reliability_status = reliability.get("status") or "unknown"

    gap = vote_stats.get("position_bias_gap")
    if gap is None:
        gap = reliability.get("position_bias_gap")
    if gap is None:
        position_bias_bucket = "unknown"
    elif float(gap) <= 0.25:
        position_bias_bucket = "none_or_low"
    else:
        position_bias_bucket = "high"

    distance = metadata.get("feature_hamming_distance")
    if distance is None:
        feature_distance_bucket = "unknown"
    elif float(distance) == 0:
        feature_distance_bucket = "0"
    elif float(distance) <= 3:
        feature_distance_bucket = "1-3"
    elif float(distance) <= 7:
        feature_distance_bucket = "4-7"
    else:
        feature_distance_bucket = "8+"

    target_gap = abs(float(row["soft_target"]) - 0.5)
    if target_gap < 0.10:
        confidence_bucket = "near_tie"
    elif target_gap < 0.20:
        confidence_bucket = "uncertain"
    else:
        confidence_bucket = "decisive"
    return {
        "route_reason": str(route_reason),
        "reliability_status": str(reliability_status),
        "position_bias_bucket": position_bias_bucket,
        "feature_distance_bucket": feature_distance_bucket,
        "target_confidence_bucket": confidence_bucket,
    }


def distribution_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }
