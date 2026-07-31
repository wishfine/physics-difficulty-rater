"""Offline Bradley--Terry auditing for labeled comparison graphs.

This module deliberately ignores question text and auxiliary features.  It
fits one scalar per question and asks whether held-out pair labels can be
explained by the global score differences learned from the remaining edges.
"""
from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

from physics_difficulty.pairwise.metrics import soft_pairwise_metrics


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
        "predictions": predictions.tolist(),
        "targets": targets.tolist(),
        "metrics": metrics,
        "final_log_loss": metrics["soft_pairwise_log_loss"],
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


def connectivity_preserving_folds(
    rows: Iterable[dict[str, Any]], *, folds: int = 5, seed: int = 42
) -> dict[str, int]:
    """Return pair_id -> fold, using -1 for a spanning-forest backbone."""
    data = _validated_rows(rows)
    if folds < 2:
        raise ValueError("folds must be at least 2")
    nodes, _ = _question_index(data)
    disjoint = _DisjointSet(nodes)

    def key(row: dict[str, Any]) -> str:
        payload = f"{seed}:{row['pair_id']}:{row['question_a_id']}:{row['question_b_id']}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    assignment: dict[str, int] = {}
    redundant: list[dict[str, Any]] = []
    for row in sorted(data, key=key):
        if disjoint.union(row["question_a_id"], row["question_b_id"]):
            assignment[row["pair_id"]] = -1
        else:
            redundant.append(row)
    roots = {disjoint.find(node) for node in nodes}
    if len(roots) != 1:
        raise ValueError(
            f"comparison graph must be connected for offline BT audit; found {len(roots)} components"
        )
    if len(redundant) < folds:
        raise ValueError(
            f"comparison graph has only {len(redundant)} redundant edges; cannot create {folds} held-out folds"
        )
    for index, row in enumerate(sorted(redundant, key=key)):
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
    baseline_predictions: list[float] = []
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
        fit_weight = sum(row["sample_weight"] for row in fit_rows)
        constant = (
            sum(row["sample_weight"] * row["soft_target"] for row in fit_rows)
            / fit_weight
        )
        heldout_predictions.extend(predictions)
        heldout_targets.extend(targets)
        baseline_predictions.extend([constant] * len(targets))
        fold_reports.append(
            {
                "fold": fold,
                "fit_pairs": len(fit_rows),
                "heldout_pairs": len(heldout_rows),
                "fit_iterations": fitted["iterations"],
                "heldout_metrics": soft_pairwise_metrics(predictions, targets),
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
        "constant_baseline_metrics": soft_pairwise_metrics(
            baseline_predictions, heldout_targets
        ),
        "fold_reports": fold_reports,
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
        }
    reference = [reference_scores[node] for node in nodes]
    reference_ranks = _ranks(reference)
    tail_size = max(1, math.ceil(len(nodes) * 0.10))
    reference_order = sorted(nodes, key=reference_scores.get)
    reference_bottom = set(reference_order[:tail_size])
    reference_top = set(reference_order[-tail_size:])
    correlations, top_overlaps, bottom_overlaps = [], [], []
    rng = random.Random(seed)
    for run in range(runs):
        sampled = [data[rng.randrange(len(data))] for _ in data]
        # Bootstrap duplicates need unique IDs for validation; the statistical
        # weight is represented by repeated rows.
        sampled = [
            {**row, "pair_id": f"bootstrap-{run}-{index}"}
            for index, row in enumerate(sampled)
        ]
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
        correlations.append(correlation)
        bottom_overlaps.append(
            len(reference_bottom & set(current_order[:tail_size])) / tail_size
        )
        top_overlaps.append(
            len(reference_top & set(current_order[-tail_size:])) / tail_size
        )
    return {
        "runs": runs,
        "mean_spearman": float(np.mean(correlations)),
        "minimum_spearman": float(np.min(correlations)),
        "mean_top_10_percent_overlap": float(np.mean(top_overlaps)),
        "mean_bottom_10_percent_overlap": float(np.mean(bottom_overlaps)),
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
            }
        )
    return sorted(output, key=lambda item: item["absolute_residual"], reverse=True)


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
