"""Dependency-free metrics for five ordered difficulty classes."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List


def classification_metrics(predictions: List[int], labels: List[int], class_count: int = 5) -> Dict[str, float | list]:
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for prediction, label in zip(predictions, labels):
        matrix[label][prediction] += 1
    supports = [sum(row) for row in matrix]
    recalls, f1s = [], []
    for index in range(class_count):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(class_count)) - tp
        fn = sum(matrix[index]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        if supports[index]:
            recalls.append(recall)
    total = len(labels)
    observed = sum(matrix[index][index] for index in range(class_count)) / max(1, total)
    weighted_observed = sum(((row - column) ** 2 / max(1, (class_count - 1) ** 2)) * matrix[row][column] for row in range(class_count) for column in range(class_count)) / max(1, total)
    row_marginals = [sum(row) for row in matrix]
    col_marginals = [sum(matrix[row][column] for row in range(class_count)) for column in range(class_count)]
    weighted_expected = sum(((row - column) ** 2 / max(1, (class_count - 1) ** 2)) * row_marginals[row] * col_marginals[column] for row in range(class_count) for column in range(class_count)) / max(1, total * total)
    return {
        "accuracy": observed,
        "macro_f1": sum(f1s) / class_count,
        "balanced_accuracy": sum(recalls) / max(1, len(recalls)),
        "mean_absolute_error": sum(abs(prediction - label) for prediction, label in zip(predictions, labels)) / max(1, total),
        "adjacent_accuracy": sum(abs(prediction - label) <= 1 for prediction, label in zip(predictions, labels)) / max(1, total),
        "quadratic_weighted_kappa": 1 - weighted_observed / weighted_expected if weighted_expected else 0.0,
        "confusion_matrix": matrix,
    }


def calibration_metrics(probabilities: List[List[float]], labels: List[int], bins: int = 10) -> Dict[str, float]:
    nll = -sum(math.log(max(1e-12, probabilities[index][label])) for index, label in enumerate(labels)) / max(1, len(labels))
    ece = 0.0
    for bucket in range(bins):
        lower, upper = bucket / bins, (bucket + 1) / bins
        indices = [index for index, probability in enumerate(probabilities) if lower <= max(probability) < upper or (bucket == bins - 1 and max(probability) == 1.0)]
        if indices:
            confidence = sum(max(probabilities[index]) for index in indices) / len(indices)
            accuracy = sum(max(range(len(probabilities[index])), key=lambda class_id: probabilities[index][class_id]) == labels[index] for index in indices) / len(indices)
            ece += len(indices) / len(labels) * abs(confidence - accuracy)
    return {"negative_log_likelihood": nll, "expected_calibration_error": ece}
