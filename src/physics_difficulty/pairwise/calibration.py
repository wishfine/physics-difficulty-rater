"""Frozen empirical-CDF calibration for scalar Bradley--Terry scores."""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import statistics
from typing import Any, Iterable, Sequence


DIFFICULTY_LEVELS = ("送分题", "基础题", "中等题", "拔高题", "压轴题")
DEFAULT_DISTRIBUTION = (0.20, 0.20, 0.30, 0.20, 0.10)
CALIBRATION_SCHEMA_VERSION = "pairwise_difficulty_calibration_v1"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_distribution(distribution: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in distribution)
    if len(values) != len(DIFFICULTY_LEVELS):
        raise ValueError(f"difficulty distribution must contain {len(DIFFICULTY_LEVELS)} values")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("difficulty distribution values must be finite and positive")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("difficulty distribution must sum to 1")
    return values


def linear_quantile(sorted_values: Sequence[float], quantile: float) -> float:
    """Return the deterministic type-7 linear quantile used by NumPy defaults."""
    if not sorted_values:
        raise ValueError("cannot compute a quantile from an empty sequence")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction)


def build_calibration(
    scores: Iterable[float],
    *,
    calibration_version: str,
    checkpoint_fingerprint: str,
    reference: dict[str, Any],
    distribution: Sequence[float] = DEFAULT_DISTRIBUTION,
) -> dict[str, Any]:
    values = sorted(float(score) for score in scores)
    if not values:
        raise ValueError("reference score collection is empty")
    if any(not math.isfinite(score) for score in values):
        raise ValueError("reference scores must be finite")
    if not str(calibration_version).strip():
        raise ValueError("calibration_version must be non-empty")
    if not str(checkpoint_fingerprint).strip():
        raise ValueError("checkpoint_fingerprint must be non-empty")
    target_distribution = validate_distribution(distribution)
    boundaries: list[float] = []
    cumulative = 0.0
    for probability in target_distribution[:-1]:
        cumulative += probability
        boundaries.append(cumulative)
    thresholds = [linear_quantile(values, boundary) for boundary in boundaries]
    if any(left >= right for left, right in zip(thresholds, thresholds[1:])):
        raise ValueError(
            "reference scores do not produce four strictly increasing thresholds; "
            "the scalar scorer may have collapsed or the reference pool may be too small"
        )
    score_stats = {
        "minimum": values[0],
        "p10": linear_quantile(values, 0.10),
        "p20": linear_quantile(values, 0.20),
        "p40": linear_quantile(values, 0.40),
        "median": linear_quantile(values, 0.50),
        "p70": linear_quantile(values, 0.70),
        "p90": linear_quantile(values, 0.90),
        "maximum": values[-1],
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
    }
    payload: dict[str, Any] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "calibration_version": str(calibration_version),
        "checkpoint_fingerprint": str(checkpoint_fingerprint),
        "levels": list(DIFFICULTY_LEVELS),
        "target_distribution": list(target_distribution),
        "cumulative_boundaries": boundaries,
        "raw_score_thresholds": thresholds,
        "reference": reference,
        "reference_record_count": len(values),
        "reference_score_stats": score_stats,
        # Keeping the exact sorted scores makes the empirical percentile
        # transform deterministic for every future inference batch.
        "reference_scores_sorted": values,
    }
    payload["calibration_id"] = _canonical_sha256(payload)
    return payload


def validate_calibration(calibration: dict[str, Any]) -> None:
    if calibration.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported calibration schema")
    if calibration.get("levels") != list(DIFFICULTY_LEVELS):
        raise ValueError("calibration difficulty levels do not match the frozen schema")
    validate_distribution(calibration.get("target_distribution", []))
    scores = calibration.get("reference_scores_sorted")
    thresholds = calibration.get("raw_score_thresholds")
    if not isinstance(scores, list) or not scores:
        raise ValueError("calibration lacks reference_scores_sorted")
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("calibration reference scores must be finite")
    if any(float(left) > float(right) for left, right in zip(scores, scores[1:])):
        raise ValueError("calibration reference scores must be sorted")
    if not isinstance(thresholds, list) or len(thresholds) != len(DIFFICULTY_LEVELS) - 1:
        raise ValueError("calibration must contain four raw score thresholds")
    if any(float(left) >= float(right) for left, right in zip(thresholds, thresholds[1:])):
        raise ValueError("calibration thresholds must be strictly increasing")
    expected_id = calibration.get("calibration_id")
    unsigned = {key: value for key, value in calibration.items() if key != "calibration_id"}
    if expected_id != _canonical_sha256(unsigned):
        raise ValueError("calibration_id does not match calibration contents")


def apply_calibration(
    raw_score: float,
    calibration: dict[str, Any],
    *,
    calibration_already_validated: bool = False,
) -> dict[str, Any]:
    if not calibration_already_validated:
        validate_calibration(calibration)
    score = float(raw_score)
    if not math.isfinite(score):
        raise ValueError("raw difficulty score must be finite")
    reference_scores = calibration["reference_scores_sorted"]
    percentile = bisect.bisect_right(reference_scores, score) / len(reference_scores)
    thresholds = calibration["raw_score_thresholds"]
    level_id = bisect.bisect_right(thresholds, score)
    return {
        "raw_difficulty_score": score,
        "difficulty_percentile": percentile,
        "difficulty_score": percentile * 100.0,
        "difficulty_level_id": level_id,
        "difficulty_level": DIFFICULTY_LEVELS[level_id],
        "calibration_version": calibration["calibration_version"],
        "calibration_id": calibration["calibration_id"],
    }
