"""Position-corrected soft preference targets from local-teacher votes."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable


def smoothed_probability(wins: int, total: int, prior: float = 0.5) -> float:
    """Return a symmetric Beta-prior posterior mean.

    ``prior=0.5`` is Jeffreys smoothing.  It prevents tiny vote batches from
    producing exact zero/one labels while retaining the judge's direction.
    """
    if total <= 0:
        raise ValueError("total votes must be positive")
    if not 0 <= wins <= total:
        raise ValueError("wins must be between zero and total")
    if prior < 0:
        raise ValueError("prior must be non-negative")
    return (wins + prior) / (total + 2 * prior)


def aggregate_pair_votes(votes: Iterable[Dict[str, Any]], prior: float = 0.5) -> Dict[str, Any]:
    """Aggregate valid forward/backward votes by real question identity.

    Every raw vote stores ``winner_question_id``.  This deliberately avoids
    treating the positional labels A/B as stable identities after reversal.
    """
    rows = [row for row in votes if row.get("valid", True)]
    if not rows:
        raise ValueError("pair has no valid votes")
    pair_ids = {str(row["pair_id"]) for row in rows}
    question_a_ids = {str(row["question_a_id"]) for row in rows}
    question_b_ids = {str(row["question_b_id"]) for row in rows}
    if len(pair_ids) != 1 or len(question_a_ids) != 1 or len(question_b_ids) != 1:
        raise ValueError("votes from multiple pairs cannot be aggregated together")
    question_a_id = next(iter(question_a_ids))
    question_b_id = next(iter(question_b_ids))
    directions: dict[str, list[Dict[str, Any]]] = {"forward": [], "backward": []}
    for row in rows:
        direction = str(row.get("direction"))
        if direction not in directions:
            raise ValueError(f"unknown vote direction: {direction}")
        if str(row.get("winner_question_id")) not in {question_a_id, question_b_id}:
            raise ValueError("valid vote winner must be one of the compared question IDs")
        directions[direction].append(row)
    if not directions["forward"] or not directions["backward"]:
        raise ValueError("both forward and backward votes are required")

    stats: Dict[str, Any] = {}
    probabilities = []
    for direction, direction_rows in directions.items():
        a_wins = sum(str(row.get("winner_question_id")) == question_a_id for row in direction_rows)
        total = len(direction_rows)
        probability = smoothed_probability(a_wins, total, prior)
        stats[f"{direction}_votes"] = total
        stats[f"{direction}_question_a_wins"] = a_wins
        stats[f"{direction}_probability_a_harder"] = probability
        probabilities.append(probability)
    stats["soft_target"] = sum(probabilities) / 2
    stats["position_bias_gap"] = abs(probabilities[0] - probabilities[1])
    stats["valid_votes"] = sum(len(value) for value in directions.values())
    stats["raw_output_counts"] = dict(Counter(str(row.get("raw_output", "")) for row in rows))
    return stats


def pair_reliability(position_bias_gap: float, medium_threshold: float = 0.15, high_threshold: float = 0.30) -> Dict[str, Any]:
    """Map order sensitivity to a transparent training action and weight."""
    if not 0 <= position_bias_gap <= 1:
        raise ValueError("position_bias_gap must be in [0, 1]")
    if not 0 <= medium_threshold < high_threshold <= 1:
        raise ValueError("reliability thresholds must satisfy 0 <= medium < high <= 1")
    if position_bias_gap <= medium_threshold:
        return {"status": "stable", "sample_weight": 1.0, "action": "keep"}
    if position_bias_gap <= high_threshold:
        return {"status": "order_sensitive", "sample_weight": 0.5, "action": "keep_downweighted"}
    return {"status": "unstable", "sample_weight": 0.0, "action": "quarantine"}
