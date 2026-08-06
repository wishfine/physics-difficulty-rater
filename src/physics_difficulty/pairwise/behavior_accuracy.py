"""Utilities for turning aggregated answer accuracy into difficulty evidence.

The upstream ``difficulty`` field is deliberately outside this module's data
contract.  Only observed answer counts and percent-correct values are used.
"""
from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Iterable, Sequence


PERCENT_QUANTUM = Decimal("0.01")


def parse_answered_count(value: Any) -> int:
    """Parse a strictly positive integer response count."""
    if isinstance(value, bool):
        raise ValueError("answered_count cannot be boolean")
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, AttributeError) as exc:
        raise ValueError(f"invalid answered_count: {value!r}") from exc
    if not parsed.is_finite() or parsed != parsed.to_integral_value() or parsed <= 0:
        raise ValueError(f"answered_count must be a positive integer: {value!r}")
    return int(parsed)


def parse_percent_correct(value: Any) -> Decimal:
    """Parse a percent in the closed interval [0, 100]."""
    text = str(value).strip().removesuffix("%").strip()
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"invalid percent_correct: {value!r}") from exc
    if not parsed.is_finite() or parsed < 0 or parsed > 100:
        raise ValueError(f"percent_correct must be in [0, 100]: {value!r}")
    return parsed


def recover_correct_count(answered_count: int, percent_correct: Decimal) -> dict[str, Any]:
    """Recover an integer correct count from a percentage rounded to 2 decimals.

    If more than one count can produce the reported percentage, the closest
    candidate is used and the ambiguity is explicitly reported.
    """
    if answered_count <= 0:
        raise ValueError("answered_count must be positive")
    target = percent_correct.quantize(PERCENT_QUANTUM, rounding=ROUND_HALF_UP)
    center = Decimal(answered_count) * percent_correct / Decimal(100)
    floor_center = int(center.to_integral_value(rounding="ROUND_FLOOR"))
    search_radius = max(3, math.ceil(answered_count / 20000) + 2)
    candidates = []
    for correct in range(
        max(0, floor_center - search_radius),
        min(answered_count, floor_center + search_radius) + 1,
    ):
        rendered = (
            Decimal(100) * Decimal(correct) / Decimal(answered_count)
        ).quantize(PERCENT_QUANTUM, rounding=ROUND_HALF_UP)
        if rendered == target:
            candidates.append(correct)

    if candidates:
        correct = min(candidates, key=lambda item: (abs(Decimal(item) - center), item))
        status = "rounded_exact_unique" if len(candidates) == 1 else "rounded_exact_ambiguous"
        effective_correct_count = float(correct)
        evidence_type = "integer_recovered"
        evidence_quality = 1.0 if len(candidates) == 1 else 0.85
    else:
        correct = min(
            range(max(0, floor_center - 2), min(answered_count, floor_center + 2) + 1),
            key=lambda item: (abs(Decimal(item) - center), item),
        )
        status = "continuous_rate_pseudocount"
        effective_correct_count = float(center)
        evidence_type = "continuous_rate_pseudocount"
        # The reported rate is useful evidence, but may include partial credit
        # or a denominator different from a literal correct/incorrect count.
        evidence_quality = 0.5
    reconstructed = float(Decimal(100) * Decimal(correct) / Decimal(answered_count))
    return {
        "correct_count": correct if evidence_type == "integer_recovered" else None,
        "incorrect_count": (
            answered_count - correct if evidence_type == "integer_recovered" else None
        ),
        "integer_correct_count_estimate": correct,
        "integer_incorrect_count_estimate": answered_count - correct,
        "effective_correct_count": effective_correct_count,
        "effective_incorrect_count": answered_count - effective_correct_count,
        "behavior_evidence_type": evidence_type,
        "behavior_evidence_quality": evidence_quality,
        "recovery_status": status,
        "matching_integer_count_candidate_count": len(candidates),
        "matching_integer_count_candidates_sample": candidates[:20],
        "reconstructed_percent_correct": reconstructed,
        "absolute_percent_reconstruction_error": abs(reconstructed - float(percent_correct)),
    }


def _logit(value: float, epsilon: float = 1e-12) -> float:
    clipped = min(1.0 - epsilon, max(epsilon, value))
    return math.log(clipped / (1.0 - clipped))


def beta_posterior_summary(
    answered_count: int,
    correct_count: float,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> dict[str, float | str]:
    """Return posterior accuracy and a higher-is-harder difficulty logit."""
    if answered_count <= 0 or not 0 <= correct_count <= answered_count:
        raise ValueError("invalid answered/correct counts")
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("Beta prior parameters must be positive")
    alpha = prior_alpha + correct_count
    beta = prior_beta + answered_count - correct_count
    total = alpha + beta
    mean = alpha / total
    variance = alpha * beta / (total * total * (total + 1.0))
    std = math.sqrt(variance)
    lower = max(0.0, mean - 1.959963984540054 * std)
    upper = min(1.0, mean + 1.959963984540054 * std)
    return {
        "posterior_alpha": alpha,
        "posterior_beta": beta,
        "posterior_correct_rate_mean": mean,
        "posterior_correct_rate_standard_deviation": std,
        "posterior_correct_rate_lower_95": lower,
        "posterior_correct_rate_upper_95": upper,
        "behavior_difficulty_score": -_logit(mean),
        "behavior_difficulty_lower_95": -_logit(upper),
        "behavior_difficulty_upper_95": -_logit(lower),
        "interval_method": "normal_approximation_to_beta_posterior",
    }


def score_behavior_row(
    row: dict[str, Any],
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> dict[str, Any]:
    """Convert one behavior row into a label-free difficulty evidence row."""
    question_id = str(row.get("question_id") or "").strip()
    if not question_id:
        raise ValueError("missing question_id")
    parent_id = str(row.get("parent_id") or question_id).strip()
    answered = parse_answered_count(row.get("answered_count"))
    percent = parse_percent_correct(row.get("percent_correct"))
    recovered = recover_correct_count(answered, percent)
    posterior = beta_posterior_summary(
        answered,
        float(recovered["effective_correct_count"]),
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
    )
    stem = str(row.get("stem") or "").strip()
    if not stem:
        raise ValueError("missing or empty stem")
    sub_questions = row.get("sub_questions")
    if sub_questions is None:
        sub_questions = []
    if not isinstance(sub_questions, list):
        raise ValueError("sub_questions must be a list")
    output = {
        "schema_version": "behavior_accuracy_score_v2",
        "question_id": question_id,
        "parent_id": parent_id,
        "structure_type": str(row.get("structure_type") or "unknown"),
        "answered_count": answered,
        "reported_percent_correct": float(percent),
        **recovered,
        "beta_prior_alpha": prior_alpha,
        "beta_prior_beta": prior_beta,
        **posterior,
        "has_subquestions": bool(sub_questions),
        "subquestion_count": len(sub_questions),
        "stem_sha256": hashlib.sha256(stem.encode("utf-8")).hexdigest(),
        "forbidden_source_difficulty_used": False,
    }
    return output


def row_fingerprint(row: dict[str, Any]) -> str:
    """Fingerprint behavior evidence while explicitly excluding difficulty."""
    payload = {
        key: row.get(key)
        for key in (
            "parent_id",
            "question_id",
            "stem",
            "options",
            "analysis",
            "structure_type",
            "answered_count",
            "percent_correct",
            "sub_questions",
        )
    }
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def behavior_pair_probability(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Approximate P(question A is harder than question B).

    Harder means a lower latent correctness rate. Independent beta posteriors
    are approximated as normal distributions for an efficient 40k-pair audit.
    """
    mean_a = float(a["posterior_correct_rate_mean"])
    mean_b = float(b["posterior_correct_rate_mean"])
    std_a = float(a["posterior_correct_rate_standard_deviation"])
    std_b = float(b["posterior_correct_rate_standard_deviation"])
    denominator = math.sqrt(std_a * std_a + std_b * std_b)
    if denominator <= 0:
        return 1.0 if mean_a < mean_b else 0.0 if mean_a > mean_b else 0.5
    return min(1.0, max(0.0, normal_cdf((mean_b - mean_a) / denominator)))


def entropy_confidence(probability: float) -> float:
    """Map a Bernoulli probability to [0,1] confidence using entropy."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if probability in (0.0, 1.0):
        return 1.0
    entropy = -probability * math.log(probability) - (1 - probability) * math.log(1 - probability)
    return 1.0 - entropy / math.log(2.0)


def harmonic_mean(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        return 0.0
    return 2.0 / (1.0 / left + 1.0 / right)


def distribution_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {"count": 0, "minimum": None, "p10": None, "median": None, "mean": None, "p90": None, "maximum": None}

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)

    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "p10": percentile(0.10),
        "median": percentile(0.50),
        "mean": sum(ordered) / len(ordered),
        "p90": percentile(0.90),
        "maximum": ordered[-1],
    }


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - mean_left) ** 2 for a in left)
        * sum((b - mean_right) ** 2 for b in right)
    )
    return numerator / denominator if denominator else None


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return pearson_correlation(average_ranks(left), average_ranks(right))
