"""Feature-aware node selection and coverage reports for pair construction."""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

from physics_difficulty.schema import FEATURE_VALUES


def validate_feature_map(
    feature_rows: Iterable[dict[str, Any]],
    *,
    allowed_question_ids: Iterable[str] | None = None,
) -> dict[str, dict[str, str]]:
    allowed = set(map(str, allowed_question_ids)) if allowed_question_ids is not None else None
    result: dict[str, dict[str, str]] = {}
    for row in feature_rows:
        question_id = str(row.get("id") or row.get("question_id") or "").strip()
        if not question_id or (allowed is not None and question_id not in allowed):
            continue
        if question_id in result:
            raise ValueError(f"duplicate auxiliary feature ID: {question_id}")
        source = row.get("teacher_features")
        if not isinstance(source, dict):
            raise ValueError(f"question {question_id} is missing teacher_features")
        normalized: dict[str, str] = {}
        for name, values in FEATURE_VALUES.items():
            value = str(source.get(name) or "")
            if value not in values:
                raise ValueError(
                    f"question {question_id} has invalid auxiliary feature {name}={value!r}"
                )
            normalized[name] = value
        result[question_id] = normalized
    if allowed is not None:
        missing = allowed - set(result)
        if missing:
            raise ValueError(
                f"auxiliary feature file is missing {len(missing)} candidate questions"
            )
    return result


def _stable_key(seed: int, question_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{question_id}".encode("utf-8")).hexdigest()


def select_feature_balanced_ids(
    question_ids: Sequence[str],
    features: dict[str, dict[str, str]],
    *,
    target_count: int,
    seed: int = 42,
) -> list[str]:
    """Select nodes while preserving every category and approximate marginals."""
    ids = [str(value) for value in question_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("question IDs must be unique")
    if not 1 <= target_count <= len(ids):
        raise ValueError("target_count must be within the question pool")
    missing = set(ids) - set(features)
    if missing:
        raise ValueError(f"missing auxiliary features for {len(missing)} questions")

    category_members: dict[tuple[str, str], list[str]] = defaultdict(list)
    for question_id in ids:
        for name in FEATURE_VALUES:
            category_members[(name, features[question_id][name])].append(question_id)
    for category in category_members:
        category_members[category].sort(key=lambda value: _stable_key(seed, value))

    selected: set[str] = set()
    selected_counts: Counter[tuple[str, str]] = Counter()

    # A short weighted set-cover pass protects rare categories from vanishing.
    uncovered = set(category_members)
    while uncovered and len(selected) < target_count:
        candidates = {
            question_id
            for category in uncovered
            for question_id in category_members[category][: min(32, len(category_members[category]))]
            if question_id not in selected
        }
        if not candidates:
            break

        def coverage_score(question_id: str) -> tuple[float, str]:
            score = 0.0
            for name in FEATURE_VALUES:
                category = (name, features[question_id][name])
                if category in uncovered:
                    score += 1.0 / len(category_members[category])
            return score, _stable_key(seed + len(selected), question_id)

        chosen = max(candidates, key=coverage_score)
        selected.add(chosen)
        for name in FEATURE_VALUES:
            category = (name, features[chosen][name])
            selected_counts[category] += 1
            uncovered.discard(category)

    target_quota = {
        category: target_count * len(members) / len(ids)
        for category, members in category_members.items()
    }
    cursors: Counter[tuple[str, str]] = Counter()
    while len(selected) < target_count:
        deficits = sorted(
            category_members,
            key=lambda category: (
                -(
                    target_quota[category] - selected_counts[category]
                )
                / max(1.0, target_quota[category]),
                category,
            ),
        )
        chosen = None
        for category in deficits:
            members = category_members[category]
            cursor = cursors[category]
            while cursor < len(members) and members[cursor] in selected:
                cursor += 1
            cursors[category] = cursor
            if cursor < len(members):
                chosen = members[cursor]
                cursors[category] += 1
                break
        if chosen is None:
            remaining = sorted(set(ids) - selected, key=lambda value: _stable_key(seed, value))
            chosen = remaining[0]
        selected.add(chosen)
        for name in FEATURE_VALUES:
            selected_counts[(name, features[chosen][name])] += 1
    return sorted(selected, key=lambda value: _stable_key(seed, value))


def feature_hamming_distance(
    left: dict[str, str], right: dict[str, str]
) -> int:
    return sum(left[name] != right[name] for name in FEATURE_VALUES)


def _jensen_shannon(
    pool_counts: Counter[str],
    selected_counts: Counter[str],
    pool_total: int,
    selected_total: int,
) -> float:
    divergence = 0.0
    for value in set(pool_counts) | set(selected_counts):
        p = pool_counts[value] / max(1, pool_total)
        q = selected_counts[value] / max(1, selected_total)
        midpoint = 0.5 * (p + q)
        if p:
            divergence += 0.5 * p * math.log(p / midpoint, 2)
        if q:
            divergence += 0.5 * q * math.log(q / midpoint, 2)
    return divergence


def feature_coverage_report(
    pool_features: dict[str, dict[str, str]], selected_ids: Iterable[str]
) -> dict[str, Any]:
    selected = set(map(str, selected_ids))
    if not selected <= set(pool_features):
        raise ValueError("selected IDs are not a subset of the feature pool")
    report: dict[str, Any] = {}
    zero_covered = 0
    divergences = []
    for name, values in FEATURE_VALUES.items():
        pool_counts = Counter(features[name] for features in pool_features.values())
        selected_counts = Counter(pool_features[question_id][name] for question_id in selected)
        categories = {}
        for value in values:
            if not pool_counts[value]:
                continue
            if not selected_counts[value]:
                zero_covered += 1
            categories[value] = {
                "pool_count": pool_counts[value],
                "selected_count": selected_counts[value],
                "pool_share": pool_counts[value] / len(pool_features),
                "selected_share": selected_counts[value] / max(1, len(selected)),
            }
        divergence = _jensen_shannon(
            pool_counts, selected_counts, len(pool_features), len(selected)
        )
        divergences.append(divergence)
        categories["_jensen_shannon_divergence"] = divergence
        report[name] = categories
    return {
        "pool_questions": len(pool_features),
        "selected_questions": len(selected),
        "zero_covered_source_categories": zero_covered,
        "mean_marginal_jensen_shannon_divergence": sum(divergences)
        / len(divergences),
        "maximum_marginal_jensen_shannon_divergence": max(divergences, default=0.0),
        "features": report,
    }


def pair_feature_coverage_report(
    features: dict[str, dict[str, str]],
    pairs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(pairs)
    adjacency: dict[str, set[str]] = defaultdict(set)
    endpoint_counts: Counter[tuple[str, str]] = Counter()
    distances: Counter[int] = Counter()
    for row in rows:
        left, right = str(row["question_a_id"]), str(row["question_b_id"])
        if left not in features or right not in features:
            raise ValueError("pair endpoint is missing auxiliary features")
        adjacency[left].add(right)
        adjacency[right].add(left)
        distances[feature_hamming_distance(features[left], features[right])] += 1
        for name in FEATURE_VALUES:
            endpoint_counts[(name, features[left][name])] += 1
            endpoint_counts[(name, features[right][name])] += 1
    selected = set(adjacency)
    sliced = {}
    for name, values in FEATURE_VALUES.items():
        categories = {}
        for value in values:
            nodes = [
                question_id
                for question_id in selected
                if features[question_id][name] == value
            ]
            if not nodes:
                continue
            degrees = [len(adjacency[node]) for node in nodes]
            categories[value] = {
                "question_count": len(nodes),
                "endpoint_occurrences": endpoint_counts[(name, value)],
                "minimum_degree": min(degrees),
                "mean_degree": sum(degrees) / len(degrees),
                "maximum_degree": max(degrees),
            }
        sliced[name] = categories
    return {
        "pair_records": len(rows),
        "feature_hamming_distance_distribution": {
            str(distance): count for distance, count in sorted(distances.items())
        },
        "coverage_by_feature_category": sliced,
    }
