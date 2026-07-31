"""CPU-only BT-decile and auxiliary-feature balanced question selection."""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Sequence

from physics_difficulty.schema import FEATURE_VALUES


def _stable_key(seed: int, question_id: str, namespace: str) -> str:
    payload = f"{seed}\0{namespace}\0{question_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_inputs(
    question_ids: Sequence[str],
    scores: dict[str, float],
    features: dict[str, dict[str, str]],
) -> list[str]:
    ids = [str(value) for value in question_ids]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("question IDs must be non-empty and unique")
    expected = set(ids)
    if set(scores) != expected:
        raise ValueError(
            f"score IDs do not match question pool: missing={len(expected - set(scores))}, "
            f"extra={len(set(scores) - expected)}"
        )
    if set(features) != expected:
        raise ValueError(
            f"feature IDs do not match question pool: missing={len(expected - set(features))}, "
            f"extra={len(set(features) - expected)}"
        )
    for question_id in ids:
        if not math.isfinite(float(scores[question_id])):
            raise ValueError(f"question {question_id} has a non-finite BT score")
        for name, values in FEATURE_VALUES.items():
            if features[question_id].get(name) not in values:
                raise ValueError(
                    f"question {question_id} has invalid auxiliary feature {name}"
                )
    return ids


def assign_bt_deciles(
    question_ids: Sequence[str],
    scores: dict[str, float],
    *,
    deciles: int = 10,
    seed: int = 42,
) -> dict[str, int]:
    """Assign equal-frequency score bins using stable ID tie-breaking."""
    if deciles < 2 or len(question_ids) < deciles:
        raise ValueError("BT score bin count must be at least 2 and no larger than the pool")
    ordered = sorted(
        map(str, question_ids),
        key=lambda question_id: (
            float(scores[question_id]),
            _stable_key(seed, question_id, "score-tie"),
        ),
    )
    count = len(ordered)
    return {
        question_id: min(deciles, rank * deciles // count + 1)
        for rank, question_id in enumerate(ordered)
    }


def _reason_counts(
    quota: int,
    distribution_fraction: float,
    rare_fraction: float,
    random_fraction: float,
) -> tuple[int, int, int]:
    fractions = distribution_fraction + rare_fraction + random_fraction
    if not math.isclose(fractions, 1.0, abs_tol=1e-9):
        raise ValueError("selection fractions must sum to 1")
    rare = round(quota * rare_fraction)
    random = round(quota * random_fraction)
    distribution = quota - rare - random
    if min(distribution, rare, random) < 0:
        raise ValueError("selection fractions produce a negative quota")
    return distribution, rare, random


def _category_floor_targets(
    ids: Sequence[str],
    assignments: dict[str, int],
    features: dict[str, dict[str, str]],
    *,
    deciles: int,
    minimum_global: int,
    minimum_per_decile: int,
) -> dict[int, dict[tuple[str, str], int]]:
    """Allocate deterministic category coverage floors across BT deciles."""
    if minimum_global < 0 or minimum_per_decile < 0:
        raise ValueError("category coverage floors must be non-negative")
    pool_counts: Counter[tuple[int, str, str]] = Counter()
    global_counts: Counter[tuple[str, str]] = Counter()
    for question_id in ids:
        decile = assignments[question_id]
        for name in FEATURE_VALUES:
            category = (name, features[question_id][name])
            pool_counts[(decile, *category)] += 1
            global_counts[category] += 1

    targets: dict[int, dict[tuple[str, str], int]] = {
        decile: {} for decile in range(1, deciles + 1)
    }
    for category, global_count in sorted(global_counts.items()):
        allocations = {
            decile: min(
                minimum_per_decile,
                pool_counts[(decile, *category)],
            )
            for decile in range(1, deciles + 1)
        }
        desired = max(
            sum(allocations.values()),
            min(minimum_global, global_count),
        )
        remaining = desired - sum(allocations.values())
        while remaining:
            candidates = [
                decile
                for decile in range(1, deciles + 1)
                if allocations[decile] < pool_counts[(decile, *category)]
            ]
            if not candidates:
                raise RuntimeError(f"cannot allocate category floor for {category}")
            chosen = max(
                candidates,
                key=lambda decile: (
                    (
                        pool_counts[(decile, *category)] - allocations[decile]
                    )
                    / pool_counts[(decile, *category)],
                    -decile,
                ),
            )
            allocations[chosen] += 1
            remaining -= 1
        for decile, count in allocations.items():
            if count:
                targets[decile][category] = count
    return targets


def _add_selected(
    question_id: str,
    reason: str,
    selected: dict[str, str],
    selected_counts: Counter[tuple[str, str]],
    features: dict[str, dict[str, str]],
) -> None:
    selected[question_id] = reason
    for name in FEATURE_VALUES:
        selected_counts[(name, features[question_id][name])] += 1


def _select_category_floor(
    pool: list[str],
    targets: dict[tuple[str, str], int],
    selected: dict[str, str],
    selected_counts: Counter[tuple[str, str]],
    features: dict[str, dict[str, str]],
    *,
    quota: int,
    seed: int,
    decile: int,
) -> None:
    """Greedy multi-feature set cover until every allocated floor is met."""
    position = 0
    while True:
        deficits = {
            category: target - selected_counts[category]
            for category, target in targets.items()
            if selected_counts[category] < target
        }
        if not deficits:
            return
        if len(selected) >= quota:
            raise ValueError(
                f"BT decile {decile} category floors exceed its quota {quota}"
            )
        candidates = [question_id for question_id in pool if question_id not in selected]
        if not candidates:
            raise ValueError(f"BT decile {decile} ran out of category-floor candidates")

        def coverage(question_id: str) -> tuple[float, str]:
            score = 0.0
            for name in FEATURE_VALUES:
                category = (name, features[question_id][name])
                if category in deficits:
                    score += deficits[category] / targets[category]
            return score, _stable_key(
                seed + position, question_id, f"category-floor-{decile}"
            )

        chosen = max(candidates, key=coverage)
        if coverage(chosen)[0] <= 0:
            raise RuntimeError(
                f"BT decile {decile} cannot satisfy category floors {deficits}"
            )
        _add_selected(
            chosen,
            "category_floor",
            selected,
            selected_counts,
            features,
        )
        position += 1


def _select_rare(
    pool: list[str],
    count: int,
    selected: dict[str, str],
    selected_counts: Counter[tuple[str, str]],
    features: dict[str, dict[str, str]],
    category_counts: Counter[tuple[str, str]],
    *,
    seed: int,
    decile: int,
) -> None:
    for position in range(count):
        candidates = [question_id for question_id in pool if question_id not in selected]
        if not candidates:
            raise ValueError(f"BT decile {decile} ran out of rare-feature candidates")

        def rarity(question_id: str) -> tuple[float, str]:
            score = 0.0
            for name in FEATURE_VALUES:
                category = (name, features[question_id][name])
                frequency = category_counts[category]
                score += 1.0 / frequency
                if selected_counts[category] == 0:
                    score += 2.0 / frequency
            return score, _stable_key(
                seed + position, question_id, f"rare-{decile}"
            )

        chosen = max(candidates, key=rarity)
        _add_selected(
            chosen,
            "rare_feature_protection",
            selected,
            selected_counts,
            features,
        )


def _select_distribution_matched(
    pool: list[str],
    count: int,
    total_quota: int,
    selected: dict[str, str],
    selected_counts: Counter[tuple[str, str]],
    features: dict[str, dict[str, str]],
    category_members: dict[tuple[str, str], list[str]],
    *,
    seed: int,
    decile: int,
) -> None:
    targets = {
        category: total_quota * len(members) / len(pool)
        for category, members in category_members.items()
    }
    cursors: Counter[tuple[str, str]] = Counter()
    for position in range(count):
        categories = sorted(
            category_members,
            key=lambda category: (
                -(
                    targets[category] - selected_counts[category]
                )
                / max(1.0, targets[category]),
                category,
            ),
        )
        chosen = None
        best_score = -math.inf
        for category in categories:
            members = category_members[category]
            cursor = cursors[category]
            while cursor < len(members) and members[cursor] in selected:
                cursor += 1
            cursors[category] = cursor
            if cursor >= len(members):
                continue
            # Look at a bounded set so the full 47k pool remains cheap on CPU.
            for question_id in members[cursor : cursor + 32]:
                if question_id in selected:
                    continue
                deficit_score = 0.0
                for name in FEATURE_VALUES:
                    own = (name, features[question_id][name])
                    deficit_score += max(
                        0.0,
                        (targets[own] - selected_counts[own])
                        / max(1.0, targets[own]),
                    )
                tie = int(
                    _stable_key(
                        seed + position,
                        question_id,
                        f"distribution-{decile}",
                    )[:12],
                    16,
                ) / 16**12
                candidate_score = deficit_score + tie * 1e-6
                if candidate_score > best_score:
                    best_score = candidate_score
                    chosen = question_id
            if chosen is not None:
                break
        if chosen is None:
            remaining = sorted(
                set(pool) - set(selected),
                key=lambda question_id: _stable_key(
                    seed, question_id, f"distribution-fallback-{decile}"
                ),
            )
            if not remaining:
                raise ValueError(f"BT decile {decile} ran out of candidates")
            chosen = remaining[0]
        _add_selected(
            chosen,
            "distribution_matched",
            selected,
            selected_counts,
            features,
        )


def select_questions_by_bt_decile(
    question_ids: Sequence[str],
    scores: dict[str, float],
    features: dict[str, dict[str, str]],
    *,
    target_count: int = 10_000,
    deciles: int = 10,
    distribution_fraction: float = 0.80,
    rare_fraction: float = 0.10,
    random_fraction: float = 0.10,
    minimum_category_count_global: int = 20,
    minimum_category_count_per_decile: int = 2,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Return an exact, deterministic selection with per-decile reason quotas."""
    ids = _validate_inputs(question_ids, scores, features)
    if not deciles <= target_count <= len(ids):
        raise ValueError("target_count must be between the BT bin count and pool size")
    assignments = assign_bt_deciles(ids, scores, deciles=deciles, seed=seed)
    floor_targets = _category_floor_targets(
        ids,
        assignments,
        features,
        deciles=deciles,
        minimum_global=minimum_category_count_global,
        minimum_per_decile=minimum_category_count_per_decile,
    )
    pools: dict[int, list[str]] = defaultdict(list)
    for question_id in ids:
        pools[assignments[question_id]].append(question_id)
    base, remainder = divmod(target_count, deciles)
    quotas = {
        decile: base + int(decile <= remainder)
        for decile in range(1, deciles + 1)
    }
    result: list[dict[str, Any]] = []
    for decile in range(1, deciles + 1):
        pool = pools[decile]
        quota = quotas[decile]
        if len(pool) < quota:
            raise ValueError(
                f"BT decile {decile} has {len(pool)} questions, below quota {quota}"
            )
        selected: dict[str, str] = {}
        selected_counts: Counter[tuple[str, str]] = Counter()
        _select_category_floor(
            pool,
            floor_targets[decile],
            selected,
            selected_counts,
            features,
            quota=quota,
            seed=seed,
            decile=decile,
        )
        remaining_quota = quota - len(selected)
        distribution_count, rare_count, random_count = _reason_counts(
            remaining_quota,
            distribution_fraction,
            rare_fraction,
            random_fraction,
        )
        random_ids = sorted(
            (question_id for question_id in pool if question_id not in selected),
            key=lambda question_id: _stable_key(
                seed, question_id, f"random-{decile}"
            ),
        )[:random_count]
        for question_id in random_ids:
            _add_selected(
                question_id,
                "random_exploration",
                selected,
                selected_counts,
                features,
            )

        category_members: dict[tuple[str, str], list[str]] = defaultdict(list)
        for question_id in pool:
            for name in FEATURE_VALUES:
                category_members[(name, features[question_id][name])].append(
                    question_id
                )
        for category, members in category_members.items():
            members.sort(
                key=lambda question_id: _stable_key(
                    seed, question_id, f"category-{decile}-{category}"
                )
            )
        category_counts = Counter(
            {category: len(members) for category, members in category_members.items()}
        )
        _select_rare(
            pool,
            rare_count,
            selected,
            selected_counts,
            features,
            category_counts,
            seed=seed,
            decile=decile,
        )
        _select_distribution_matched(
            pool,
            distribution_count,
            quota,
            selected,
            selected_counts,
            features,
            category_members,
            seed=seed,
            decile=decile,
        )
        if len(selected) != quota:
            raise RuntimeError(
                f"BT decile {decile} selected {len(selected)} instead of {quota}"
            )
        result.extend(
            {
                "question_id": question_id,
                "bt_score": float(scores[question_id]),
                "bt_decile": decile,
                "selection_reason": reason,
            }
            for question_id, reason in selected.items()
        )
    return sorted(
        result,
        key=lambda row: (
            row["bt_decile"],
            row["bt_score"],
            row["question_id"],
        ),
    )
