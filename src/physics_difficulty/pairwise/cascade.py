"""Deterministic sampling and evaluation for cascaded pairwise teachers."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from physics_difficulty.pairwise.labels import aggregate_pair_votes


LENGTH_BUCKET_ORDER = {"short": 0, "medium": 1, "long": 2, "unknown": 3}


@dataclass(frozen=True)
class CascadeThresholds:
    max_position_bias_gap: float = 0.25
    decisive_low: float = 0.30
    decisive_high: float = 0.70
    minimum_votes_per_direction: int = 3

    def __post_init__(self) -> None:
        if not 0 <= self.max_position_bias_gap <= 1:
            raise ValueError("max_position_bias_gap must be in [0, 1]")
        if not 0 <= self.decisive_low < 0.5 < self.decisive_high <= 1:
            raise ValueError("decisive thresholds must straddle 0.5")
        if self.decisive_low >= self.decisive_high:
            raise ValueError("decisive_low must be below decisive_high")
        if self.minimum_votes_per_direction < 1:
            raise ValueError("minimum_votes_per_direction must be positive")


def stable_digest(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def pair_stratum(row: dict[str, Any]) -> tuple[str, str]:
    metadata = row.get("metadata") or {}
    buckets = [str(metadata.get("length_bucket_a") or "unknown"), str(metadata.get("length_bucket_b") or "unknown")]
    buckets.sort(key=lambda value: (LENGTH_BUCKET_ORDER.get(value, 99), value))
    return str(row.get("pair_source") or "unknown"), "/".join(buckets)


def _allocate_strata(group_sizes: dict[tuple[str, str], int], sample_size: int) -> dict[tuple[str, str], int]:
    keys = sorted(group_sizes)
    quotas = {key: sample_size * group_sizes[key] / sum(group_sizes.values()) for key in keys}
    allocations = {key: min(group_sizes[key], int(quotas[key])) for key in keys}
    remaining = sample_size - sum(allocations.values())
    while remaining:
        candidates = [key for key in keys if allocations[key] < group_sizes[key]]
        if not candidates:
            raise ValueError("stratified allocation exhausted before reaching sample size")
        chosen = max(
            candidates,
            key=lambda key: (
                quotas[key] - allocations[key],
                group_sizes[key] - allocations[key],
                tuple(reversed(key)),
            ),
        )
        allocations[chosen] += 1
        remaining -= 1
    return allocations


def select_stratified_pairs(
    rows: Sequence[dict[str, Any]],
    sample_size: int,
    seed: int,
    excluded_pair_ids: set[str] | None = None,
    excluded_question_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic proportional sample using largest remainders."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    excluded = {str(value) for value in (excluded_pair_ids or set())}
    excluded_questions = {str(value) for value in (excluded_question_ids or set())}
    pair_ids = [str(row.get("pair_id")) for row in rows]
    if any(value in {"", "None"} for value in pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("candidate pair IDs must be present and unique")
    excluded_by_pair = [row for row in rows if str(row["pair_id"]) in excluded]
    excluded_by_question = [
        row for row in rows
        if str(row.get("question_a_id")) in excluded_questions or str(row.get("question_b_id")) in excluded_questions
    ]
    available = [
        row for row in rows
        if str(row["pair_id"]) not in excluded
        and str(row.get("question_a_id")) not in excluded_questions
        and str(row.get("question_b_id")) not in excluded_questions
    ]
    if sample_size > len(available):
        raise ValueError(f"requested {sample_size} pairs but only {len(available)} remain after exclusion")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in available:
        grouped[pair_stratum(row)].append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda row: (stable_digest(seed, str(row["pair_id"])), str(row["pair_id"])))
    allocations = _allocate_strata({key: len(value) for key, value in grouped.items()}, sample_size)
    selected = [row for key in sorted(grouped) for row in grouped[key][: allocations[key]]]
    selected.sort(key=lambda row: (stable_digest(seed, str(row["pair_id"])), str(row["pair_id"])))

    def readable(values: dict[tuple[str, str], int]) -> dict[str, int]:
        return {"|".join(key): values[key] for key in sorted(values)}

    stats = {
        "source_pairs": len(rows),
        "excluded_pair_ids_requested": len(excluded),
        "excluded_question_ids_requested": len(excluded_questions),
        "excluded_by_pair_id": len(excluded_by_pair),
        "excluded_by_question_id": len(excluded_by_question),
        "excluded_pairs_total": len(rows) - len(available),
        "available_pairs": len(available),
        "selected_pairs": len(selected),
        "available_by_stratum": readable({key: len(value) for key, value in grouped.items()}),
        "selected_by_stratum": readable(allocations),
    }
    return selected, stats


def split_pairs_balanced(rows: Sequence[dict[str, Any]], shard_count: int, seed: int) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Greedily balance rendered character load while preserving every pair once."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if shard_count > len(rows):
        raise ValueError("shard_count cannot exceed pair count")
    ids = [str(row.get("pair_id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("pair IDs must be unique before sharding")
    weighted = [
        (len(str(row.get("question_a_text") or "")) + len(str(row.get("question_b_text") or "")), row)
        for row in rows
    ]
    weighted.sort(key=lambda item: (-item[0], stable_digest(seed, str(item[1]["pair_id"]))))
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for cost, row in weighted:
        index = min(range(shard_count), key=lambda value: (loads[value], len(shards[value]), value))
        shards[index].append(row)
        loads[index] += cost
    for shard in shards:
        shard.sort(key=lambda row: (stable_digest(seed, str(row["pair_id"])), str(row["pair_id"])))
    return shards, {
        "shard_count": shard_count,
        "pairs": len(rows),
        "shards": [
            {"index": index, "pairs": len(shard), "text_characters": loads[index]}
            for index, shard in enumerate(shards)
        ],
    }


def _valid_direction_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("direction")) for row in rows if row.get("valid"))
    return {"forward": counts["forward"], "backward": counts["backward"]}


def decide_cascade_route(rows: Sequence[dict[str, Any]], thresholds: CascadeThresholds) -> dict[str, Any]:
    counts = _valid_direction_counts(rows)
    if min(counts.values()) < thresholds.minimum_votes_per_direction:
        return {
            "action": "escalate_thinking_1024",
            "reason": "insufficient_valid_votes",
            "valid_votes_by_direction": counts,
            "thresholds": asdict(thresholds),
        }
    stats = aggregate_pair_votes(rows)
    position_sensitive = float(stats["position_bias_gap"]) > thresholds.max_position_bias_gap
    uncertain = thresholds.decisive_low <= float(stats["soft_target"]) <= thresholds.decisive_high
    if not position_sensitive and not uncertain:
        action, reason = "accept_nonthinking", "stable_and_decisive"
    elif position_sensitive and uncertain:
        action, reason = "escalate_thinking_1024", "position_sensitive_and_uncertain"
    elif position_sensitive:
        action, reason = "escalate_thinking_1024", "position_sensitive"
    else:
        action, reason = "escalate_thinking_1024", "uncertain"
    return {
        "action": action,
        "reason": reason,
        "valid_votes_by_direction": counts,
        "soft_target": float(stats["soft_target"]),
        "position_bias_gap": float(stats["position_bias_gap"]),
        "valid_votes": int(stats["valid_votes"]),
        "thresholds": asdict(thresholds),
    }


def hard_label(target: float | None) -> str | None:
    if target is None:
        return None
    if target > 0.5:
        return "A"
    if target < 0.5:
        return "B"
    return "tie"


def _group_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    return grouped


def _aggregate_optional(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        return aggregate_pair_votes(rows)
    except ValueError:
        return None


def evaluate_cascade(
    pairs: Sequence[dict[str, Any]],
    nonthinking_rows: Sequence[dict[str, Any]],
    thinking_rows: Sequence[dict[str, Any]],
    thresholds: CascadeThresholds,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_ids = [str(row["pair_id"]) for row in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("evaluation pair IDs must be unique")
    nonthinking = _group_rows(nonthinking_rows)
    thinking = _group_rows(thinking_rows)
    records: list[dict[str, Any]] = []
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        decision = decide_cascade_route(nonthinking.get(pair_id, []), thresholds)
        thinking_stats = _aggregate_optional(thinking.get(pair_id, []))
        nonthinking_target = decision.get("soft_target")
        thinking_target = float(thinking_stats["soft_target"]) if thinking_stats else None
        nonthinking_label = hard_label(nonthinking_target)
        thinking_label = hard_label(thinking_target)
        hard_disagreement = nonthinking_label is not None and thinking_label is not None and nonthinking_label != thinking_label
        severe = hard_disagreement and nonthinking_label in {"A", "B"} and thinking_label in {"A", "B"}
        final_source = "nonthinking" if decision["action"] == "accept_nonthinking" else "thinking_1024"
        final_target = nonthinking_target if final_source == "nonthinking" else thinking_target
        records.append({
            "pair_id": pair_id,
            "route_action": decision["action"],
            "route_reason": decision["reason"],
            "nonthinking_soft_target": nonthinking_target,
            "nonthinking_position_bias_gap": decision.get("position_bias_gap"),
            "thinking_soft_target": thinking_target,
            "thinking_position_bias_gap": float(thinking_stats["position_bias_gap"]) if thinking_stats else None,
            "hard_disagreement": hard_disagreement,
            "severe_disagreement": severe,
            "final_label_source": final_source,
            "final_soft_target": final_target,
        })

    accepted = [row for row in records if row["route_action"] == "accept_nonthinking"]
    comparable = [row for row in accepted if row["thinking_soft_target"] is not None]
    agreements = [not row["hard_disagreement"] for row in comparable]
    differences = [abs(float(row["nonthinking_soft_target"]) - float(row["thinking_soft_target"])) for row in comparable]
    total_vote_rows = len(nonthinking_rows)
    report = {
        "schema_version": "cascade_teacher_evaluation_v1",
        "pairs": len(pairs),
        "thresholds": asdict(thresholds),
        "nonthinking_vote_rows": total_vote_rows,
        "nonthinking_valid_votes": sum(bool(row.get("valid")) for row in nonthinking_rows),
        "nonthinking_parse_success_rate": sum(bool(row.get("valid")) for row in nonthinking_rows) / max(1, total_vote_rows),
        "thinking_vote_rows": len(thinking_rows),
        "thinking_valid_votes": sum(bool(row.get("valid")) for row in thinking_rows),
        "direct_accept_count": len(accepted),
        "escalated_count": len(pairs) - len(accepted),
        "direct_acceptance_rate": len(accepted) / max(1, len(pairs)),
        "accepted_pairs_comparable_with_thinking": len(comparable),
        "accepted_hard_agreement_with_thinking": sum(agreements) / len(agreements) if agreements else None,
        "accepted_mean_absolute_soft_target_difference": sum(differences) / len(differences) if differences else None,
        "hard_disagreement_count": sum(bool(row["hard_disagreement"]) for row in records),
        "severe_disagreement_count": sum(bool(row["severe_disagreement"]) for row in records),
        "accepted_severe_disagreement_count": sum(bool(row["severe_disagreement"]) for row in accepted),
        "accepted_severe_disagreement_rate": sum(bool(row["severe_disagreement"]) for row in accepted) / max(1, len(accepted)),
        "route_reason_counts": dict(Counter(str(row["route_reason"]) for row in records)),
    }
    return report, records


def merge_vote_rows(shards: Sequence[Sequence[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    identities: set[tuple[str, str, int]] = set()
    questions_by_pair: dict[str, tuple[str, str]] = {}
    for rows in shards:
        for row in rows:
            identity = (str(row["pair_id"]), str(row["direction"]), int(row["sample_index"]))
            if identity in identities:
                raise ValueError(f"duplicate vote identity: {identity}")
            identities.add(identity)
            questions = (str(row["question_a_id"]), str(row["question_b_id"]))
            previous = questions_by_pair.setdefault(str(row["pair_id"]), questions)
            if previous != questions:
                raise ValueError(f"inconsistent question IDs for pair {row['pair_id']}")
            merged.append(row)
    direction_order = {"forward": 0, "backward": 1}
    merged.sort(key=lambda row: (str(row["pair_id"]), direction_order.get(str(row["direction"]), 9), int(row["sample_index"])))
    return merged, {
        "schema_version": "merged_teacher_votes_v1",
        "input_shards": len(shards),
        "pairs": len(questions_by_pair),
        "rows": len(merged),
        "valid_votes": sum(bool(row.get("valid")) for row in merged),
        "teacher_modes": sorted({str((row.get("teacher") or {}).get("mode") or "unknown") for row in merged}),
    }


def select_blind_audit_pairs(
    pairs: Sequence[dict[str, Any]],
    evaluation_records: Sequence[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    by_id = {str(row["pair_id"]): row for row in pairs}
    if sample_size > len(by_id):
        raise ValueError("audit sample cannot exceed available pairs")
    selected_ids: list[str] = []
    selected_set: set[str] = set()
    reason_counts: Counter[str] = Counter()

    def add(rows: Iterable[dict[str, Any]], reason: str) -> None:
        ordered = sorted(rows, key=lambda row: (stable_digest(seed, str(row["pair_id"])), str(row["pair_id"])))
        for row in ordered:
            pair_id = str(row["pair_id"])
            if len(selected_ids) >= sample_size:
                return
            if pair_id in by_id and pair_id not in selected_set:
                selected_ids.append(pair_id)
                selected_set.add(pair_id)
                reason_counts[reason] += 1

    add((row for row in evaluation_records if row.get("hard_disagreement")), "cross_mode_disagreement")
    add((row for row in evaluation_records if max(float(row.get("nonthinking_position_bias_gap") or 0), float(row.get("thinking_position_bias_gap") or 0)) > 0.30), "high_position_bias")
    add((row for row in evaluation_records if row.get("thinking_soft_target") is not None and 0.30 <= float(row["thinking_soft_target"]) <= 0.70), "thinking_uncertain")
    add(evaluation_records, "stable_random_fill")

    blind = []
    for index, pair_id in enumerate(selected_ids, 1):
        row = by_id[pair_id]
        blind.append({
            "audit_index": index,
            "pair_id": pair_id,
            "question_a_id": str(row["question_a_id"]),
            "question_b_id": str(row["question_b_id"]),
            "question_a_text": row["question_a_text"],
            "question_b_text": row["question_b_text"],
            "human_preference": None,
            "human_confidence": None,
            "human_notes": "",
        })
    return blind, {
        "schema_version": "blind_pair_audit_selection_v1",
        "requested": sample_size,
        "selected": len(blind),
        "seed": seed,
        "selection_reason_counts": dict(reason_counts),
        "predictions_in_blind_file": False,
    }


REPRESENTATIVE_AUDIT_STRATA = (
    "stable_and_decisive",
    "escalated_same_direction",
    "escalated_teacher_disagreement",
)


def _representative_audit_stratum(record: dict[str, Any]) -> str:
    if record.get("route_action") == "accept_nonthinking":
        return "stable_and_decisive"
    if record.get("route_action") == "escalate_thinking_1024":
        if bool(record.get("hard_disagreement")):
            return "escalated_teacher_disagreement"
        return "escalated_same_direction"
    raise ValueError(f"unsupported cascade route action for pair {record.get('pair_id')}: {record.get('route_action')}")


def _human_audit_row(pair: dict[str, Any], prior: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "audit_index": None,
        "pair_id": str(pair["pair_id"]),
        "question_a_id": str(pair["question_a_id"]),
        "question_b_id": str(pair["question_b_id"]),
        "question_a_text": pair["question_a_text"],
        "question_b_text": pair["question_b_text"],
        "human_preference": prior.get("human_preference") if prior else None,
        "human_confidence": prior.get("human_confidence") if prior else None,
        "human_notes": str(prior.get("human_notes") or "") if prior else "",
    }


def select_representative_audit_pairs(
    pairs: Sequence[dict[str, Any]],
    evaluation_records: Sequence[dict[str, Any]],
    prior_audit_rows: Sequence[dict[str, Any]],
    quotas: dict[str, int],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select a route-stratified audit, preferring unseen pairs over prior labels.

    The returned blind rows intentionally omit teacher predictions and route strata.
    Previously reviewed rows are returned separately with their human labels intact.
    """
    unknown_quotas = set(quotas) - set(REPRESENTATIVE_AUDIT_STRATA)
    missing_quotas = set(REPRESENTATIVE_AUDIT_STRATA) - set(quotas)
    if unknown_quotas or missing_quotas:
        raise ValueError(f"quotas must contain exactly {REPRESENTATIVE_AUDIT_STRATA}")
    if any(not isinstance(value, int) or value < 0 for value in quotas.values()):
        raise ValueError("audit quotas must be non-negative integers")

    by_id = {str(row.get("pair_id")): row for row in pairs}
    if len(by_id) != len(pairs) or any(value in {"", "None"} for value in by_id):
        raise ValueError("pair IDs must be present and unique")
    records_by_id = {str(row.get("pair_id")): row for row in evaluation_records}
    if len(records_by_id) != len(evaluation_records):
        raise ValueError("evaluation pair IDs must be unique")
    missing_pairs = sorted(set(records_by_id) - set(by_id))
    if missing_pairs:
        raise ValueError(f"evaluation contains {len(missing_pairs)} unknown pair IDs")

    prior_by_id: dict[str, dict[str, Any]] = {}
    for row in prior_audit_rows:
        pair_id = str(row.get("pair_id"))
        if pair_id in prior_by_id:
            raise ValueError(f"duplicate prior audit pair ID: {pair_id}")
        if pair_id not in by_id:
            continue
        if row.get("human_preference") not in {"A", "B", "tie"}:
            raise ValueError(f"prior audit pair {pair_id} lacks a valid human preference")
        prior_by_id[pair_id] = row

    grouped: dict[str, list[str]] = {key: [] for key in REPRESENTATIVE_AUDIT_STRATA}
    for pair_id, record in records_by_id.items():
        grouped[_representative_audit_stratum(record)].append(pair_id)
    for pair_ids in grouped.values():
        pair_ids.sort(key=lambda pair_id: (stable_digest(seed, pair_id), pair_id))

    new_rows: list[dict[str, Any]] = []
    reused_rows: list[dict[str, Any]] = []
    manifest_strata: dict[str, Any] = {}
    selection_sequence: list[tuple[str, str, str]] = []
    for stratum in REPRESENTATIVE_AUDIT_STRATA:
        pair_ids = grouped[stratum]
        quota = quotas[stratum]
        if quota > len(pair_ids):
            raise ValueError(f"requested {quota} rows from {stratum}, but only {len(pair_ids)} are available")
        unseen = [pair_id for pair_id in pair_ids if pair_id not in prior_by_id]
        reviewed = [pair_id for pair_id in pair_ids if pair_id in prior_by_id]
        selected_new = unseen[:quota]
        selected_reused = reviewed[: max(0, quota - len(selected_new))]
        if len(selected_new) + len(selected_reused) != quota:
            raise ValueError(f"could not satisfy audit quota for {stratum}")
        for pair_id in selected_new:
            new_rows.append(_human_audit_row(by_id[pair_id]))
            selection_sequence.append((stratum, "new", pair_id))
        for pair_id in selected_reused:
            reused_rows.append(_human_audit_row(by_id[pair_id], prior_by_id[pair_id]))
            selection_sequence.append((stratum, "reused", pair_id))
        manifest_strata[stratum] = {
            "requested": quota,
            "available": len(pair_ids),
            "new_available": len(unseen),
            "prior_reviewed_available": len(reviewed),
            "new": len(selected_new),
            "reused": len(selected_reused),
            "selected_pair_ids": selected_new + selected_reused,
        }

    new_rows.sort(key=lambda row: (stable_digest(seed, str(row["pair_id"])), str(row["pair_id"])))
    reused_rows.sort(key=lambda row: (stable_digest(seed, str(row["pair_id"])), str(row["pair_id"])))
    for index, row in enumerate(new_rows, 1):
        row["audit_index"] = index
    for index, row in enumerate(reused_rows, 1):
        row["audit_index"] = index
    return new_rows, reused_rows, {
        "schema_version": "representative_pair_audit_selection_v1",
        "seed": seed,
        "requested_total": sum(quotas.values()),
        "selected_total": len(new_rows) + len(reused_rows),
        "new_total": len(new_rows),
        "reused_total": len(reused_rows),
        "strata": manifest_strata,
        "selection_sequence": [
            {"stratum": stratum, "label_source": source, "pair_id": pair_id}
            for stratum, source, pair_id in selection_sequence
        ],
        "predictions_in_blind_file": False,
    }


def _audited_mode_metrics(rows: Sequence[dict[str, Any]], target_field: str) -> dict[str, Any]:
    non_ties = [row for row in rows if row["human_preference"] in {"A", "B"}]
    exact = [row for row in rows if hard_label(row.get(target_field)) == row["human_preference"]]
    correct = [row for row in non_ties if hard_label(row.get(target_field)) == row["human_preference"]]
    return {
        "target_field": target_field,
        "human_records": len(rows),
        "human_non_tie_records": len(non_ties),
        "teacher_predictions_available": sum(row.get(target_field) is not None for row in rows),
        "correct": len(correct),
        "directional_accuracy": len(correct) / len(non_ties) if non_ties else None,
        "exact_three_way_agreement": len(exact) / len(rows) if rows else None,
    }


def _audited_slice(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "human_records": len(rows),
        "human_non_tie_records": sum(row["human_preference"] in {"A", "B"} for row in rows),
        "human_preference_counts": dict(Counter(str(row["human_preference"]) for row in rows)),
        "human_confidence_counts": dict(Counter(str(row["human_confidence"]) for row in rows)),
        "nonthinking": _audited_mode_metrics(rows, "nonthinking_soft_target"),
        "thinking_1024": _audited_mode_metrics(rows, "thinking_soft_target"),
        "cascade_final": _audited_mode_metrics(rows, "final_soft_target"),
    }


def evaluate_human_audit(
    evaluation_records: Sequence[dict[str, Any]],
    human_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compare nonthinking, thinking and cascaded labels against a human audit."""
    records_by_id = {str(row.get("pair_id")): row for row in evaluation_records}
    if len(records_by_id) != len(evaluation_records):
        raise ValueError("evaluation pair IDs must be unique")
    human_by_id = {str(row.get("pair_id")): row for row in human_rows}
    if len(human_by_id) != len(human_rows):
        raise ValueError("human audit pair IDs must be unique")

    joined: list[dict[str, Any]] = []
    for pair_id, human in human_by_id.items():
        if pair_id not in records_by_id:
            raise ValueError(f"human audit contains unknown pair ID: {pair_id}")
        if human.get("human_preference") not in {"A", "B", "tie"}:
            raise ValueError(f"invalid human preference for pair {pair_id}")
        if human.get("human_confidence") not in {"high", "medium", "low"}:
            raise ValueError(f"invalid human confidence for pair {pair_id}")
        joined.append({**records_by_id[pair_id], **human, "pair_id": pair_id})
    joined.sort(key=lambda row: str(row["pair_id"]))

    strata: dict[str, list[dict[str, Any]]] = {key: [] for key in REPRESENTATIVE_AUDIT_STRATA}
    for row in joined:
        strata[_representative_audit_stratum(row)].append(row)
    population_counts = Counter(_representative_audit_stratum(row) for row in evaluation_records)
    stratum_reports = {key: _audited_slice(strata[key]) for key in REPRESENTATIVE_AUDIT_STRATA}
    for key, stratum_report in stratum_reports.items():
        stratum_report["population_records"] = population_counts[key]
        stratum_report["audit_coverage"] = (
            stratum_report["human_records"] / population_counts[key]
            if population_counts[key] else None
        )
    weighted: dict[str, float | None] = {}
    for mode in ("nonthinking", "thinking_1024", "cascade_final"):
        accuracies = [stratum_reports[key][mode]["directional_accuracy"] for key in REPRESENTATIVE_AUDIT_STRATA]
        estimated_non_tie_counts = {
            key: population_counts[key]
            * stratum_reports[key]["human_non_tie_records"]
            / stratum_reports[key]["human_records"]
            if stratum_reports[key]["human_records"] else 0.0
            for key in REPRESENTATIVE_AUDIT_STRATA
        }
        estimated_non_tie_total = sum(estimated_non_tie_counts.values())
        if any(value is None for value in accuracies) or not estimated_non_tie_total:
            weighted[mode] = None
        else:
            weighted[mode] = sum(
                estimated_non_tie_counts[key] * float(stratum_reports[key][mode]["directional_accuracy"])
                for key in REPRESENTATIVE_AUDIT_STRATA
            ) / estimated_non_tie_total
    confidence = {
        level: _audited_slice([row for row in joined if row["human_confidence"] == level])
        for level in ("high", "medium", "low")
    }
    return {
        "schema_version": "human_pair_audit_evaluation_v1",
        "human_records": len(joined),
        "human_non_tie_records": sum(row["human_preference"] in {"A", "B"} for row in joined),
        "population_records": len(evaluation_records),
        "population_stratum_counts": {key: population_counts[key] for key in REPRESENTATIVE_AUDIT_STRATA},
        "population_weighted_directional_accuracy": weighted,
        "overall": _audited_slice(joined),
        "confidence": confidence,
        "strata": stratum_reports,
    }
