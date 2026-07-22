#!/usr/bin/env python3
"""Build a reproducible, degree-balanced comparison graph within one split."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.metrics import graph_metrics

PAIR_SOURCE_WEIGHTS = {
    "adjacent_teacher_level": 0.35,
    "same_teacher_level": 0.25,
    "baseline_uncertain": 0.20,
    "cross_teacher_level": 0.10,
    "cross_domain_anchor": 0.10,
}


def load_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def domain(item: Dict[str, Any]) -> str:
    domains = (item.get("feature_metadata") or {}).get("knowledge_domains") or []
    if domains:
        return str(domains[0])
    structure = str((item.get("teacher_features") or {}).get("problem_structure") or "")
    for key, name in (("力学", "力学"), ("电路", "电路"), ("热学", "热学"), ("光学", "光学声学"), ("声学", "光学声学")):
        if key in structure:
            return name
    return "unknown"


def baseline_score(item: Dict[str, Any]) -> float | None:
    for container in (item, item.get("metadata") or {}, item.get("baseline") or {}):
        for key in ("baseline_expected_score", "expected_difficulty_score", "difficulty_score"):
            try:
                return float(container[key])
            except (KeyError, TypeError, ValueError):
                continue
    return None


def pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def weighted_choice(rng: random.Random, weights: Dict[str, float]) -> str:
    point = rng.random() * sum(weights.values())
    cumulative = 0.0
    for name, weight in weights.items():
        cumulative += weight
        if point <= cumulative:
            return name
    return next(reversed(weights))


def draw_partner(source: str, item: Dict[str, Any], rows: list[Dict[str, Any]], by_level: Dict[int, list[Dict[str, Any]]], by_domain: Dict[str, list[Dict[str, Any]]], score_sorted: list[Dict[str, Any]], rng: random.Random) -> Dict[str, Any]:
    level = item.get("teacher_difficulty_id")
    item_domain = domain(item)
    if source == "same_teacher_level" and isinstance(level, int):
        pool = by_level[level]
        return rng.choice(pool)
    if source == "adjacent_teacher_level" and isinstance(level, int):
        levels = [candidate for candidate in (level - 1, level + 1) if by_level.get(candidate)]
        return rng.choice(by_level[rng.choice(levels)]) if levels else rng.choice(rows)
    if source == "cross_teacher_level" and isinstance(level, int):
        levels = [candidate for candidate, values in by_level.items() if values and abs(candidate - level) >= 2]
        return rng.choice(by_level[rng.choice(levels)]) if levels else rng.choice(rows)
    if source == "cross_domain_anchor":
        domains = [candidate for candidate, values in by_domain.items() if values and candidate != item_domain]
        return rng.choice(by_domain[rng.choice(domains)]) if domains else rng.choice(rows)
    if source == "baseline_uncertain" and baseline_score(item) is not None and score_sorted:
        # Nearest-score candidates are found without materializing all O(N^2)
        # distances.  The sorted list index is cached on each temporary record.
        index = int(item["_score_index"])
        lower, upper = max(0, index - 30), min(len(score_sorted), index + 31)
        return rng.choice(score_sorted[lower:upper])
    # A missing baseline score should not erase 20% of the graph.  Same-level
    # questions are the closest available deterministic fallback.
    return rng.choice(by_level.get(level, rows) if isinstance(level, int) else rows)


def choose_partner(source: str, item: Dict[str, Any], rows: list[Dict[str, Any]], by_level: Dict[int, list[Dict[str, Any]]], by_domain: Dict[str, list[Dict[str, Any]]], score_sorted: list[Dict[str, Any]], existing: set[tuple[str, str]], degrees: Counter[str], max_degree: int, rng: random.Random) -> Dict[str, Any] | None:
    left_id = str(item["id"])
    for _ in range(100):
        partner = draw_partner(source, item, rows, by_level, by_domain, score_sorted, rng)
        right_id = str(partner["id"])
        if left_id == right_id or pair_key(left_id, right_id) in existing:
            continue
        if degrees[right_id] >= max_degree:
            continue
        return partner
    # Relax the source constraint to maintain coverage/connectivity.
    for _ in range(min(500, len(rows) * 2)):
        partner = rng.choice(rows)
        right_id = str(partner["id"])
        if left_id != right_id and pair_key(left_id, right_id) not in existing and degrees[right_id] < max_degree:
            return partner
    return None


def make_pair(left: Dict[str, Any], right: Dict[str, Any], source: str, split: str, index: int) -> Dict[str, Any]:
    left_id, right_id = str(left["id"]), str(right["id"])
    pair_digest = hashlib.sha256(f"{split}\0{min(left_id, right_id)}\0{max(left_id, right_id)}".encode()).hexdigest()[:20]
    return {
        "schema_version": "pair_candidate_v1",
        "pair_id": f"{split}_{pair_digest}",
        "split": split,
        "question_a_id": left_id,
        "question_b_id": right_id,
        "question_a_text": left["text"],
        "question_b_text": right["text"],
        "pair_source": source,
        "domain_relation": "same_domain" if domain(left) == domain(right) else "cross_domain",
        "metadata": {
            "candidate_index": index,
            "teacher_level_a": left.get("teacher_difficulty_id"),
            "teacher_level_b": right.get("teacher_difficulty_id"),
            "baseline_score_a": baseline_score(left),
            "baseline_score_b": baseline_score(right),
            "has_image_a": bool((left.get("diagnostics") or {}).get("has_image")),
            "has_image_b": bool((right.get("diagnostics") or {}).get("has_image")),
            "truncated_a": bool(((left.get("diagnostics") or {}).get("truncation") or {}).get("truncated")),
            "truncated_b": bool(((right.get("diagnostics") or {}).get("truncation") or {}).get("truncated")),
        },
    }


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    known, _ = bootstrap.parse_known_args()
    defaults = json.loads(Path(known.config).read_text(encoding="utf-8")) if known.config else {}
    parser = argparse.ArgumentParser(parents=[bootstrap])
    parser.set_defaults(**defaults)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selected-questions-output", help="write the exact sampled node set used by this graph")
    parser.add_argument("--target-pairs", type=int, required="target_pairs" not in defaults)
    parser.add_argument("--minimum-degree", type=int, default=4)
    parser.add_argument("--maximum-degree", type=int, default=12)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.target_pairs <= 0 or args.minimum_degree < 1 or args.maximum_degree < args.minimum_degree:
        raise ValueError("invalid target pair/degree configuration")

    rng = random.Random(args.seed)
    rows = load_jsonl(Path(args.questions))
    if args.max_questions is not None:
        if args.max_questions < 2:
            raise ValueError("max-questions must be at least two")
        rng.shuffle(rows)
        rows = rows[:args.max_questions]
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("question IDs must be unique")
    split_values = {str(row["split"]) for row in rows}
    if len(split_values) != 1:
        raise ValueError("one candidate file must contain exactly one split")
    split = next(iter(split_values))
    feasible_minimum = len(rows) * args.minimum_degree / 2
    feasible_maximum = min(len(rows) * args.maximum_degree / 2, len(rows) * (len(rows) - 1) / 2)
    if args.target_pairs < feasible_minimum:
        raise ValueError(f"target-pairs must be at least {feasible_minimum:.0f} for minimum-degree={args.minimum_degree}")
    if args.target_pairs > feasible_maximum:
        raise ValueError(f"target-pairs cannot exceed {feasible_maximum:.0f} for the requested graph")

    by_level: Dict[int, list[Dict[str, Any]]] = defaultdict(list)
    by_domain: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        level = row.get("teacher_difficulty_id")
        if isinstance(level, int):
            by_level[level].append(row)
        by_domain[domain(row)].append(row)
    score_sorted = sorted((row for row in rows if baseline_score(row) is not None), key=lambda row: float(baseline_score(row)))
    for index, row in enumerate(score_sorted):
        row["_score_index"] = index

    pairs: list[Dict[str, Any]] = []
    existing: set[tuple[str, str]] = set()
    degrees: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    def add_for(item: Dict[str, Any], source: str) -> bool:
        partner = choose_partner(source, item, rows, by_level, by_domain, score_sorted, existing, degrees, args.maximum_degree, rng)
        if partner is None:
            return False
        key = pair_key(str(item["id"]), str(partner["id"]))
        existing.add(key)
        degrees[str(item["id"])] += 1
        degrees[str(partner["id"])] += 1
        pairs.append(make_pair(item, partner, source, split, len(pairs)))
        source_counts[source] += 1
        return True

    # Coverage pass: repeatedly prioritize the currently least-connected nodes.
    while min((degrees[str(row["id"])] for row in rows), default=args.minimum_degree) < args.minimum_degree:
        progress = False
        ordered = sorted(rows, key=lambda row: (degrees[str(row["id"])], rng.random()))
        for item in ordered:
            if degrees[str(item["id"])] >= args.minimum_degree or len(pairs) >= args.target_pairs:
                continue
            progress |= add_for(item, weighted_choice(rng, PAIR_SOURCE_WEIGHTS))
        if not progress:
            break

    # Distribution pass: fill the requested edge budget using the target mix.
    attempts = 0
    while len(pairs) < args.target_pairs and attempts < args.target_pairs * 30:
        attempts += 1
        eligible = [row for row in rows if degrees[str(row["id"])] < args.maximum_degree]
        if not eligible:
            break
        min_degree = min(degrees[str(row["id"])] for row in eligible)
        low_degree = [row for row in eligible if degrees[str(row["id"])] <= min_degree + 1]
        add_for(rng.choice(low_degree), weighted_choice(rng, PAIR_SOURCE_WEIGHTS))

    for row in rows:
        row.pop("_score_index", None)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as target:
        for pair in pairs:
            target.write(json.dumps(pair, ensure_ascii=False) + "\n")
    selected_questions_output = None
    if args.selected_questions_output:
        selected_path = Path(args.selected_questions_output)
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        with selected_path.open("w", encoding="utf-8") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
        selected_questions_output = str(selected_path.resolve())
    graph = graph_metrics(pairs, (str(row["id"]) for row in rows))
    manifest = {
        "schema_version": "pair_candidates_v1",
        "questions": str(Path(args.questions).resolve()),
        "output": str(output.resolve()),
        "selected_questions_output": selected_questions_output,
        "split": split,
        "seed": args.seed,
        "question_count": len(rows),
        "requested_pairs": args.target_pairs,
        "created_pairs": len(pairs),
        "minimum_degree_requested": args.minimum_degree,
        "maximum_degree_requested": args.maximum_degree,
        "source_counts": dict(source_counts),
        "source_target_weights": PAIR_SOURCE_WEIGHTS,
        "graph": graph,
        "warnings": [] if len(pairs) == args.target_pairs and graph["degree"]["minimum"] >= args.minimum_degree else ["Could not satisfy every requested pair/degree constraint"],
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
