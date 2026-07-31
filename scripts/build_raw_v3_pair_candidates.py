#!/usr/bin/env python3
"""Build a label-free, connected comparison graph for raw V3 questions."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.data.text_only import normalize_for_dedup
from physics_difficulty.pairwise.feature_coverage import (
    feature_coverage_report,
    feature_hamming_distance,
    pair_feature_coverage_report,
    select_feature_balanced_ids,
    validate_feature_map,
)
from physics_difficulty.pairwise.metrics import graph_metrics


TEXT_ONLY_PAIR_SOURCE_WEIGHTS = {
    "lexical_near": 0.30,
    "structure_matched": 0.30,
    "random_global": 0.25,
    "graph_bridge": 0.05,
    "low_degree_repair": 0.10,
}
FEATURE_AWARE_PAIR_SOURCE_WEIGHTS = {
    "feature_near": 0.30,
    "feature_contrast": 0.25,
    "lexical_near": 0.10,
    "structure_matched": 0.10,
    "random_global": 0.15,
    "graph_bridge": 0.05,
    "low_degree_repair": 0.05,
}
FORBIDDEN_KEYS = {
    "difficulty",
    "raw_difficulty",
    "teacher_difficulty_id",
    "teacher_difficulty_level",
    "teacher_features",
    "teacher_features_legacy18",
}
TEXT_CHARACTERS = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in values}
        self.components = len(self.parent)

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]
        self.components -= 1


def load_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in FORBIDDEN_KEYS or str(key).startswith("teacher_difficulty"):
                paths.append(path)
            paths.extend(forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(forbidden_paths(child, f"{prefix}[{index}]"))
    return paths


def stable_rank(seed: int, question_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{question_id}".encode("utf-8")).hexdigest()


def pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def text_ngrams(text: str, maximum: int = 512) -> list[str]:
    compact = TEXT_CHARACTERS.sub("", normalize_for_dedup(text))
    if len(compact) < 3:
        return [compact] if compact else []
    grams = [compact[index:index + 3] for index in range(len(compact) - 2)]
    if len(grams) <= maximum:
        return grams
    # Evenly sample the whole question so long analyses do not erase the stem.
    return [grams[index * len(grams) // maximum] for index in range(maximum)]


def simhash64(text: str) -> int:
    weights = [0] * 64
    counts = Counter(text_ngrams(text))
    for gram, count in counts.items():
        encoded = gram.encode("utf-8")
        value = (zlib.crc32(encoded) << 32) | zlib.crc32(b"v3\0" + encoded)
        for bit in range(64):
            weights[bit] += count if value & (1 << bit) else -count
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def hashed_ngram_set(text: str) -> frozenset[int]:
    return frozenset(zlib.crc32(gram.encode("utf-8")) for gram in text_ngrams(text, maximum=256))


def jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def structure_signature(row: Dict[str, Any]) -> tuple[Any, ...]:
    diagnostics = row.get("diagnostics") or {}
    count = int(diagnostics.get("subquestion_count") or 0)
    count_bucket = "0" if count == 0 else "1" if count == 1 else "2-3" if count <= 3 else "4+"
    return (
        str(diagnostics.get("input_length_bucket") or "unknown"),
        count_bucket,
        bool(diagnostics.get("has_analysis")),
        bool(diagnostics.get("has_options")),
        str(diagnostics.get("image_dependency_risk") or "unknown"),
    )


def weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    point = rng.random() * sum(weights.values())
    cumulative = 0.0
    for name, weight in weights.items():
        cumulative += weight
        if point <= cumulative:
            return name
    return "random_global"


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    known, _ = bootstrap.parse_known_args()
    defaults = json.loads(Path(known.config).read_text(encoding="utf-8")) if known.config else {}
    parser = argparse.ArgumentParser(parents=[bootstrap])
    parser.set_defaults(**defaults)
    parser.add_argument("--questions", required=True)
    parser.add_argument(
        "--features-file",
        help="Curated feature JSONL keyed by id; only teacher_features is read.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--selected-questions-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target-pairs", type=int, required="target_pairs" not in defaults)
    parser.add_argument("--minimum-degree", type=int, default=4)
    parser.add_argument("--maximum-degree", type=int, default=12)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument(
        "--maximum-feature-jsd",
        type=float,
        default=0.05,
        help="Maximum allowed marginal Jensen-Shannon divergence after feature-aware selection.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.questions))
    if len(rows) < 2:
        raise ValueError("at least two questions are required")
    if len({str(row.get("id")) for row in rows}) != len(rows):
        raise ValueError("question IDs must be present and unique")
    split_values = {str(row.get("split")) for row in rows}
    if len(split_values) != 1:
        raise ValueError("one candidate file must contain exactly one split")
    for index, row in enumerate(rows, 1):
        paths = forbidden_paths(row)
        if paths:
            raise ValueError(f"question {index} contains forbidden historical label fields: {paths}")

    rows.sort(key=lambda row: stable_rank(args.seed, str(row["id"])))
    feature_map: dict[str, dict[str, str]] | None = None
    pool_feature_map: dict[str, dict[str, str]] | None = None
    feature_coverage: dict[str, Any] | None = None
    if args.features_file:
        pool_ids = [str(row["id"]) for row in rows]
        pool_feature_map = validate_feature_map(
            load_jsonl(Path(args.features_file)),
            allowed_question_ids=pool_ids,
        )
    if args.max_questions is not None:
        if args.max_questions < 2:
            raise ValueError("max-questions must be at least two")
        if pool_feature_map is not None:
            selected_ids = set(
                select_feature_balanced_ids(
                    [str(row["id"]) for row in rows],
                    pool_feature_map,
                    target_count=args.max_questions,
                    seed=args.seed,
                )
            )
            rows = [row for row in rows if str(row["id"]) in selected_ids]
        else:
            rows = rows[:args.max_questions]
    if pool_feature_map is not None:
        feature_map = {
            str(row["id"]): pool_feature_map[str(row["id"])] for row in rows
        }
        feature_coverage = feature_coverage_report(
            pool_feature_map, feature_map
        )
        if feature_coverage["zero_covered_source_categories"]:
            raise ValueError(
                "feature-balanced selection left one or more source auxiliary categories uncovered"
            )
        if (
            feature_coverage["maximum_marginal_jensen_shannon_divergence"]
            > args.maximum_feature_jsd
        ):
            raise ValueError(
                "feature-balanced selection exceeds maximum marginal Jensen-Shannon divergence: "
                f"{feature_coverage['maximum_marginal_jensen_shannon_divergence']:.6f} "
                f"> {args.maximum_feature_jsd:.6f}"
            )
    count = len(rows)
    minimum_required = max(count - 1, (count * args.minimum_degree + 1) // 2)
    maximum_allowed = min(count * (count - 1) // 2, count * args.maximum_degree // 2)
    if args.target_pairs < minimum_required:
        raise ValueError(f"target-pairs must be at least {minimum_required} for connectivity and minimum degree")
    if args.target_pairs > maximum_allowed:
        raise ValueError(f"target-pairs cannot exceed {maximum_allowed} for maximum degree")
    if args.minimum_degree < 1 or args.maximum_degree < args.minimum_degree:
        raise ValueError("invalid degree constraints")

    rng = random.Random(args.seed)
    pair_source_weights = (
        FEATURE_AWARE_PAIR_SOURCE_WEIGHTS
        if feature_map is not None
        else TEXT_ONLY_PAIR_SOURCE_WEIGHTS
    )
    ids = [str(row["id"]) for row in rows]
    by_id = {str(row["id"]): row for row in rows}
    fingerprints = {question_id: simhash64(by_id[question_id]["text"]) for question_id in ids}
    ngram_sets = {question_id: hashed_ngram_set(by_id[question_id]["text"]) for question_id in ids}
    lexical_bands: dict[tuple[int, int], list[str]] = defaultdict(list)
    for question_id, fingerprint in fingerprints.items():
        for band in range(8):
            lexical_bands[(band, (fingerprint >> (band * 8)) & 0xFF)].append(question_id)
    lexical_candidates: dict[str, set[str]] = defaultdict(set)
    for question_id, fingerprint in fingerprints.items():
        for band in range(8):
            lexical_candidates[question_id].update(lexical_bands[(band, (fingerprint >> (band * 8)) & 0xFF)])
        lexical_candidates[question_id].discard(question_id)

    structures: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    coarse_structures: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for question_id in ids:
        signature = structure_signature(by_id[question_id])
        structures[signature].append(question_id)
        coarse_structures[signature[:2]].append(question_id)
    feature_signatures: dict[tuple[str, ...], list[str]] = defaultdict(list)
    if feature_map is not None:
        for question_id in ids:
            signature = tuple(feature_map[question_id].values())
            feature_signatures[signature].append(question_id)

    existing: set[tuple[str, str]] = set()
    degrees: Counter[str] = Counter()
    pairs: list[Dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    union_find = UnionFind(ids)

    def eligible(left_id: str, candidates: Iterable[str]) -> list[str]:
        return [
            right_id for right_id in candidates
            if right_id != left_id
            and degrees[right_id] < args.maximum_degree
            and pair_key(left_id, right_id) not in existing
        ]

    def choose_partner(left_id: str, requested_source: str) -> tuple[str | None, str]:
        if requested_source in {"feature_near", "feature_contrast"}:
            if feature_map is None:
                raise RuntimeError("feature pair source requested without --features-file")
            if requested_source == "feature_near":
                signature = tuple(feature_map[left_id].values())
                candidates = eligible(left_id, feature_signatures[signature])
                if not candidates:
                    candidates = eligible(
                        left_id, rng.sample(ids, min(512, len(ids)))
                    )
                candidates.sort(
                    key=lambda value: (
                        feature_hamming_distance(
                            feature_map[left_id], feature_map[value]
                        ),
                        degrees[value],
                        stable_rank(args.seed, value),
                    )
                )
            else:
                candidates = eligible(
                    left_id, rng.sample(ids, min(512, len(ids)))
                )
                candidates.sort(
                    key=lambda value: (
                        -feature_hamming_distance(
                            feature_map[left_id], feature_map[value]
                        ),
                        degrees[value],
                        stable_rank(args.seed, value),
                    )
                )
            if candidates:
                return rng.choice(candidates[: min(8, len(candidates))]), requested_source
        elif requested_source == "lexical_near":
            candidates = eligible(left_id, lexical_candidates[left_id])
            if not candidates:
                # Sparse LSH buckets must not silently turn a requested near
                # edge into a random edge. Probe a bounded deterministic RNG
                # sample and retain the closest SimHash candidates.
                probe = rng.sample(ids, min(256, len(ids)))
                candidates = eligible(left_id, probe)
            if candidates:
                candidates.sort(key=lambda value: ((fingerprints[left_id] ^ fingerprints[value]).bit_count(), value))
                candidates = candidates[: min(128, len(candidates))]
                candidates.sort(key=lambda value: (-jaccard(ngram_sets[left_id], ngram_sets[value]), degrees[value], value))
                return rng.choice(candidates[: min(8, len(candidates))]), requested_source
        elif requested_source == "structure_matched":
            signature = structure_signature(by_id[left_id])
            candidates = eligible(left_id, structures[signature]) or eligible(left_id, coarse_structures[signature[:2]])
            if candidates:
                minimum = min(degrees[value] for value in candidates)
                candidates = [value for value in candidates if degrees[value] <= minimum + 1]
                return rng.choice(candidates), requested_source
        elif requested_source == "graph_bridge":
            candidates = eligible(left_id, ids)
            candidates = [value for value in candidates if union_find.find(value) != union_find.find(left_id)]
            if candidates:
                minimum = min(degrees[value] for value in candidates)
                candidates = [value for value in candidates if degrees[value] <= minimum + 1]
                return rng.choice(candidates), requested_source
        elif requested_source == "low_degree_repair":
            candidates = eligible(left_id, ids)
            if candidates:
                minimum = min(degrees[value] for value in candidates)
                candidates = [value for value in candidates if degrees[value] == minimum]
                return rng.choice(candidates), requested_source
        else:
            candidates = eligible(left_id, ids)
            if candidates:
                return rng.choice(candidates), "random_global"

        candidates = eligible(left_id, ids)
        if not candidates:
            return None, requested_source
        return rng.choice(candidates), "random_global"

    def add_pair(left_id: str, requested_source: str) -> bool:
        if degrees[left_id] >= args.maximum_degree:
            return False
        right_id, source = choose_partner(left_id, requested_source)
        if right_id is None:
            return False
        edge = pair_key(left_id, right_id)
        existing.add(edge)
        degrees[left_id] += 1
        degrees[right_id] += 1
        union_find.union(left_id, right_id)
        pair_digest = hashlib.sha256(f"{next(iter(split_values))}\0{edge[0]}\0{edge[1]}".encode("utf-8")).hexdigest()[:20]
        left, right = by_id[left_id], by_id[right_id]
        left_signature, right_signature = structure_signature(left), structure_signature(right)
        lexical_jaccard = jaccard(ngram_sets[left_id], ngram_sets[right_id])
        pairs.append(
            {
                "schema_version": "raw_v3_pair_candidate_v1",
                "pair_id": f"{next(iter(split_values))}_{pair_digest}",
                "split": next(iter(split_values)),
                "question_a_id": left_id,
                "question_b_id": right_id,
                "question_a_text": left["text"],
                "question_b_text": right["text"],
                "pair_source": source,
                "metadata": {
                    "candidate_index": len(pairs),
                    "lexical_hamming_distance": (fingerprints[left_id] ^ fingerprints[right_id]).bit_count(),
                    "lexical_jaccard": lexical_jaccard,
                    "same_structure_signature": left_signature == right_signature,
                    "length_bucket_a": left_signature[0],
                    "length_bucket_b": right_signature[0],
                    "subquestion_bucket_a": left_signature[1],
                    "subquestion_bucket_b": right_signature[1],
                    "has_image_a": bool((left.get("diagnostics") or {}).get("has_image")),
                    "has_image_b": bool((right.get("diagnostics") or {}).get("has_image")),
                    **(
                        {
                            "feature_hamming_distance": feature_hamming_distance(
                                feature_map[left_id], feature_map[right_id]
                            ),
                            "feature_match_count": len(feature_map[left_id])
                            - feature_hamming_distance(
                                feature_map[left_id], feature_map[right_id]
                            ),
                        }
                        if feature_map is not None
                        else {}
                    ),
                },
            }
        )
        source_counts[source] += 1
        return True

    # First satisfy per-question coverage using the requested source mixture.
    attempts = 0
    while min(degrees[question_id] for question_id in ids) < args.minimum_degree:
        attempts += 1
        if attempts > args.target_pairs * 20:
            raise RuntimeError("unable to satisfy minimum degree")
        candidates = [question_id for question_id in ids if degrees[question_id] < args.minimum_degree]
        candidates.sort(key=lambda value: (degrees[value], stable_rank(args.seed + attempts, value)))
        left_id = candidates[0]
        source = weighted_choice(rng, pair_source_weights)
        if not add_pair(left_id, source) and not add_pair(left_id, "low_degree_repair"):
            raise RuntimeError(f"unable to add coverage edge for {left_id}")

    # Fill the budget. Reserve the final edges for component bridges when needed.
    attempts = 0
    while len(pairs) < args.target_pairs:
        attempts += 1
        if attempts > args.target_pairs * 50:
            raise RuntimeError("unable to satisfy pair budget within degree constraints")
        remaining = args.target_pairs - len(pairs)
        source = "graph_bridge" if union_find.components > 1 and remaining <= union_find.components - 1 else weighted_choice(rng, pair_source_weights)
        candidates = [question_id for question_id in ids if degrees[question_id] < args.maximum_degree]
        if source == "graph_bridge":
            candidates = [question_id for question_id in candidates if union_find.size[union_find.find(question_id)] < count]
        if not candidates:
            continue
        minimum = min(degrees[value] for value in candidates)
        low_degree = [value for value in candidates if degrees[value] <= minimum + 1]
        left_id = rng.choice(low_degree)
        add_pair(left_id, source)

    graph = graph_metrics(pairs, ids)
    if graph["connected_components"] != 1:
        raise RuntimeError(f"comparison graph is disconnected: {graph['connected_components']} components")
    if graph["degree"]["minimum"] < args.minimum_degree or graph["degree"]["maximum"] > args.maximum_degree:
        raise RuntimeError("comparison graph violates degree constraints")

    output = Path(args.output)
    selected_output = Path(args.selected_questions_output)
    manifest_path = Path(args.manifest)
    for path in (output, selected_output, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pairs), encoding="utf-8")
    selected_output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    similarities: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        similarities[pair["pair_source"]].append(float(pair["metadata"]["lexical_jaccard"]))
    mean_similarities = {
        source: sum(values) / len(values)
        for source, values in similarities.items()
        if values
    }
    manifest = {
        "schema_version": "raw_v3_pair_candidates_v1",
        "questions": str(Path(args.questions).resolve()),
        "output": str(output.resolve()),
        "selected_questions_output": str(selected_output.resolve()),
        "split": next(iter(split_values)),
        "seed": args.seed,
        "selection_method": (
            "versioned auxiliary marginal coverage with deterministic deficit repair"
            if feature_map is not None
            else "lowest sha256(seed, question_id)"
        ),
        "question_count": count,
        "requested_pairs": args.target_pairs,
        "created_pairs": len(pairs),
        "minimum_degree_requested": args.minimum_degree,
        "maximum_degree_requested": args.maximum_degree,
        "source_target_weights": pair_source_weights,
        "source_counts": dict(source_counts),
        "mean_lexical_jaccard_by_source": mean_similarities,
        "auxiliary_features_used_for_sampling": feature_map is not None,
        "features_file": str(Path(args.features_file).resolve()) if args.features_file else None,
        "feature_coverage": feature_coverage,
        "maximum_feature_jsd_allowed": args.maximum_feature_jsd,
        "pair_feature_coverage": (
            pair_feature_coverage_report(feature_map, pairs)
            if feature_map is not None
            else None
        ),
        "raw_difficulty_used": False,
        "absolute_difficulty_labels_used": False,
        "old_teacher_labels_used": False,
        "graph": graph,
        "warnings": [],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
