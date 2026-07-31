#!/usr/bin/env python3
"""Audit V3 question-node coverage and unlabeled pair-graph construction.

This is intentionally a pre-teacher check: it accepts only label-free V3
questions and never reads difficulty or auxiliary labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.data.text_only import forbidden_source_label_paths
from physics_difficulty.pairwise.metrics import graph_metrics


FORBIDDEN_KEYS = {
    "difficulty", "raw_difficulty", "teacher_difficulty_id", "teacher_difficulty_level",
    "teacher_features", "teacher_features_legacy18", "label_quality", "label_source",
}
PAIR_SOURCES = ("lexical_near", "structure_matched", "random_global", "graph_bridge", "low_degree_repair")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    paths = list(forbidden_source_label_paths(value, prefix))
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in FORBIDDEN_KEYS:
                paths.append(path)
            paths.extend(forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(forbidden_paths(child, f"{prefix}[{index}]"))
    return sorted(set(paths))


def subquestion_bucket(row: dict[str, Any]) -> str:
    diagnostics = row.get("diagnostics") or {}
    try:
        count = int(diagnostics.get("subquestion_count") or 0)
    except (TypeError, ValueError):
        return "unknown"
    return "0" if count == 0 else "1" if count == 1 else "2-3" if count <= 3 else "4+"


def field_value(row: dict[str, Any], name: str) -> str:
    diagnostics = row.get("diagnostics") or {}
    if name == "subquestion_count_bucket":
        return subquestion_bucket(row)
    value = diagnostics.get(name, "unknown")
    return str(value) if value is not None else "unknown"


def distribution(rows: list[dict[str, Any]], name: str) -> Counter[str]:
    return Counter(field_value(row, name) for row in rows)


def distribution_report(pool: list[dict[str, Any]], selected: list[dict[str, Any]], name: str, minimum_pool: int, minimum_selected: int) -> tuple[dict[str, Any], list[str]]:
    pool_counts, selected_counts = distribution(pool, name), distribution(selected, name)
    rows: dict[str, Any] = {}
    warnings: list[str] = []
    for value in sorted(set(pool_counts) | set(selected_counts)):
        pool_count, selected_count = pool_counts[value], selected_counts[value]
        pool_share = pool_count / max(1, len(pool))
        selected_share = selected_count / max(1, len(selected))
        rows[value] = {
            "pool_count": pool_count, "selected_count": selected_count,
            "pool_share": pool_share, "selected_share": selected_share,
            "share_delta": selected_share - pool_share,
        }
        if pool_count >= minimum_pool and selected_count < minimum_selected:
            warnings.append(f"{name}={value}: pool={pool_count}, selected={selected_count} below required {minimum_selected}")
    return rows, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V3 prelabel question and pair coverage")
    parser.add_argument("--pool-questions", required=True, help="Full label-free V3 train pool")
    parser.add_argument("--selected-questions", required=True, help="Candidate graph nodes selected from the pool")
    parser.add_argument("--pairs", required=True, help="Unlabeled pair candidates")
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-degree", type=int, default=6)
    parser.add_argument("--maximum-degree", type=int, default=12)
    parser.add_argument("--minimum-pool-per-stratum", type=int, default=100)
    parser.add_argument("--minimum-selected-per-stratum", type=int, default=25)
    args = parser.parse_args()
    if args.minimum_degree < 1 or args.maximum_degree < args.minimum_degree:
        raise ValueError("invalid degree bounds")
    if args.minimum_pool_per_stratum < 1 or args.minimum_selected_per_stratum < 1:
        raise ValueError("stratum thresholds must be positive")

    pool, selected, pairs = read_jsonl(Path(args.pool_questions)), read_jsonl(Path(args.selected_questions)), read_jsonl(Path(args.pairs))
    pool_by_id = {str(row.get("id") or ""): row for row in pool}
    selected_by_id = {str(row.get("id") or ""): row for row in selected}
    if not pool_by_id or len(pool_by_id) != len(pool):
        raise ValueError("pool questions must have unique non-empty IDs")
    if not selected_by_id or len(selected_by_id) != len(selected):
        raise ValueError("selected questions must have unique non-empty IDs")
    foreign_nodes = set(selected_by_id) - set(pool_by_id)
    if foreign_nodes:
        raise ValueError(f"selected questions are not a subset of the train pool: {len(foreign_nodes)}")
    for name, rows in (("pool", pool), ("selected", selected)):
        split_values = {str(row.get("split")) for row in rows}
        if split_values != {"train"}:
            raise ValueError(f"{name} questions must be train split only, got {sorted(split_values)}")
        for row in rows:
            paths = forbidden_paths(row)
            if paths:
                raise ValueError(f"{name} question {row.get('id')} is not label-free: {paths}")

    coverage_dimensions = (
        "input_length_bucket", "has_subquestions", "subquestion_count_bucket",
        "has_analysis", "has_options", "image_dependency_risk",
    )
    coverage: dict[str, Any] = {}
    coverage_warnings: list[str] = []
    for name in coverage_dimensions:
        values, warnings = distribution_report(pool, selected, name, args.minimum_pool_per_stratum, args.minimum_selected_per_stratum)
        coverage[name] = values
        coverage_warnings.extend(warnings)

    pair_ids: set[str] = set()
    edges: set[tuple[str, str]] = set()
    self_loops = duplicate_pair_ids = duplicate_edges = unknown_endpoints = 0
    source_counts: Counter[str] = Counter()
    similarities: dict[str, list[float]] = defaultdict(list)
    structure_match: Counter[str] = Counter()
    length_pairs: Counter[str] = Counter()
    subquestion_pairs: Counter[str] = Counter()
    for pair in pairs:
        pair_id = str(pair.get("pair_id") or "")
        left, right = str(pair.get("question_a_id") or ""), str(pair.get("question_b_id") or "")
        if pair_id in pair_ids:
            duplicate_pair_ids += 1
        pair_ids.add(pair_id)
        if left == right:
            self_loops += 1
        edge = tuple(sorted((left, right)))
        if edge in edges:
            duplicate_edges += 1
        edges.add(edge)
        if left not in selected_by_id or right not in selected_by_id:
            unknown_endpoints += 1
        source = str(pair.get("pair_source") or "unknown")
        source_counts[source] += 1
        metadata = pair.get("metadata") or {}
        try:
            similarities[source].append(float(metadata.get("lexical_jaccard")))
        except (TypeError, ValueError):
            pass
        structure_match[str(bool(metadata.get("same_structure_signature")))] += 1
        length_pairs["|".join(sorted((str(metadata.get("length_bucket_a", "unknown")), str(metadata.get("length_bucket_b", "unknown")))))] += 1
        subquestion_pairs["|".join(sorted((str(metadata.get("subquestion_bucket_a", "unknown")), str(metadata.get("subquestion_bucket_b", "unknown")))))] += 1

    graph = graph_metrics(pairs, selected_by_id)
    integrity = {
        "pair_records": len(pairs), "unique_pair_ids": len(pair_ids), "unique_edges": len(edges),
        "self_loops": self_loops, "duplicate_pair_ids": duplicate_pair_ids,
        "duplicate_edges": duplicate_edges, "unknown_question_endpoints": unknown_endpoints,
    }
    errors: list[str] = []
    if any(integrity[name] for name in ("self_loops", "duplicate_pair_ids", "duplicate_edges", "unknown_question_endpoints")):
        errors.append("pair integrity violation")
    if graph["node_coverage"] != 1.0 or graph["connected_components"] != 1:
        errors.append("pair graph must cover every selected node and have one connected component")
    if graph["degree"]["minimum"] < args.minimum_degree or graph["degree"]["maximum"] > args.maximum_degree:
        errors.append("pair graph degree bounds violated")
    pair_report = {
        "integrity": integrity, "graph": graph,
        "pair_source_counts": dict(source_counts),
        "missing_configured_pair_sources": [source for source in PAIR_SOURCES if not source_counts[source]],
        "mean_lexical_jaccard_by_source": {source: sum(values) / len(values) for source, values in similarities.items() if values},
        "same_structure_signature": dict(structure_match),
        "length_bucket_pair_matrix": dict(length_pairs),
        "subquestion_bucket_pair_matrix": dict(subquestion_pairs),
    }
    report = {
        "schema_version": "v3_prelabel_pair_construction_audit_v1",
        "status": "FAIL" if errors else ("WARN" if coverage_warnings else "PASS"),
        "errors": errors, "warnings": coverage_warnings,
        "question_selection": {
            "pool_questions": len(pool), "selected_questions": len(selected),
            "selection_rate": len(selected) / max(1, len(pool)), "coverage_by_dimension": coverage,
        },
        "pair_construction": pair_report,
        "guardrails": {
            "teacher_called": False, "raw_difficulty_used": False,
            "absolute_difficulty_labels_used": False, "auxiliary_labels_used": False,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "errors": errors, "warning_count": len(coverage_warnings), "selected_questions": len(selected), "pairs": len(pairs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
