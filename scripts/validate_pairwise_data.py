#!/usr/bin/env python3
"""Validate curated soft-pair data and report graph/label diagnostics."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.data.text_only import forbidden_source_label_paths, leakage_findings
from physics_difficulty.pairwise.metrics import graph_metrics, soft_target_distribution

REQUIRED_KEYS = {
    "pair_id", "split", "question_a_id", "question_b_id",
    "question_a_text", "question_b_text", "soft_target", "sample_weight",
}
def load_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--questions", help="optional text-only question file for complete node coverage")
    parser.add_argument("--output")
    parser.add_argument("--minimum-largest-component-ratio", type=float, default=0.99)
    parser.add_argument("--minimum-node-coverage", type=float, default=0.99)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    errors: list[str] = []
    warnings: list[str] = []
    pair_ids: set[str] = set()
    edge_keys: set[tuple[str, str]] = set()
    split_values: set[str] = set()
    stats: Counter[str] = Counter()
    for index, row in enumerate(rows, 1):
        missing = REQUIRED_KEYS - set(row)
        if missing:
            errors.append(f"line {index}: missing keys {sorted(missing)}")
            continue
        pair_id = str(row["pair_id"])
        if pair_id in pair_ids:
            errors.append(f"line {index}: duplicate pair_id {pair_id}")
        pair_ids.add(pair_id)
        left, right = str(row["question_a_id"]), str(row["question_b_id"])
        if left == right:
            errors.append(f"line {index}: self comparison")
        edge = tuple(sorted((left, right)))
        if edge in edge_keys:
            errors.append(f"line {index}: duplicate unordered question pair")
        edge_keys.add(edge)
        split_values.add(str(row["split"]))
        target, weight = float(row["soft_target"]), float(row["sample_weight"])
        if not math.isfinite(target) or not 0 <= target <= 1:
            errors.append(f"line {index}: soft_target outside [0, 1]")
        if not math.isfinite(weight) or not 0 < weight <= 1:
            errors.append(f"line {index}: sample_weight outside (0, 1]")
        # Wrong historical difficulty is forbidden everywhere except candidate
        # sampling metadata.  It must never enter a student training record.
        forbidden_paths = forbidden_source_label_paths(row)
        if forbidden_paths:
            errors.append(f"line {index}: forbidden historical difficulty field at {forbidden_paths[:5]}")
        for field in ("question_a_text", "question_b_text"):
            text = str(row[field])
            if not text.strip():
                errors.append(f"line {index}: empty {field}")
            if leakage_findings(text):
                errors.append(f"line {index}: difficulty label leakage in {field}")
            if "http://" in text or "https://" in text or "<image>" in text.lower():
                errors.append(f"line {index}: image/URL payload in {field}")
        stats["has_image_metadata"] += bool((row.get("metadata") or {}).get("has_image_a") or (row.get("metadata") or {}).get("has_image_b"))
        stats[f"pair_source_{(row.get('metadata') or {}).get('pair_source', 'unknown')}"] += 1

    if len(split_values) != 1:
        errors.append(f"curated file must contain exactly one split, got {sorted(split_values)}")
    expected_nodes = None
    if args.questions:
        questions = load_jsonl(Path(args.questions))
        expected_nodes = [str(row["id"]) for row in questions]
        question_splits = {str(row["split"]) for row in questions}
        if split_values and question_splits != split_values:
            errors.append(f"question split {sorted(question_splits)} differs from pair split {sorted(split_values)}")
    graph = graph_metrics(rows, expected_nodes)
    if graph["node_coverage"] < args.minimum_node_coverage:
        warnings.append(f"node coverage {graph['node_coverage']:.4f} is below requested {args.minimum_node_coverage:.4f}")
    if graph["largest_component_ratio"] < args.minimum_largest_component_ratio:
        warnings.append(f"largest component ratio {graph['largest_component_ratio']:.4f} is below requested {args.minimum_largest_component_ratio:.4f}")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "schema_version": "physics_pairwise_soft_v1",
        "input": str(Path(args.input).resolve()),
        "records": len(rows),
        "split": next(iter(split_values)) if len(split_values) == 1 else None,
        "images_uploaded": False,
        "raw_difficulty_used": False,
        "errors": errors[:100],
        "error_count": len(errors),
        "warnings": warnings,
        "stats": dict(stats),
        "soft_targets": soft_target_distribution(float(row["soft_target"]) for row in rows if "soft_target" in row),
        "graph": graph,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
