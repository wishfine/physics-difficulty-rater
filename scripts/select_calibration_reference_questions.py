#!/usr/bin/env python3
"""Select a frozen calibration reference while keeping labels out of model input."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.data.text_only import (
    forbidden_source_label_paths,
    leakage_findings,
    normalize_for_dedup,
    question_group_identifier,
    question_identifier,
)
from physics_difficulty.schema import DIFFICULTY_LEVELS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def exclusions(paths: list[str]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    texts: set[str] = set()
    for value in paths:
        path = Path(value)
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))
        else:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        for row in rows:
            for key in ("id", "question_id", "question_a_id", "question_b_id", "题目ID"):
                if row.get(key) is not None and str(row[key]).strip():
                    ids.add(str(row[key]).strip())
            for key in ("text", "question_a_text", "question_b_text"):
                if str(row.get(key) or "").strip():
                    texts.add(normalize_for_dedup(row[key]))
    return ids, texts


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def proportional_quotas(counts: Counter[str], total: int, order: list[str]) -> dict[str, int]:
    population = sum(counts.values())
    if population <= 0:
        raise ValueError("stratification population is empty")
    exact = {value: total * counts[value] / population for value in order}
    quotas = {value: int(exact[value]) for value in order}
    remaining = total - sum(quotas.values())
    ranked = sorted(
        order,
        key=lambda value: (-(exact[value] - quotas[value]), order.index(value)),
    )
    for value in ranked[:remaining]:
        quotas[value] += 1
    return quotas


def load_stratification_labels(path: Path, field: str) -> tuple[dict[str, str], Counter[str]]:
    labels: dict[str, str] = {}
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = question_identifier(row)
            if question_id in labels:
                raise ValueError(f"{path}: line {line_number} duplicates question {question_id}")
            value = str(row.get(field) or "").strip()
            if value not in DIFFICULTY_LEVELS:
                raise ValueError(f"{path}: line {line_number} has invalid {field}={value!r}")
            labels[question_id] = value
            counts[value] += 1
    if not labels:
        raise ValueError("stratification label file is empty")
    return labels, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--records", type=int, default=10000)
    parser.add_argument("--smoke-records", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--source-provenance", default="unknown")
    parser.add_argument(
        "--stratification-labels",
        help="Private teacher-label JSONL used only to preserve its natural five-level proportions.",
    )
    parser.add_argument("--stratification-field", default="teacher_difficulty_level")
    parser.add_argument("--business-natural-distribution", action="store_true")
    args = parser.parse_args()
    if not 0 < args.smoke_records <= args.records:
        raise ValueError("require 0 < smoke-records <= records")

    excluded_ids, excluded_texts = exclusions(args.exclude)
    labels: dict[str, str] = {}
    population_counts: Counter[str] = Counter()
    if args.stratification_labels:
        labels, population_counts = load_stratification_labels(
            Path(args.stratification_labels), args.stratification_field
        )
    elif args.business_natural_distribution:
        raise ValueError("--business-natural-distribution requires --stratification-labels")

    accepted: list[tuple[str, dict[str, Any], str | None]] = []
    counters: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    with Path(args.input).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            counters["source_records"] += 1
            row = json.loads(line)
            question_id = question_identifier(row)
            if question_id in seen_ids:
                raise ValueError(f"line {line_number}: duplicate question id {question_id}")
            seen_ids.add(question_id)
            if forbidden_source_label_paths(row):
                counters["forbidden_label"] += 1
                continue
            text = str(row.get("text") or "").strip()
            if not text:
                counters["empty_text"] += 1
                continue
            if leakage_findings(text):
                counters["label_leakage"] += 1
                continue
            normalized = normalize_for_dedup(text)
            if question_id in excluded_ids or normalized in excluded_texts:
                counters["excluded_overlap"] += 1
                continue
            if normalized in seen_texts:
                counters["duplicate_text"] += 1
                continue
            seen_texts.add(normalized)
            label = labels.get(question_id) if labels else None
            if labels and label is None:
                counters["missing_stratification_label"] += 1
                continue
            group_id = question_group_identifier(row, question_id)
            reference_row = dict(row)
            reference_row["source_split"] = row.get("split")
            reference_row["split"] = "calibration_reference"
            accepted.append((stable_key(args.seed, group_id), reference_row, label))
    accepted.sort(key=lambda item: (item[0], question_identifier(item[1])))
    if len(accepted) < args.records:
        raise ValueError(f"only {len(accepted)} eligible records; requested {args.records}")

    selected_entries: list[tuple[str, dict[str, Any], str | None]]
    selected_counts: Counter[str] = Counter()
    smoke_counts: Counter[str] = Counter()
    record_quotas: dict[str, int] = {}
    smoke_quotas: dict[str, int] = {}
    if labels:
        order = list(DIFFICULTY_LEVELS)
        record_quotas = proportional_quotas(population_counts, args.records, order)
        smoke_quotas = proportional_quotas(population_counts, args.smoke_records, order)
        pools: dict[str, list[tuple[str, dict[str, Any], str | None]]] = defaultdict(list)
        for entry in accepted:
            pools[str(entry[2])].append(entry)
        unavailable = {
            value: {"required": record_quotas[value], "available": len(pools[value])}
            for value in order
            if len(pools[value]) < record_quotas[value]
        }
        if unavailable:
            raise ValueError(f"eligible strata cannot satisfy natural-distribution quotas: {unavailable}")
        selected_entries = [
            entry
            for value in order
            for entry in pools[value][: record_quotas[value]]
        ]
        smoke_entries = [
            entry
            for value in order
            for entry in pools[value][: smoke_quotas[value]]
        ]
        selected_entries.sort(key=lambda item: (item[0], question_identifier(item[1])))
        smoke_entries.sort(key=lambda item: (item[0], question_identifier(item[1])))
        selected_counts.update(str(entry[2]) for entry in selected_entries)
        smoke_counts.update(str(entry[2]) for entry in smoke_entries)
    else:
        selected_entries = accepted[: args.records]
        smoke_entries = selected_entries[: args.smoke_records]
    selected = [row for _, row, _ in selected_entries]
    smoke = [row for _, row, _ in smoke_entries]
    selected_source_splits = Counter(str(row.get("source_split") or "unspecified") for row in selected)
    smoke_source_splits = Counter(str(row.get("source_split") or "unspecified") for row in smoke)
    output = Path(args.output)
    smoke_output = Path(args.smoke_output)
    manifest = Path(args.manifest)
    write_jsonl(output, selected)
    write_jsonl(smoke_output, smoke)
    report = {
        "schema_version": "single_question_calibration_reference_v1",
        "selection_method": (
            "natural_five_level_proportional_strata_then_lowest_sha256"
            if labels
            else "lowest_sha256(seed, existing_question_group_id)"
        ),
        "seed": args.seed,
        "source_provenance": args.source_provenance,
        "distribution_claim": (
            "business_natural_distribution"
            if args.business_natural_distribution
            else "source_distribution_only; natural-business status requires external provenance"
        ),
        "labels_used_for_proportional_stratification": bool(labels),
        "labels_exported": False,
        "labels_used_for_threshold_fitting": False,
        "input": str(Path(args.input).resolve()),
        "input_sha256": sha256_file(Path(args.input)),
        "excluded_files": [str(Path(value).resolve()) for value in args.exclude],
        "stratification": (
            {
                "labels_file": str(Path(args.stratification_labels).resolve()),
                "labels_file_sha256": sha256_file(Path(args.stratification_labels)),
                "field": args.stratification_field,
                "population_counts": dict(population_counts),
                "population_proportions": {
                    value: population_counts[value] / sum(population_counts.values())
                    for value in DIFFICULTY_LEVELS
                },
                "reference_quotas": record_quotas,
                "reference_counts": dict(selected_counts),
                "smoke_quotas": smoke_quotas,
                "smoke_counts": dict(smoke_counts),
            }
            if labels
            else None
        ),
        "stats": dict(counters),
        "eligible_records": len(accepted),
        "selected_records": len(selected),
        "smoke_records": len(smoke),
        "selected_source_split_counts": dict(selected_source_splits),
        "smoke_source_split_counts": dict(smoke_source_splits),
        "output_split": "calibration_reference",
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "smoke_output": str(smoke_output.resolve()),
        "smoke_output_sha256": sha256_file(smoke_output),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
