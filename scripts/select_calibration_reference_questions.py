#!/usr/bin/env python3
"""Select a frozen source-distribution reference set without using labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
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
        with Path(value).open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                for key in ("id", "question_id", "question_a_id", "question_b_id"):
                    if row.get(key) is not None:
                        ids.add(str(row[key]))
                for key in ("text", "question_a_text", "question_b_text"):
                    if str(row.get(key) or "").strip():
                        texts.add(normalize_for_dedup(row[key]))
    return ids, texts


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


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
    args = parser.parse_args()
    if not 0 < args.smoke_records <= args.records:
        raise ValueError("require 0 < smoke-records <= records")

    excluded_ids, excluded_texts = exclusions(args.exclude)
    accepted: list[tuple[str, dict[str, Any]]] = []
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
            group_id = question_group_identifier(row, question_id)
            accepted.append((stable_key(args.seed, group_id), row))
    accepted.sort(key=lambda item: (item[0], question_identifier(item[1])))
    if len(accepted) < args.records:
        raise ValueError(f"only {len(accepted)} eligible records; requested {args.records}")
    selected = [row for _, row in accepted[: args.records]]
    smoke = selected[: args.smoke_records]
    output = Path(args.output)
    smoke_output = Path(args.smoke_output)
    manifest = Path(args.manifest)
    write_jsonl(output, selected)
    write_jsonl(smoke_output, smoke)
    report = {
        "schema_version": "single_question_calibration_reference_v1",
        "selection_method": "lowest_sha256(seed, existing_question_group_id)",
        "seed": args.seed,
        "source_provenance": args.source_provenance,
        "distribution_claim": "source_distribution_only; natural-business status requires external provenance",
        "labels_used": False,
        "input": str(Path(args.input).resolve()),
        "input_sha256": sha256_file(Path(args.input)),
        "excluded_files": [str(Path(value).resolve()) for value in args.exclude],
        "stats": dict(counters),
        "eligible_records": len(accepted),
        "selected_records": len(selected),
        "smoke_records": len(smoke),
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
