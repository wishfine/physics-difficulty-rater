#!/usr/bin/env python3
"""Select a deterministic simple-random validation sample without labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.data.text_only import (
    forbidden_source_label_paths,
    leakage_findings,
    normalize_for_dedup,
)


FORBIDDEN_FIELDS = {
    "difficulty",
    "raw_difficulty",
    "teacher_difficulty_id",
    "teacher_difficulty_level",
    "teacher_features",
    "teacher_features_legacy18",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, question_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{question_id}".encode("utf-8")).hexdigest()


def nested_forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in FORBIDDEN_FIELDS or str(key).startswith("teacher_difficulty"):
                findings.append(path)
            findings.extend(nested_forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(nested_forbidden_paths(child, f"{prefix}[{index}]"))
    return findings


def jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def diagnostics_report(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = {
        "input_length_bucket": Counter(),
        "has_analysis": Counter(),
        "has_subquestions": Counter(),
        "image_dependency_risk": Counter(),
    }
    for row in rows:
        diagnostics = row.get("diagnostics") or {}
        for field, counts in result.items():
            counts[str(diagnostics.get(field, "unknown"))] += 1
    return {field: dict(counts) for field, counts in result.items()}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Take a deterministic simple-random sample from one already-isolated "
            "question split. No difficulty labels or auxiliary labels are read."
        )
    )
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--records", type=int, required=True)
    parser.add_argument("--expected-split", default="validation")
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    if args.records < 2:
        raise ValueError("--records must be at least two")

    source_path = Path(args.questions)
    accepted: list[dict[str, Any]] = []
    ids: set[str] = set()
    normalized_texts: set[str] = set()
    for line_number, row in enumerate(jsonl_rows(source_path), 1):
        question_id = str(row.get("id") or "").strip()
        if not question_id:
            raise ValueError(f"{source_path}:{line_number}: missing id")
        if str(row.get("split") or "") != args.expected_split:
            raise ValueError(
                f"{source_path}:{line_number}: expected split={args.expected_split!r}, "
                f"got {row.get('split')!r}"
            )
        forbidden = forbidden_source_label_paths(row) + nested_forbidden_paths(row)
        if forbidden:
            raise ValueError(
                f"{source_path}:{line_number}: contains forbidden label fields: "
                f"{sorted(set(forbidden))}"
            )
        text = str(row.get("text") or "").strip()
        if not text:
            raise ValueError(f"{source_path}:{line_number}: empty text")
        if leakage_findings(text):
            raise ValueError(f"{source_path}:{line_number}: label leakage in text")
        normalized = normalize_for_dedup(text)
        if question_id in ids:
            raise ValueError(f"{source_path}:{line_number}: duplicate id {question_id}")
        if normalized in normalized_texts:
            raise ValueError(
                f"{source_path}:{line_number}: duplicate normalized text in source split"
            )
        ids.add(question_id)
        normalized_texts.add(normalized)
        accepted.append(row)

    if len(accepted) < args.records:
        raise ValueError(
            f"validation source has only {len(accepted)} questions; requested {args.records}"
        )
    selected = sorted(
        accepted,
        key=lambda row: (stable_rank(args.seed, str(row["id"])), str(row["id"])),
    )[: args.records]
    write_jsonl(Path(args.output), selected)

    report = {
        "schema_version": "v4_validation_question_selection_v1",
        "selection_method": "lowest_sha256(seed, question_id) within pre-isolated split",
        "selection_interpretation": (
            "deterministic simple-random sample; no quota, difficulty label, or "
            "auxiliary feature influences node inclusion"
        ),
        "seed": args.seed,
        "input": str(source_path.resolve()),
        "input_sha256": sha256_file(source_path),
        "input_split": args.expected_split,
        "source_questions": len(accepted),
        "selected_questions": len(selected),
        "source_diagnostics": diagnostics_report(accepted),
        "selected_diagnostics": diagnostics_report(selected),
        "guardrails": {
            "absolute_difficulty_labels_used": False,
            "auxiliary_features_used_for_node_selection": False,
            "raw_difficulty_used": False,
            "labels_exported": False,
            "input_has_unique_ids": True,
            "input_has_unique_normalized_text": True,
        },
        "output": str(Path(args.output).resolve()),
        "output_sha256": sha256_file(Path(args.output)),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
