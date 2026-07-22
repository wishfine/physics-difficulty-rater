#!/usr/bin/env python3
"""Prepare and split the raw 25k physics source without using its labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.data.formatting import canonical_sections, diagnostics, render_sections, sorted_subquestions
from physics_difficulty.data.text_only import (
    has_semantic_content,
    leakage_findings,
    normalize_for_dedup,
    normalize_text_only,
    question_group_identifier,
    question_identifier,
)


RAW25K_REQUIRED_FIELDS = {
    "parent_id",
    "question_id",
    "stem",
    "options",
    "analysis",
    "sub_questions",
    "stem_pic_url",
    "analysis_pic_url",
    "difficulty",
}
SPLITS = ("train", "validation", "test")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_raw25k(record: Dict[str, Any], line_number: int) -> None:
    missing = RAW25K_REQUIRED_FIELDS - set(record)
    if missing:
        raise ValueError(f"line {line_number}: raw25k input is missing fields {sorted(missing)}")
    if not str(record.get("question_id") or "").strip():
        raise ValueError(f"line {line_number}: raw25k question_id is empty")
    if not str(record.get("parent_id") or "").strip():
        raise ValueError(f"line {line_number}: raw25k parent_id is empty")
    if not isinstance(record.get("sub_questions"), list):
        raise ValueError(f"line {line_number}: raw25k sub_questions must be a list")


def clean_sections(record: Dict[str, Any]) -> list[Dict[str, Any]]:
    result = []
    for raw in canonical_sections(record):
        title = normalize_text_only(raw.get("title"))
        text = normalize_text_only(raw.get("text"))
        if not title and not text:
            continue
        result.append(
            {
                "kind": str(raw.get("kind") or "unknown"),
                "title": title,
                "text": text,
                "required": bool(raw.get("required")),
                "inline": bool(raw.get("inline")),
            }
        )
    return result


def stable_split(group_id: str, seed: int, train_ratio: float, validation_ratio: float) -> str:
    digest = hashlib.sha256(f"{seed}\0{group_id}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    if fraction < train_ratio:
        return "train"
    if fraction < train_ratio + validation_ratio:
        return "validation"
    return "test"


def build_diagnostics(record: Dict[str, Any], text: str) -> Dict[str, Any]:
    base = diagnostics(record, text)
    children = sorted_subquestions(record)
    return {
        "has_analysis": base["has_analysis"],
        "has_options": bool(normalize_text_only(record.get("options")))
        or any(bool(normalize_text_only(child.get("options"))) for child in children if isinstance(child, dict)),
        "has_subquestions": base["has_subquestions"],
        "subquestion_count": len(children),
        "has_image": base["has_image_url"],
        "has_image_url": base["has_image_url"],
        "has_image_reference_marker": base["has_image_reference_marker"],
        "image_dependency_risk": base["image_dependency_risk"],
        "images_uploaded": False,
        "char_length": base["char_length"],
        "input_length_bucket": base["input_length_bucket"],
    }


def write_jsonl(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--allow-label-leakage", action="store_true")
    args = parser.parse_args()
    if not 0 < args.train_ratio < 1 or not 0 < args.validation_ratio < 1:
        raise ValueError("split ratios must be positive")
    if args.train_ratio + args.validation_ratio >= 1:
        raise ValueError("train-ratio plus validation-ratio must be less than one")

    source_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    accepted: list[Dict[str, Any]] = []
    quarantined: list[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_text: dict[str, str] = {}

    with source_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            stats["source_records"] += 1
            record = json.loads(line)
            validate_raw25k(record, line_number)
            question_id = question_identifier(record)
            group_id = question_group_identifier(record, question_id)
            sections = clean_sections(record)
            text = normalize_text_only(render_sections(sections))
            if not has_semantic_content(text):
                stats["semantically_empty"] += 1
                quarantined.append({"id": question_id, "reason": "semantically_empty", "source_line": line_number})
                continue
            findings = leakage_findings(text)
            if findings and not args.allow_label_leakage:
                stats["label_leakage"] += 1
                quarantined.append(
                    {
                        "id": question_id,
                        "reason": "label_leakage",
                        "source_line": line_number,
                        "findings": findings,
                    }
                )
                continue
            if question_id in seen_ids:
                stats["duplicate_id"] += 1
                quarantined.append({"id": question_id, "reason": "duplicate_id", "source_line": line_number})
                continue
            normalized_text = normalize_for_dedup(text)
            normalized_digest = sha256_text(normalized_text)
            if normalized_digest in seen_text:
                stats["duplicate_normalized_text"] += 1
                quarantined.append(
                    {
                        "id": question_id,
                        "reason": "duplicate_normalized_text",
                        "duplicate_of": seen_text[normalized_digest],
                        "source_line": line_number,
                    }
                )
                continue
            split = stable_split(group_id, args.seed, args.train_ratio, args.validation_ratio)
            item = {
                "id": question_id,
                "parent_id": str(record["parent_id"]),
                "question_group_id": group_id,
                "split": split,
                "text": text,
                "input_sections": sections,
                "text_sha256": sha256_text(text),
                "normalized_text_sha256": normalized_digest,
                "text_schema_version": "v3_text_only_raw25k_v1",
                "diagnostics": build_diagnostics(record, text),
                "source_provenance": {
                    "source_file": source_path.name,
                    "source_line": line_number,
                    "source_question_id": question_id,
                },
            }
            seen_ids.add(question_id)
            seen_text[normalized_digest] = question_id
            accepted.append(item)
            stats["accepted"] += 1
            stats[f"split_{split}"] += 1
            stats["has_analysis"] += item["diagnostics"]["has_analysis"]
            stats["has_subquestions"] += item["diagnostics"]["has_subquestions"]
            stats["has_image_metadata"] += item["diagnostics"]["has_image"]
            stats[f"image_dependency_{item['diagnostics']['image_dependency_risk']}"] += 1
            stats[f"length_{item['diagnostics']['input_length_bucket']}"] += 1

    split_rows = {name: [row for row in accepted if row["split"] == name] for name in SPLITS}
    write_jsonl(output_dir / "all.jsonl", accepted)
    for name, rows in split_rows.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    write_jsonl(output_dir / "quarantine.jsonl", quarantined)

    manifest = {
        "schema_version": "v3_raw25k_preparation_v1",
        "input": str(source_path.resolve()),
        "input_sha256": sha256_file(source_path),
        "output_dir": str(output_dir.resolve()),
        "seed": args.seed,
        "split_ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": round(1 - args.train_ratio - args.validation_ratio, 12),
        },
        "split_method": "sha256(seed, question_group_id)",
        "deduplication_key": "sha256(NFKC(normalized rendered text))",
        "images_uploaded": False,
        "forbidden_source_fields": ["difficulty"],
        "raw_difficulty_used": False,
        "known_sampling_bias": "source file was historically sampled as 5000 records per unusable difficulty value",
        "allow_label_leakage": args.allow_label_leakage,
        "stats": dict(stats),
        "outputs": {
            "all": str((output_dir / "all.jsonl").resolve()),
            "train": str((output_dir / "train.jsonl").resolve()),
            "validation": str((output_dir / "validation.jsonl").resolve()),
            "test": str((output_dir / "test.jsonl").resolve()),
            "quarantine": str((output_dir / "quarantine.jsonl").resolve()),
        },
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
