#!/usr/bin/env python3
"""Create immutable text-only question records for pairwise annotation."""
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

from physics_difficulty.data.formatting import canonical_sections, format_question, render_sections
from physics_difficulty.data.text_only import leakage_findings, normalize_text_only, question_group_identifier, question_identifier, source_text
from physics_difficulty.data.truncation import render_with_token_budget


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_tokenizer(path: str | None) -> Any:
    if not path:
        return None
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def truncate(tokenizer: Any, text: str, max_length: int) -> tuple[str, Dict[str, Any]]:
    if tokenizer is None:
        return text, {"tokenizer": None, "max_length": None, "truncated": False, "original_token_count": None, "retained_token_count": None}
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_length:
        return text, {"tokenizer": tokenizer.name_or_path, "max_length": max_length, "truncated": False, "original_token_count": len(tokens), "retained_token_count": len(tokens)}
    retained = tokens[:max_length]
    return tokenizer.decode(retained, skip_special_tokens=True).rstrip(), {
        "tokenizer": tokenizer.name_or_path,
        "max_length": max_length,
        "truncated": True,
        "original_token_count": len(tokens),
        "retained_token_count": len(retained),
    }


def normalized_sections(record: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Return only render fields; never forward arbitrary source metadata."""
    raw_sections = record.get("input_sections") or canonical_sections(record)
    sections = []
    for raw in raw_sections:
        if not isinstance(raw, dict):
            continue
        title = normalize_text_only(raw.get("title"))
        text = normalize_text_only(raw.get("text"))
        if not title and not text:
            continue
        sections.append({
            "kind": str(raw.get("kind") or "unknown"),
            "title": title,
            "text": text,
            "required": bool(raw.get("required")),
            "inline": bool(raw.get("inline")),
        })
    return sections


def safe_feature_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    """Whitelist non-label fields used only to diversify candidate pairs."""
    metadata = record.get("feature_metadata") or {}
    domains = metadata.get("knowledge_domains") or []
    if not isinstance(domains, list):
        domains = [domains]
    return {"knowledge_domains": [str(value) for value in domains if str(value).strip()]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--quarantine-output")
    parser.add_argument("--split", required=True, choices=["train", "validation", "test", "pilot", "reference"])
    parser.add_argument("--student-tokenizer-path")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--allow-label-leakage", action="store_true")
    args = parser.parse_args()
    if args.max_length <= 0:
        raise ValueError("max-length must be positive")

    source_path = Path(args.input)
    output_path = Path(args.output)
    quarantine_path = Path(args.quarantine_output) if args.quarantine_output else output_path.with_suffix(".quarantine.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.student_tokenizer_path)
    accepted: list[Dict[str, Any]] = []
    quarantined: list[Dict[str, Any]] = []
    stats: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()

    with source_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            stats["source_records"] += 1
            record = json.loads(line)
            try:
                question_id = question_identifier(record)
            except ValueError as error:
                raise ValueError(f"line {line_number}: {error}") from error
            sections = normalized_sections(record)
            if sections:
                clean_text = render_sections(sections)
            else:
                raw_text = source_text(record, format_question)
                clean_text = normalize_text_only(raw_text)
            if not clean_text:
                stats["empty_text"] += 1
                quarantined.append({"id": question_id, "reason": "empty_text", "source_line": line_number})
                continue
            findings = leakage_findings(clean_text)
            if tokenizer is not None and sections:
                rendered, truncation = render_with_token_budget(sections, tokenizer, args.max_length)
                truncation["tokenizer"] = tokenizer.name_or_path
                truncation["max_length"] = args.max_length
            else:
                rendered, truncation = truncate(tokenizer, clean_text, args.max_length)
            digest = sha256_text(rendered)
            item = {
                "id": question_id,
                "question_group_id": question_group_identifier(record, question_id),
                "split": args.split,
                "text": rendered,
                "text_sha256": digest,
                "text_schema_version": "text_only_v1",
                "source_line": line_number,
                "teacher_difficulty_id": record.get("teacher_difficulty_id"),
                "teacher_difficulty_level": record.get("teacher_difficulty_level"),
                # The old API label is retained only as a candidate-sampling
                # stratum. It is never shown to the judge or student model.
                "feature_metadata": safe_feature_metadata(record),
                "teacher_features": {
                    "problem_structure": (record.get("teacher_features") or {}).get("problem_structure"),
                },
                "diagnostics": {
                    "has_image": bool((record.get("diagnostics") or {}).get("has_image_url")),
                    "images_uploaded": False,
                    "label_leakage_findings": findings,
                    "truncation": truncation,
                },
            }
            if question_id in seen_ids:
                stats["duplicate_id"] += 1
                quarantined.append(item | {"reason": "duplicate_id"})
                continue
            seen_ids.add(question_id)
            if digest in seen_hashes:
                stats["duplicate_rendered_text"] += 1
                quarantined.append(item | {"reason": "duplicate_rendered_text"})
                continue
            seen_hashes.add(digest)
            if findings and not args.allow_label_leakage:
                stats["label_leakage"] += 1
                quarantined.append(item | {"reason": "label_leakage"})
                continue
            accepted.append(item)
            stats["accepted"] += 1
            stats["truncated"] += bool(truncation["truncated"])
            stats["has_image_metadata"] += item["diagnostics"]["has_image"]

    with output_path.open("w", encoding="utf-8") as target:
        for item in accepted:
            target.write(json.dumps(item, ensure_ascii=False) + "\n")
    with quarantine_path.open("w", encoding="utf-8") as target:
        for item in quarantined:
            target.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": "text_only_questions_v1",
        "input": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "quarantine_output": str(quarantine_path.resolve()),
        "split": args.split,
        "images_uploaded": False,
        "student_tokenizer_path": args.student_tokenizer_path,
        "max_length": args.max_length if tokenizer is not None else None,
        "allow_label_leakage": args.allow_label_leakage,
        "stats": dict(stats),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
