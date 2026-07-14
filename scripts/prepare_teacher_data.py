#!/usr/bin/env python3
"""Convert API+V7 result JSONL to immutable V2 training JSONL."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from physics_difficulty.data.formatting import diagnostics, format_question
from physics_difficulty.data.quality import score_label_quality
from physics_difficulty.schema import difficulty_id, normalize_v2_features

def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="API+postprocess JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    seen, stats = set(), Counter()
    with open(args.input, encoding="utf-8") as source, open(output, "w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            rating = record.get("difficulty_rating") or {}
            level = rating.get("difficulty_level")
            if not level:
                stats["missing_teacher_label"] += 1; continue
            text = format_question(record)
            digest = text_hash(text)
            if digest in seen:
                stats["exact_duplicate"] += 1; continue
            seen.add(digest)
            features = normalize_v2_features(rating.get("features"))
            quality = score_label_quality(level, features, record)
            item = {
                "id": str(record.get("question_id", line_number)), "parent_id": str(record.get("parent_id", record.get("question_id", line_number))),
                "text": text, "text_sha256": digest, "difficulty_level": level, "difficulty_id": difficulty_id(level),
                "raw_difficulty": record.get("difficulty"), "teacher_features": features, "label_source": "api_v7",
                "diagnostics": diagnostics(record, text), "label_quality": quality,
            }
            target.write(json.dumps(item, ensure_ascii=False) + "\n")
            stats[f"level_{level}"] += 1; stats[f"quality_{quality['label_quality']}"] += 1
    manifest = {"input": str(Path(args.input).resolve()), "output": str(output.resolve()), "schema_version": "v2", "records": sum(v for k, v in stats.items() if k.startswith("level_")), "stats": dict(stats)}
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
