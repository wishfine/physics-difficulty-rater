#!/usr/bin/env python3
"""Score text-only questions with the learned scalar difficulty function."""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.data.text_only import forbidden_source_label_paths, leakage_findings
from physics_difficulty.models.pairwise_loading import load_pairwise_rater

LEVELS = ["送分题", "基础题", "中等题", "拔高题", "压轴题"]


class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows:
            raise ValueError("question file is empty")
        for row in self.rows:
            if forbidden_source_label_paths(row):
                raise ValueError(f"question {row.get('id')} contains forbidden historical difficulty")
            if not str(row.get("text") or "").strip():
                raise ValueError(f"question {row.get('id')} has empty text")
            if leakage_findings(str(row["text"])):
                raise ValueError(f"question {row.get('id')} contains explicit difficulty leakage")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.max_length <= 0 or args.batch_size <= 0:
        raise ValueError("max-length and batch-size must be positive")

    calibration = None
    if args.calibration:
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        if calibration.get("checkpoint_dir") != str(Path(args.checkpoint_dir).resolve()):
            raise ValueError("calibration was produced by a different checkpoint")
        if len(calibration.get("thresholds", [])) != 4:
            raise ValueError("calibration must contain exactly four thresholds")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_pairwise_rater(args.model_path, args.checkpoint_dir, device, args.bf16)
    dataset = QuestionDataset(args.questions)

    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer([str(row["text"]) for row in rows], truncation=True, max_length=args.max_length, padding=True, return_tensors="pt")
        return {"rows": rows, **encoded}

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as target, torch.no_grad():
        for batch in loader:
            scores = model.score(batch["input_ids"].to(device), batch["attention_mask"].to(device)).float().cpu().tolist()
            for row, score in zip(batch["rows"], scores):
                result = {"id": str(row["id"]), "split": row.get("split"), "difficulty_score": score}
                if calibration:
                    level_id = bisect.bisect_right(calibration["thresholds"], score)
                    result.update({"difficulty_id": level_id, "difficulty_level": LEVELS[level_id]})
                target.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(dataset), "output": str(output.resolve()), "calibrated": calibration is not None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
