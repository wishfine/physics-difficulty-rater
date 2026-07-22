"""Dataset for text-only soft Bradley-Terry training and evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from physics_difficulty.data.text_only import forbidden_source_label_paths, leakage_findings


class PairwiseDifficultyDataset(Dataset):
    def __init__(self, path: str, tokenizer: Any, max_length: int):
        self.items = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.items:
            raise ValueError("pairwise dataset is empty")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.padding_side = "right"
        for item in self.items:
            forbidden = forbidden_source_label_paths(item)
            if forbidden:
                raise ValueError(f"pairwise training item contains forbidden historical label fields: {forbidden[:5]}")
            target = float(item["soft_target"])
            weight = float(item.get("sample_weight", 1.0))
            if not 0 <= target <= 1:
                raise ValueError("soft_target must be in [0, 1]")
            if not 0 < weight <= 1:
                raise ValueError("sample_weight must be in (0, 1]")
            if not str(item["question_a_text"]).strip() or not str(item["question_b_text"]).strip():
                raise ValueError("pairwise questions must contain non-empty text")
            if leakage_findings(str(item["question_a_text"])) or leakage_findings(str(item["question_b_text"])):
                raise ValueError("pairwise question text contains an explicit difficulty label")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.items[index]

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [str(item["question_a_text"]) for item in batch] + [str(item["question_b_text"]) for item in batch]
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        return {
            **encoded,
            "pair_ids": [str(item["pair_id"]) for item in batch],
            "question_a_ids": [str(item["question_a_id"]) for item in batch],
            "question_b_ids": [str(item["question_b_id"]) for item in batch],
            "pair_count": len(batch),
            "soft_targets": torch.tensor([float(item["soft_target"]) for item in batch], dtype=torch.float32),
            "sample_weights": torch.tensor([float(item.get("sample_weight", 1.0)) for item in batch], dtype=torch.float32),
            "metadata": [item.get("metadata") or {} for item in batch],
        }
