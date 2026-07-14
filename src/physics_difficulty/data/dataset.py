from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
import torch
from torch.utils.data import Dataset
from physics_difficulty.schema import FEATURE_TO_ID, MULTI_LABEL_FEATURES

class DifficultyDataset(Dataset):
    def __init__(self, path: str, tokenizer: Any, max_length: int):
        self.items = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.tokenizer, self.max_length = tokenizer, max_length
        self.tokenizer.padding_side = "right"

    def __len__(self) -> int: return len(self.items)
    def __getitem__(self, index: int) -> Dict[str, Any]: return self.items[index]

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        encoded = self.tokenizer([item["text"] for item in batch], truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
        feature_labels = {}
        for name, value_to_id in FEATURE_TO_ID.items():
            values = [item["teacher_features"][name] for item in batch]
            if name in MULTI_LABEL_FEATURES:
                feature_labels[name] = torch.tensor(
                    [[float(tag in values_for_item) for tag in value_to_id] for values_for_item in values], dtype=torch.float32
                )
            else:
                feature_labels[name] = torch.tensor([value_to_id[value] for value in values], dtype=torch.long)
        return {**encoded, "difficulty_labels": torch.tensor([item["difficulty_id"] for item in batch], dtype=torch.long), "sample_weights": torch.tensor([item["label_quality"]["sample_weight"] for item in batch], dtype=torch.float32), "feature_labels": feature_labels}
