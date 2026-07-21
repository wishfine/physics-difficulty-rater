from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from physics_difficulty.data.truncation import render_with_token_budget
from physics_difficulty.schema import FEATURE_TO_ID


class DifficultyDataset(Dataset):
    """Versioned training/evaluation data with shared section-aware truncation."""
    def __init__(self, path: str, tokenizer: Any, max_length: int, require_labels: bool = True):
        self.items = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.tokenizer, self.max_length, self.require_labels = tokenizer, max_length, require_labels
        self.tokenizer.padding_side = "right"
        # Rendering with the section-aware truncator calls the tokenizer several
        # times for long questions. Inputs are immutable within a run, so cache
        # each rendered result and avoid repeating this CPU work every epoch.
        self._render_cache: Dict[int, tuple[str, Dict[str, Any]]] = {}
        if require_labels:
            for item in self.items:
                if "teacher_difficulty_id" not in item and "difficulty_id" not in item:
                    raise ValueError("Training/evaluation item is missing teacher_difficulty_id")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.items[index]

    def _render(self, item: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        sections = item.get("input_sections")
        if sections:
            return render_with_token_budget(sections, self.tokenizer, self.max_length)
        text = item["text"]
        token_count = len(self.tokenizer.encode(text, add_special_tokens=False))
        if token_count <= self.max_length:
            return text, {"truncated": False, "original_token_count": token_count, "retained_token_count": token_count, "truncation_strategy_version": "legacy"}
        tokens = self.tokenizer.encode(text, add_special_tokens=False)[: self.max_length]
        return self.tokenizer.decode(tokens, skip_special_tokens=True), {"truncated": True, "original_token_count": token_count, "retained_token_count": self.max_length, "truncation_strategy_version": "legacy"}

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        rendered = []
        for item in batch:
            cache_key = id(item)
            cached = self._render_cache.get(cache_key)
            if cached is None:
                cached = self._render(item)
                self._render_cache[cache_key] = cached
            rendered.append(cached)
        encoded = self.tokenizer([item[0] for item in rendered], truncation=False, padding=True, return_tensors="pt")
        result = {
            **encoded,
            "ids": [str(item.get("id", "")) for item in batch],
            "metadata": [item.get("diagnostics", {}) | {"source_dataset_id": item.get("source_dataset_id"), "parent_id": item.get("parent_id")} for item in batch],
            "truncation": [item[1] for item in rendered],
        }
        if not self.require_labels:
            return result

        feature_labels = {}
        for name, value_to_id in FEATURE_TO_ID.items():
            values = [item["teacher_features"][name] for item in batch]
            feature_labels[name] = torch.tensor([value_to_id[value] for value in values], dtype=torch.long)
        result.update({
            "difficulty_labels": torch.tensor([item.get("teacher_difficulty_id", item.get("difficulty_id")) for item in batch], dtype=torch.long),
            "sample_weights": torch.tensor([item["label_quality"]["sample_weight"] for item in batch], dtype=torch.float32),
            "feature_labels": feature_labels,
        })
        return result
