"""Dataset for text-only soft Bradley-Terry training and evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from physics_difficulty.data.text_only import forbidden_source_label_paths, leakage_findings
from physics_difficulty.schema import FEATURE_TO_ID


class PairwiseDifficultyDataset(Dataset):
    def __init__(self, path: str, tokenizer: Any, max_length: int, require_auxiliary_features: bool = False):
        self.items = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.items:
            raise ValueError("pairwise dataset is empty")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.require_auxiliary_features = require_auxiliary_features
        self.tokenizer.padding_side = "right"
        self.question_degrees: Dict[str, int] = {}
        feature_counts = {name: [0] * len(value_to_id) for name, value_to_id in FEATURE_TO_ID.items()}
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
            for id_field in ("question_a_id", "question_b_id"):
                question_id = str(item[id_field])
                self.question_degrees[question_id] = self.question_degrees.get(question_id, 0) + 1
            if require_auxiliary_features:
                auxiliary = item.get("auxiliary_features")
                qualities = item.get("auxiliary_feature_quality")
                if not isinstance(auxiliary, dict) or not isinstance(qualities, dict):
                    raise ValueError("V2 item is missing auxiliary_features or auxiliary_feature_quality")
                for side in ("question_a", "question_b"):
                    side_features = auxiliary.get(side)
                    quality = float(qualities.get(side, 0.0))
                    if side_features is None:
                        if quality != 0:
                            raise ValueError("missing auxiliary labels must have zero quality")
                        continue
                    if not 0 < quality <= 1:
                        raise ValueError("auxiliary feature quality must be in (0, 1]")
                    if set(side_features) != set(FEATURE_TO_ID):
                        raise ValueError("auxiliary feature names do not match the frozen ten-dimensional schema")
                    for name, value_to_id in FEATURE_TO_ID.items():
                        value = side_features[name]
                        if value not in value_to_id:
                            raise ValueError(f"invalid auxiliary feature {name}={value!r}")
                        feature_counts[name][value_to_id[value]] += 1
        self.feature_class_weights: Dict[str, torch.Tensor] = {}
        if require_auxiliary_features:
            for name, counts in feature_counts.items():
                count_tensor = torch.tensor(counts, dtype=torch.float32).clamp_min(1)
                weights = (count_tensor.rsqrt() / count_tensor.rsqrt().mean()).clamp(0.5, 2.0)
                self.feature_class_weights[name] = weights / weights.mean()

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
        result = {
            **encoded,
            "pair_ids": [str(item["pair_id"]) for item in batch],
            "question_a_ids": [str(item["question_a_id"]) for item in batch],
            "question_b_ids": [str(item["question_b_id"]) for item in batch],
            "pair_count": len(batch),
            "soft_targets": torch.tensor([float(item["soft_target"]) for item in batch], dtype=torch.float32),
            "sample_weights": torch.tensor([float(item.get("sample_weight", 1.0)) for item in batch], dtype=torch.float32),
            "metadata": [item.get("metadata") or {} for item in batch],
        }
        if self.require_auxiliary_features:
            targets_a = {name: [] for name in FEATURE_TO_ID}
            targets_b = {name: [] for name in FEATURE_TO_ID}
            weights_a: List[float] = []
            weights_b: List[float] = []
            for item in batch:
                for side, targets, weights, id_field in (
                    ("question_a", targets_a, weights_a, "question_a_id"),
                    ("question_b", targets_b, weights_b, "question_b_id"),
                ):
                    side_features = item["auxiliary_features"].get(side)
                    quality = float(item["auxiliary_feature_quality"].get(side, 0.0))
                    degree = self.question_degrees[str(item[id_field])]
                    weights.append(quality / degree if side_features is not None else 0.0)
                    for name, value_to_id in FEATURE_TO_ID.items():
                        targets[name].append(value_to_id[side_features[name]] if side_features is not None else -100)
            result.update({
                "auxiliary_targets_a": {name: torch.tensor(values, dtype=torch.long) for name, values in targets_a.items()},
                "auxiliary_targets_b": {name: torch.tensor(values, dtype=torch.long) for name, values in targets_b.items()},
                "auxiliary_weights_a": torch.tensor(weights_a, dtype=torch.float32),
                "auxiliary_weights_b": torch.tensor(weights_b, dtype=torch.float32),
            })
        return result
