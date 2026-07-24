import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.schema import FEATURE_TO_ID, FEATURE_VALUES

try:
    import torch
    import torch.nn as nn
    from physics_difficulty.data.pairwise_dataset import PairwiseDifficultyDataset
    from physics_difficulty.models.qwen_pairwise import QwenPairwiseRater
    from physics_difficulty.pairwise.losses import auxiliary_loss_weight, normalized_auxiliary_loss
except ModuleNotFoundError:
    torch = None
    nn = None


class DummyTokenizer:
    padding_side = "left"

    def __call__(self, texts, **kwargs):
        lengths = [max(1, min(3, len(text))) for text in texts]
        width = max(lengths)
        input_ids = torch.zeros((len(texts), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, length in enumerate(lengths):
            input_ids[index, :length] = torch.arange(1, length + 1)
            attention_mask[index, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


if nn is not None:
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=6)
            self.embedding = nn.Embedding(8, 6)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def features(first=False):
    return {name: values[min(1, len(values) - 1)] if first else values[0] for name, values in FEATURE_VALUES.items()}


class PairwiseAuxiliaryTests(unittest.TestCase):
    def test_production_v1_v2_configs_differ_only_in_auxiliary_objective(self):
        v1 = json.loads((ROOT / "configs" / "v3_bt_production_v1.json").read_text(encoding="utf-8"))
        v2 = json.loads((ROOT / "configs" / "v3_bt_production_v2_aux10.json").read_text(encoding="utf-8"))
        self.assertFalse(v1["auxiliary_features"])
        self.assertEqual(v1["auxiliary_loss_weight"], 0.0)
        self.assertTrue(v2["auxiliary_features"])
        self.assertEqual(v2["auxiliary_loss_weight"], 0.1)
        ignored = {"auxiliary_features", "auxiliary_loss_weight"}
        self.assertEqual(
            {key: value for key, value in v1.items() if key not in ignored},
            {key: value for key, value in v2.items() if key not in ignored},
        )
        self.assertEqual(v2["checkpoint_every_epochs"], 0.25)
        self.assertEqual(v2["num_train_epochs"], 3)

    def test_join_uses_only_id_features_and_quality_not_absolute_difficulty(self):
        pair = {
            "pair_id": "p1", "question_a_id": "qa", "question_b_id": "qb",
            "question_a_text": "题 A", "question_b_text": "题 B",
            "soft_target": 0.75, "sample_weight": 0.8,
        }
        teachers = [
            {"id": "qa", "difficulty": 5, "teacher_difficulty_id": 4, "teacher_features": features(), "label_quality": {"sample_weight": 0.6}},
            {"id": "qb", "difficulty": 1, "teacher_difficulty_level": "送分题", "teacher_features": features(True), "label_quality": {"sample_weight": 0.9}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pairs = directory / "pairs.jsonl"
            teacher = directory / "teacher.jsonl"
            output = directory / "joined.jsonl"
            manifest = directory / "manifest.json"
            pairs.write_text(json.dumps(pair, ensure_ascii=False) + "\n", encoding="utf-8")
            teacher.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in teachers), encoding="utf-8")
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "attach_pairwise_auxiliary_features.py"),
                "--pairs", str(pairs), "--features", str(teacher), "--output", str(output),
                "--manifest", str(manifest), "--minimum-question-coverage", "1.0",
            ], check=True, capture_output=True, text=True)
            joined = json.loads(output.read_text(encoding="utf-8"))
            serialized = json.dumps(joined, ensure_ascii=False)
            self.assertNotIn("difficulty", serialized)
            self.assertEqual(joined["auxiliary_features"]["question_a"], features())
            self.assertEqual(joined["auxiliary_feature_quality"]["question_b"], 0.9)
            report = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(report["question_coverage"], 1.0)

    @unittest.skipUnless(torch is not None, "PyTorch is not installed in this local test runtime")
    def test_dataset_collates_ten_targets_and_degree_corrected_weights(self):
        rows = []
        for index, other in enumerate(("qb", "qc")):
            rows.append({
                "pair_id": f"p{index}", "question_a_id": "qa", "question_b_id": other,
                "question_a_text": "题 A", "question_b_text": f"题 {other}",
                "soft_target": 0.5, "sample_weight": 1.0,
                "auxiliary_features": {"question_a": features(), "question_b": features(True)},
                "auxiliary_feature_quality": {"question_a": 0.8, "question_b": 1.0},
            })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            dataset = PairwiseDifficultyDataset(str(path), DummyTokenizer(), 32, require_auxiliary_features=True)
            batch = dataset.collate_fn(rows)
            self.assertEqual(set(batch["auxiliary_targets_a"]), set(FEATURE_TO_ID))
            self.assertEqual(batch["auxiliary_targets_a"]["problem_structure"].tolist(), [0, 0])
            self.assertEqual(batch["auxiliary_targets_b"]["problem_structure"].tolist(), [1, 1])
            self.assertTrue(torch.allclose(batch["auxiliary_weights_a"], torch.tensor([0.4, 0.4])))
            self.assertTrue(torch.allclose(batch["auxiliary_weights_b"], torch.tensor([1.0, 1.0])))

    @unittest.skipUnless(torch is not None, "PyTorch is not installed in this local test runtime")
    def test_model_returns_scalar_and_all_auxiliary_logits(self):
        model = QwenPairwiseRater(DummyBackbone(), dropout=0.0, auxiliary_features=True)
        ids = torch.tensor([[1, 2], [2, 0], [3, 4], [4, 0]])
        mask = torch.tensor([[1, 1], [1, 0], [1, 1], [1, 0]])
        output = model(ids, mask, pair_count=2)
        self.assertEqual(output["pair_logits"].shape, (2,))
        self.assertEqual(set(output["auxiliary_logits_a"]), set(FEATURE_VALUES))
        for name, values in FEATURE_VALUES.items():
            self.assertEqual(output["auxiliary_logits_a"][name].shape, (2, len(values)))

    @unittest.skipUnless(torch is not None, "PyTorch is not installed in this local test runtime")
    def test_auxiliary_loss_is_class_count_normalized_and_warmed_up(self):
        logits_a = {name: torch.zeros((2, len(values))) for name, values in FEATURE_VALUES.items()}
        logits_b = {name: torch.zeros((2, len(values))) for name, values in FEATURE_VALUES.items()}
        targets = {name: torch.zeros(2, dtype=torch.long) for name in FEATURE_VALUES}
        class_weights = {name: torch.ones(len(values)) for name, values in FEATURE_VALUES.items()}
        loss = normalized_auxiliary_loss(
            logits_a, logits_b, targets, targets, torch.ones(2), torch.ones(2), class_weights,
        )
        self.assertAlmostEqual(float(loss), 1.0, places=5)
        self.assertEqual(auxiliary_loss_weight(0, 100, 0.1, 0.1), 0.0)
        self.assertAlmostEqual(auxiliary_loss_weight(5, 100, 0.1, 0.1), 0.05)
        self.assertAlmostEqual(auxiliary_loss_weight(10, 100, 0.1, 0.1), 0.1)


if __name__ == "__main__":
    unittest.main()
