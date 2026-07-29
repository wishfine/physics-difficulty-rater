import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import torch

    from experiment_vllm_pairwise_parity import (
        extract_vllm_data,
        ranking_agreement,
    )
    from physics_difficulty.models.external_pairwise_head import ExternalPairwiseHead
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class VllmPairwiseParityTests(unittest.TestCase):
    def test_external_head_loads_scalar_and_auxiliary_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            original = ExternalPairwiseHead(hidden_size=6, auxiliary_features=True)
            torch.save(
                {
                    "norm": original.norm.state_dict(),
                    "score_head": original.score_head.state_dict(),
                    "auxiliary_heads": original.auxiliary_heads.state_dict(),
                },
                checkpoint / "pairwise_head.pt",
            )
            (checkpoint / "pairwise_config.json").write_text(
                json.dumps({"auxiliary_features": True}),
                encoding="utf-8",
            )
            loaded = ExternalPairwiseHead.from_checkpoint(checkpoint)
            output = loaded(torch.ones((2, 6)))
            self.assertEqual(output["scores"].shape, (2,))
            self.assertEqual(set(output["auxiliary_logits"]), set(original.auxiliary_heads))

    def test_external_head_rejects_tokenwise_hidden_states(self):
        head = ExternalPairwiseHead(hidden_size=6, auxiliary_features=False)
        with self.assertRaisesRegex(ValueError, "batch, hidden_size"):
            head(torch.ones((2, 3, 6)))

    def test_ranking_agreement_uses_all_non_tied_pairs(self):
        agreement, count = ranking_agreement(
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([0.0, 1.1, 1.9]),
        )
        self.assertEqual(count, 3)
        self.assertEqual(agreement, 1.0)

    def test_extract_vllm_data_supports_current_pooling_output(self):
        class Inner:
            data = [1.0, 2.0]

        class Outer:
            outputs = Inner()

        self.assertEqual(extract_vllm_data(Outer()), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
