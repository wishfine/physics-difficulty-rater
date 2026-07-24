import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_pairwise_checkpoint_series.py"


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_pairwise_checkpoint_series", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PairwiseEvaluationSeriesTests(unittest.TestCase):
    def test_discovers_only_complete_checkpoints_in_optimizer_step_order(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            for name, step in (
                ("checkpoint-epoch-1", 470),
                ("checkpoint-epoch-1-step-118", 118),
                ("checkpoint-epoch-1-step-354", 354),
            ):
                checkpoint = run / name
                (checkpoint / "adapter").mkdir(parents=True)
                (checkpoint / "pairwise_head.pt").write_text("head")
                (checkpoint / "pairwise_config.json").write_text("{}")
                (checkpoint / "trainer_state.json").write_text(json.dumps({"optimizer_step": step}))
            incomplete = run / "checkpoint-epoch-1-step-236"
            incomplete.mkdir()
            (incomplete / "trainer_state.json").write_text(json.dumps({"optimizer_step": 236}))

            discovered = module.discover_checkpoints(run)
            self.assertEqual(
                [(step, path.name) for step, path in discovered],
                [
                    (118, "checkpoint-epoch-1-step-118"),
                    (354, "checkpoint-epoch-1-step-354"),
                    (470, "checkpoint-epoch-1"),
                ],
            )

    def test_result_name_is_stable_and_step_prefixed(self):
        module = load_module()
        self.assertEqual(
            module.result_name(118, Path("checkpoint-epoch-1-step-118")),
            "step-000118_checkpoint-epoch-1-step-118.json",
        )
        self.assertEqual(
            module.result_name(0, Path("checkpoint-initial")),
            "step-000000_checkpoint-initial.json",
        )

    def test_reads_auxiliary_mode_and_target_epoch_from_training_config(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "training_config.json").write_text(json.dumps({
                "auxiliary_features": True,
                "num_train_epochs": 3,
                "max_length": 1024,
            }))
            config = module.load_training_config(run)
            self.assertTrue(config["auxiliary_features"])
            self.assertEqual(config["num_train_epochs"], 3)


if __name__ == "__main__":
    unittest.main()
