#!/usr/bin/env python3
"""Continuously evaluate initial and quarter-epoch pairwise checkpoints."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_training_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "training_config.json"
    if not path.is_file():
        raise ValueError(f"missing training config: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config.get("num_train_epochs", 0)) <= 0:
        raise ValueError("training config has invalid num_train_epochs")
    return config


def checkpoint_complete(path: Path) -> bool:
    return (
        (path / "adapter").is_dir()
        and (path / "pairwise_head.pt").is_file()
        and (path / "pairwise_config.json").is_file()
        and (path / "trainer_state.json").is_file()
    )


def discover_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint-epoch-*"):
        if not path.is_dir() or not checkpoint_complete(path):
            continue
        state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        checkpoints.append((int(state["optimizer_step"]), path))
    checkpoints.sort(key=lambda item: (item[0], item[1].name))
    return checkpoints


def result_name(step: int, checkpoint: Path) -> str:
    return f"step-{step:06d}_{checkpoint.name}.json"


def run(command: list[str]) -> None:
    print(json.dumps({"command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def wait_for_file(path: Path, poll_seconds: int) -> None:
    while not path.is_file() or path.stat().st_size == 0:
        print(json.dumps({"message": "waiting for file", "path": str(path)}), flush=True)
        time.sleep(poll_seconds)


def prepare_evaluation_file(
    validation_pairs: Path,
    output_dir: Path,
    auxiliary_features: bool,
    features_file: Path | None,
) -> Path:
    if not auxiliary_features:
        return validation_pairs
    if features_file is None:
        raise ValueError("--features-file is required for an auxiliary V2 run")
    output = output_dir / "validation_aux10.jsonl"
    manifest = output_dir / "validation_aux10.manifest.json"
    if not output.is_file() or not manifest.is_file():
        run([
            sys.executable,
            str(ROOT / "scripts" / "attach_pairwise_auxiliary_features.py"),
            "--pairs", str(validation_pairs),
            "--features", str(features_file),
            "--output", str(output),
            "--manifest", str(manifest),
            "--minimum-question-coverage", "0.95",
        ])
    return output


def ensure_initial_checkpoint(
    model_path: Path,
    run_dir: Path,
    auxiliary_features: bool,
    config: dict[str, Any],
) -> Path:
    checkpoint = run_dir / "checkpoint-initial"
    complete = (
        (checkpoint / "adapter").is_dir()
        and (checkpoint / "pairwise_head.pt").is_file()
        and (checkpoint / "pairwise_config.json").is_file()
        and (checkpoint / "initial_state.json").is_file()
    )
    if complete:
        return checkpoint
    command = [
        sys.executable,
        str(ROOT / "scripts" / "create_initial_pairwise_checkpoint.py"),
        "--model-path", str(model_path),
        "--output-dir", str(checkpoint),
        "--lora-r", str(config.get("lora_r", 8)),
        "--lora-alpha", str(config.get("lora_alpha", 16)),
        "--lora-dropout", str(config.get("lora_dropout", 0.05)),
        "--seed", str(config.get("seed", 42)),
    ]
    if auxiliary_features:
        command.append("--auxiliary-features")
    run(command)
    return checkpoint


def evaluate_checkpoint(
    model_path: Path,
    checkpoint: Path,
    eval_file: Path,
    output_dir: Path,
    step: int,
    max_length: int,
    batch_size: int,
) -> Path:
    metrics = output_dir / result_name(step, checkpoint)
    if metrics.is_file():
        print(json.dumps({"message": "skipping existing result", "path": str(metrics)}), flush=True)
        return metrics
    predictions = output_dir / metrics.name.replace(".json", "_predictions.jsonl")
    run([
        sys.executable,
        str(ROOT / "evaluate_pairwise.py"),
        "--model-path", str(model_path),
        "--checkpoint-dir", str(checkpoint),
        "--eval-file", str(eval_file),
        "--max-length", str(max_length),
        "--batch-size", str(batch_size),
        "--output-file", str(metrics),
        "--predictions-file", str(predictions),
    ])
    return metrics


def write_series_manifest(
    output_dir: Path,
    run_dir: Path,
    eval_file: Path,
    results: list[tuple[int, Path, Path]],
) -> None:
    records = []
    for step, checkpoint, result in sorted(results):
        metrics = json.loads(result.read_text(encoding="utf-8"))
        records.append({
            "optimizer_step": step,
            "checkpoint": str(checkpoint.resolve()),
            "result": str(result.resolve()),
            "pairwise": metrics["pairwise"],
        })
    payload = {
        "schema_version": "pairwise_checkpoint_evaluation_series_v1",
        "run_dir": str(run_dir.resolve()),
        "eval_file": str(eval_file.resolve()),
        "evaluated_checkpoints": len(records),
        "results": records,
    }
    (output_dir / "series_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--validation-pairs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--features-file")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.poll_seconds <= 0:
        raise ValueError("batch-size and poll-seconds must be positive")

    model_path = Path(args.model_path)
    run_dir = Path(args.run_dir)
    validation_pairs = Path(args.validation_pairs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_training_config(run_dir)
    auxiliary_features = bool(config.get("auxiliary_features", False))
    target_epoch = int(config["num_train_epochs"])

    wait_for_file(validation_pairs, args.poll_seconds)
    eval_file = prepare_evaluation_file(
        validation_pairs,
        output_dir,
        auxiliary_features,
        Path(args.features_file) if args.features_file else None,
    )
    initial = ensure_initial_checkpoint(model_path, run_dir, auxiliary_features, config)
    evaluated: dict[str, tuple[int, Path, Path]] = {}
    initial_result = evaluate_checkpoint(
        model_path, initial, eval_file, output_dir, 0,
        int(config.get("max_length", 1024)), args.batch_size,
    )
    evaluated[str(initial.resolve())] = (0, initial, initial_result)

    final_checkpoint = run_dir / f"checkpoint-epoch-{target_epoch}"
    while True:
        for step, checkpoint in discover_checkpoints(run_dir):
            key = str(checkpoint.resolve())
            result = evaluate_checkpoint(
                model_path, checkpoint, eval_file, output_dir, step,
                int(config.get("max_length", 1024)), args.batch_size,
            )
            evaluated[key] = (step, checkpoint, result)
            write_series_manifest(output_dir, run_dir, eval_file, list(evaluated.values()))
        final_key = str(final_checkpoint.resolve())
        if final_key in evaluated:
            print(json.dumps({
                "message": "evaluation series complete",
                "checkpoints": len(evaluated),
                "manifest": str((output_dir / "series_manifest.json").resolve()),
            }, ensure_ascii=False), flush=True)
            break
        print(json.dumps({
            "message": "waiting for more checkpoints",
            "target": str(final_checkpoint),
            "evaluated": len(evaluated),
        }, ensure_ascii=False), flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
