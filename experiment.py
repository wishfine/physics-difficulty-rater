#!/usr/bin/env python3
"""Reproducible entry point for a named V2 training run."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default="configs/v2_teacher_train.json")
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    command = [
        "torchrun", f"--nproc_per_node={args.gpus}", "train_difficulty.py",
        "--config", args.config, "--model_path", args.model_path,
        "--train_file", args.train_file, "--output_dir", args.output_dir,
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=Path(__file__).resolve().parent, check=True)


if __name__ == "__main__":
    main()
