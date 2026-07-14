#!/usr/bin/env python3
"""Plot training loss from the metrics JSONL emitted by train_difficulty.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.metrics).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("metrics file contains no records")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    plt.plot([row["epoch"] for row in rows], [row["mean_loss"] for row in rows], marker="o")
    plt.xlabel("epoch")
    plt.ylabel("mean training loss")
    plt.grid(alpha=0.25)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
