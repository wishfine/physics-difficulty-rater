#!/usr/bin/env python3
"""Score text-only questions with the learned scalar difficulty function."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.data.text_only import forbidden_source_label_paths, leakage_findings, question_identifier
from physics_difficulty.models.pairwise_loading import load_pairwise_rater
from physics_difficulty.pairwise.calibration import apply_calibration, validate_calibration


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint(checkpoint: Path, model_path: Path) -> str:
    paths = [checkpoint / "pairwise_head.pt", checkpoint / "pairwise_config.json"]
    paths.extend(path for path in (checkpoint / "adapter").rglob("*") if path.is_file())
    if (checkpoint / "tokenizer").is_dir():
        paths.extend(path for path in (checkpoint / "tokenizer").rglob("*") if path.is_file())
    if (model_path / "config.json").is_file():
        paths.append(model_path / "config.json")
    missing = [str(path) for path in paths[:2] if not path.is_file()]
    if missing:
        raise ValueError(f"checkpoint fingerprint inputs are missing: {missing}")
    digest = hashlib.sha256()
    digest.update(str(model_path.resolve()).encode("utf-8"))
    digest.update(b"\0")
    for path in sorted(paths, key=lambda item: str(item.resolve())):
        try:
            identity = f"checkpoint/{path.relative_to(checkpoint)}"
        except ValueError:
            identity = f"base_model/{path.relative_to(model_path)}"
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_excluded_ids(paths: list[str]) -> set[str]:
    excluded: set[str] = set()
    for value in paths:
        path = Path(value)
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                question_id = str(row.get("id") or row.get("question_id") or "").strip()
                if not question_id:
                    raise ValueError(f"{path}: line {line_number} lacks id/question_id")
                excluded.add(question_id)
    return excluded


class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, excluded_ids: set[str] | None = None):
        excluded_ids = excluded_ids or set()
        source_rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.source_records = len(source_rows)
        self.rows = []
        self.excluded_records = 0
        seen_ids: set[str] = set()
        for row in source_rows:
            question_id = question_identifier(row)
            if question_id in seen_ids:
                raise ValueError(f"question file contains duplicate id {question_id}")
            seen_ids.add(question_id)
            if question_id in excluded_ids:
                self.excluded_records += 1
                continue
            self.rows.append(row)
        if not self.rows:
            raise ValueError("question file is empty after exclusions")
        for row in self.rows:
            if forbidden_source_label_paths(row):
                raise ValueError(f"question {row.get('id')} contains forbidden historical difficulty")
            if not str(row.get("text") or "").strip():
                raise ValueError(f"question {row.get('id')} has empty text")
            if leakage_findings(str(row["text"])):
                raise ValueError(f"question {row.get('id')} contains explicit difficulty leakage")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument(
        "--exclude-question-ids",
        action="append",
        default=[],
        help="Question JSONL whose id/question_id values must be excluded; repeatable",
    )
    parser.add_argument("--calibration")
    parser.add_argument(
        "--include-auxiliary",
        action="store_true",
        help="Export auxiliary predictions from a checkpoint that contains auxiliary heads.",
    )
    parser.add_argument(
        "--include-auxiliary-probabilities",
        action="store_true",
        help="Store every auxiliary class probability in addition to the winning label.",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.max_length <= 0 or args.batch_size <= 0:
        raise ValueError("max-length and batch-size must be positive")

    checkpoint = Path(args.checkpoint_dir)
    model_path = Path(args.model_path)
    fingerprint = checkpoint_fingerprint(checkpoint, model_path)
    calibration = None
    if args.calibration:
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        validate_calibration(calibration)
        if calibration.get("checkpoint_fingerprint") != fingerprint:
            raise ValueError("calibration was produced by a different checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_pairwise_rater(args.model_path, args.checkpoint_dir, device, args.bf16)
    if args.include_auxiliary and not model.auxiliary_features:
        raise ValueError("--include-auxiliary requires a checkpoint with auxiliary heads")
    if args.include_auxiliary_probabilities and not args.include_auxiliary:
        raise ValueError("--include-auxiliary-probabilities requires --include-auxiliary")
    excluded_ids = load_excluded_ids(args.exclude_question_ids)
    dataset = QuestionDataset(args.questions, excluded_ids)

    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer([str(row["text"]) for row in rows], truncation=True, max_length=args.max_length, padding=True, return_tensors="pt")
        return {"rows": rows, **encoded}

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    output = Path(args.output)
    manifest = Path(args.manifest) if args.manifest else output.with_name(
        f"{output.name[:-6] if output.name.endswith('.jsonl') else output.name}.manifest.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    score_values: list[float] = []
    split_counts: Counter[str] = Counter()
    auxiliary_counts = {
        name: Counter() for name in model.feature_values
    } if args.include_auxiliary else {}
    auxiliary_confidences = {
        name: [] for name in model.feature_values
    } if args.include_auxiliary else {}
    auxiliary_normalized_entropies = {
        name: [] for name in model.feature_values
    } if args.include_auxiliary else {}
    print(json.dumps({
        "message": "Scoring individual questions; raw scores establish global ordering but are not absolute difficulty levels.",
        "records": len(dataset),
        "excluded_records": dataset.excluded_records,
    }, ensure_ascii=False), flush=True)
    with temporary.open("w", encoding="utf-8") as target, torch.no_grad():
        for batch in loader:
            representation = model.encode(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            scores = model.score_head(representation).squeeze(-1).float().cpu().tolist()
            auxiliary_probabilities: dict[str, torch.Tensor] = {}
            if args.include_auxiliary:
                auxiliary_probabilities = {
                    name: torch.softmax(head(representation).float(), dim=-1).cpu()
                    for name, head in model.auxiliary_heads.items()
                }
            for row_index, (row, score) in enumerate(zip(batch["rows"], scores)):
                if not math.isfinite(float(score)):
                    raise ValueError(f"question {row.get('id')} produced a non-finite score")
                question_id = question_identifier(row)
                split = str(row.get("split") or "unspecified")
                score_values.append(float(score))
                split_counts[split] += 1
                result = {
                    "question_id": question_id,
                    "split": row.get("split"),
                    "text_sha256": row.get("text_sha256")
                    or hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest(),
                    "raw_difficulty_score": float(score),
                }
                if args.include_auxiliary:
                    result["auxiliary_predictions"] = {}
                    for name, values in model.feature_values.items():
                        probabilities = auxiliary_probabilities[name][row_index]
                        label_id = int(probabilities.argmax().item())
                        confidence = float(probabilities[label_id].item())
                        entropy = float(
                            -(probabilities * probabilities.clamp_min(1e-12).log()).sum().item()
                        )
                        normalized_entropy = entropy / math.log(len(values))
                        prediction = {
                            "label": values[label_id],
                            "label_id": label_id,
                            "confidence": confidence,
                            "normalized_entropy": normalized_entropy,
                        }
                        if args.include_auxiliary_probabilities:
                            prediction["probabilities"] = {
                                label: float(probability)
                                for label, probability in zip(values, probabilities.tolist())
                            }
                        result["auxiliary_predictions"][name] = prediction
                        auxiliary_counts[name][values[label_id]] += 1
                        auxiliary_confidences[name].append(confidence)
                        auxiliary_normalized_entropies[name].append(normalized_entropy)
                if calibration:
                    result.update(
                        apply_calibration(
                            float(score),
                            calibration,
                            calibration_already_validated=True,
                        )
                    )
                target.write(json.dumps(result, ensure_ascii=False) + "\n")
    temporary.replace(output)
    report = {
        "schema_version": (
            "pairwise_single_question_scores_with_auxiliary_v2"
            if args.include_auxiliary
            else "pairwise_single_question_scores_v1"
        ),
        "records": len(dataset),
        "source_records": dataset.source_records,
        "excluded_question_count": dataset.excluded_records,
        "questions": str(Path(args.questions).resolve()),
        "questions_sha256": sha256_file(Path(args.questions)),
        "excluded_question_files": [str(Path(path).resolve()) for path in args.exclude_question_ids],
        "checkpoint_dir": str(checkpoint.resolve()),
        "checkpoint_fingerprint": fingerprint,
        "model_path": str(Path(args.model_path).resolve()),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "bf16": args.bf16,
        "splits": dict(split_counts),
        "score_stats": {
            "minimum": min(score_values),
            "maximum": max(score_values),
            "mean": statistics.fmean(score_values),
            "population_std": statistics.pstdev(score_values),
        },
        "calibrated": calibration is not None,
        "calibration_id": calibration.get("calibration_id") if calibration else None,
        "auxiliary_exported": args.include_auxiliary,
        "auxiliary_probabilities_exported": args.include_auxiliary_probabilities,
        "auxiliary_feature_values": model.feature_values if args.include_auxiliary else None,
        "auxiliary_summary": (
            {
                name: {
                    "predicted_class_counts": dict(auxiliary_counts[name]),
                    "mean_confidence": statistics.fmean(auxiliary_confidences[name]),
                    "mean_normalized_entropy": statistics.fmean(
                        auxiliary_normalized_entropies[name]
                    ),
                }
                for name in model.feature_values
            }
            if args.include_auxiliary
            else None
        ),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
