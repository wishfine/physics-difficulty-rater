#!/usr/bin/env python3
"""Compare HF and vLLM LAST pooling with the trained external pairwise heads."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.data.text_only import question_identifier
from physics_difficulty.models.external_pairwise_head import ExternalPairwiseHead
from physics_difficulty.models.pairwise_loading import load_pairwise_rater
from physics_difficulty.schema import FEATURE_VALUES


def load_questions(path: str | Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = question_identifier(row)
            if question_id in seen:
                continue
            text = str(row.get("text") or "").strip()
            if not text:
                raise ValueError(f"{path}: line {line_number} has empty text")
            seen.add(question_id)
            rows.append({"question_id": question_id, "text": text})
            if len(rows) >= limit:
                break
    if not rows:
        raise ValueError("questions file produced no usable records")
    return rows


def tokenize_questions(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_length: int,
) -> list[list[int]]:
    token_ids: list[list[int]] = []
    for row in rows:
        encoded = tokenizer(
            row["text"],
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        ids = [int(value) for value in encoded["input_ids"]]
        if not ids:
            raise ValueError(f"question {row['question_id']} tokenized to an empty sequence")
        token_ids.append(ids)
    return token_ids


def padded_batch(
    sequences: list[list[int]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences))
    input_ids = torch.full(
        (len(sequences), width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[index, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[index, :length] = 1
    return input_ids, attention_mask


def extract_vllm_data(output: Any) -> list[float]:
    """Handle current and older PoolingRequestOutput field names."""
    value = getattr(output, "outputs", output)
    for field in ("data", "embedding"):
        data = getattr(value, field, None)
        if data is not None:
            return [float(item) for item in data]
    if isinstance(value, dict):
        for field in ("data", "embedding"):
            if field in value:
                return [float(item) for item in value[field]]
    raise TypeError(f"unsupported vLLM pooling output: {type(output)!r}")


def pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.float().flatten()
    right = right.float().flatten()
    if left.numel() < 2 or float(left.std(unbiased=False)) == 0.0 or float(right.std(unbiased=False)) == 0.0:
        return None
    return float(torch.corrcoef(torch.stack((left, right)))[0, 1])


def ranking_agreement(left: torch.Tensor, right: torch.Tensor) -> tuple[float, int]:
    agreements = 0
    comparisons = 0
    for first in range(left.numel()):
        for second in range(first + 1, left.numel()):
            left_delta = float(left[first] - left[second])
            right_delta = float(right[first] - right[second])
            if left_delta == 0.0 or right_delta == 0.0:
                continue
            agreements += int((left_delta > 0.0) == (right_delta > 0.0))
            comparisons += 1
    return (agreements / comparisons if comparisons else float("nan"), comparisons)


def tensor_difference(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    difference = (reference.float() - candidate.float()).abs()
    return {
        "mean_absolute_error": float(difference.mean()),
        "maximum_absolute_error": float(difference.max()),
    }


def auxiliary_predictions(logits: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    return {
        name: values.float().argmax(dim=-1).cpu().tolist()
        for name, values in logits.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--hf-batch-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.limit <= 1 or args.max_length <= 0 or args.hf_batch_size <= 0:
        raise ValueError("limit must exceed one and lengths/batch sizes must be positive")

    checkpoint = Path(args.checkpoint_dir)
    adapter_path = checkpoint / "adapter"
    if not adapter_path.is_dir():
        raise ValueError(f"checkpoint is missing {adapter_path}")
    adapter_config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    lora_rank = int(adapter_config.get("r", 0))
    if lora_rank <= 0:
        raise ValueError("adapter_config.json has no positive LoRA rank")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_questions(args.questions, args.limit)
    print(
        json.dumps(
            {
                "message": "HF reference: verify current checkpoint and export raw LAST-token states.",
                "records": len(rows),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    device = torch.device("cuda")
    hf_load_start = time.perf_counter()
    model, tokenizer = load_pairwise_rater(args.model_path, checkpoint, device, bf16=True)
    hf_model_load_seconds = time.perf_counter() - hf_load_start
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer has no pad_token_id")
    token_ids = tokenize_questions(tokenizer, rows, args.max_length)
    hf_raw_parts: list[torch.Tensor] = []
    hf_representation_parts: list[torch.Tensor] = []
    hf_score_parts: list[torch.Tensor] = []
    hf_aux_parts: dict[str, list[torch.Tensor]] = {
        name: [] for name in FEATURE_VALUES
    } if model.auxiliary_features else {}
    hf_inference_start = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(rows), args.hf_batch_size):
            batch_ids, mask = padded_batch(
                token_ids[start : start + args.hf_batch_size],
                pad_token_id,
                device,
            )
            output = model.backbone(input_ids=batch_ids, attention_mask=mask)
            raw = model._pool(output.last_hidden_state, mask)
            representation = model.norm(raw)
            scores = model.score_head(representation).squeeze(-1)
            hf_raw_parts.append(raw.float().cpu())
            hf_representation_parts.append(representation.float().cpu())
            hf_score_parts.append(scores.float().cpu())
            for name, head in model.auxiliary_heads.items():
                hf_aux_parts[name].append(head(representation).float().cpu())
    hf_raw = torch.cat(hf_raw_parts)
    hf_representation = torch.cat(hf_representation_parts)
    hf_scores = torch.cat(hf_score_parts)
    hf_aux = {name: torch.cat(parts) for name, parts in hf_aux_parts.items()}
    hf_inference_seconds = time.perf_counter() - hf_inference_start

    external_head = ExternalPairwiseHead.from_checkpoint(checkpoint)
    with torch.no_grad():
        hf_external = external_head(hf_raw)
    external_contract = tensor_difference(hf_scores, hf_external["scores"])

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "message": "vLLM experiment: compare base and LoRA LAST pooling, then apply the external heads.",
                "lora_rank": lora_rank,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    import vllm
    from vllm import LLM
    from vllm.config import PoolerConfig
    from vllm.lora.request import LoRARequest

    pooler_config = PoolerConfig(pooling_type="LAST", use_activation=False)
    engine_start = time.perf_counter()
    llm = LLM(
        model=args.model_path,
        tokenizer=str(checkpoint / "tokenizer") if (checkpoint / "tokenizer").is_dir() else args.model_path,
        runner="pooling",
        convert="embed",
        pooler_config=pooler_config,
        language_model_only=True,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_length,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=lora_rank,
        enforce_eager=args.enforce_eager,
    )
    engine_load_seconds = time.perf_counter() - engine_start
    prompts = [{"prompt_token_ids": ids} for ids in token_ids]
    lora_request = LoRARequest("physics_difficulty", 1, str(adapter_path))

    warmup_start = time.perf_counter()
    llm.encode(prompts[: min(2, len(prompts))], pooling_task="embed")
    llm.encode(
        prompts[: min(2, len(prompts))],
        pooling_task="embed",
        lora_request=lora_request,
    )
    warmup_seconds = time.perf_counter() - warmup_start
    base_start = time.perf_counter()
    base_outputs = llm.encode(prompts, pooling_task="embed")
    base_seconds = time.perf_counter() - base_start
    base_raw = torch.tensor(
        [extract_vllm_data(output) for output in base_outputs],
        dtype=torch.float32,
    )

    lora_start = time.perf_counter()
    lora_outputs = llm.encode(
        prompts,
        pooling_task="embed",
        lora_request=lora_request,
    )
    lora_seconds = time.perf_counter() - lora_start
    vllm_raw = torch.tensor(
        [extract_vllm_data(output) for output in lora_outputs],
        dtype=torch.float32,
    )
    if vllm_raw.shape != hf_raw.shape:
        raise ValueError(
            f"vLLM representation shape {tuple(vllm_raw.shape)} != HF {tuple(hf_raw.shape)}"
        )
    with torch.no_grad():
        vllm_head = external_head(vllm_raw)

    vllm_scores = vllm_head["scores"].float()
    raw_cosine = F.cosine_similarity(hf_raw.float(), vllm_raw.float(), dim=-1)
    representation_cosine = F.cosine_similarity(
        hf_representation.float(),
        vllm_head["representation"].float(),
        dim=-1,
    )
    score_difference = tensor_difference(hf_scores, vllm_scores)
    agreement, comparison_count = ranking_agreement(hf_scores, vllm_scores)
    lora_delta = torch.linalg.vector_norm(vllm_raw - base_raw, dim=-1)
    lora_changed = int((lora_delta > 1e-6).sum())

    auxiliary_report: dict[str, Any] = {}
    hf_aux_predictions = auxiliary_predictions(hf_aux) if hf_aux else {}
    vllm_aux = vllm_head.get("auxiliary_logits", {})
    vllm_aux_predictions = auxiliary_predictions(vllm_aux) if vllm_aux else {}
    for name in hf_aux:
        matches = [
            int(left == right)
            for left, right in zip(
                hf_aux_predictions[name],
                vllm_aux_predictions[name],
            )
        ]
        auxiliary_report[name] = {
            **tensor_difference(hf_aux[name], vllm_aux[name]),
            "argmax_agreement": statistics.fmean(matches),
        }

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as target:
        for index, row in enumerate(rows):
            record: dict[str, Any] = {
                "question_id": row["question_id"],
                "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
                "token_count": len(token_ids[index]),
                "hf_score": float(hf_scores[index]),
                "vllm_score": float(vllm_scores[index]),
                "absolute_score_difference": abs(
                    float(hf_scores[index]) - float(vllm_scores[index])
                ),
                "vllm_lora_vs_base_embedding_l2": float(lora_delta[index]),
            }
            if hf_aux:
                record["hf_auxiliary_predictions"] = {
                    name: FEATURE_VALUES[name][hf_aux_predictions[name][index]]
                    for name in FEATURE_VALUES
                }
                record["vllm_auxiliary_predictions"] = {
                    name: FEATURE_VALUES[name][vllm_aux_predictions[name][index]]
                    for name in FEATURE_VALUES
                }
            target.write(json.dumps(record, ensure_ascii=False) + "\n")

    score_correlation = pearson(hf_scores, vllm_scores)
    finite_metrics = all(
        math.isfinite(value)
        for value in (
            score_difference["mean_absolute_error"],
            score_difference["maximum_absolute_error"],
            float(raw_cosine.mean()),
            float(representation_cosine.mean()),
            agreement,
        )
    )
    acceptance_gate = {
        "external_head_maximum_absolute_error_lte_1e-5": (
            external_contract["maximum_absolute_error"] <= 1e-5
        ),
        "lora_changed_every_representation": lora_changed == len(rows),
        "raw_hidden_cosine_mean_gte_0_999": float(raw_cosine.mean()) >= 0.999,
        "score_mean_absolute_error_lte_0_05": (
            score_difference["mean_absolute_error"] <= 0.05
        ),
        "score_pearson_gte_0_99": (
            score_correlation is not None and score_correlation >= 0.99
        ),
        "ranking_pairwise_agreement_gte_0_98": agreement >= 0.98,
    }
    report = {
        "schema_version": "vllm_pairwise_parity_v1",
        "status": (
            "PASS"
            if finite_metrics and all(acceptance_gate.values())
            else "REVIEW"
        ),
        "records": len(rows),
        "model_path": str(Path(args.model_path).resolve()),
        "checkpoint_dir": str(checkpoint.resolve()),
        "questions": str(Path(args.questions).resolve()),
        "versions": {
            "torch": torch.__version__,
            "vllm": getattr(vllm, "__version__", "unknown"),
        },
        "configuration": {
            "pooling_type": "LAST",
            "pooler_activation": False,
            "language_model_only": True,
            "lora_rank": lora_rank,
            "max_length": args.max_length,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
        },
        "classification_head_contract": {
            **external_contract,
            "status": (
                "PASS"
                if external_contract["maximum_absolute_error"] <= 1e-5
                else "FAIL"
            ),
        },
        "vllm_lora_application": {
            "changed_representation_count": lora_changed,
            "unchanged_representation_count": len(rows) - lora_changed,
            "mean_embedding_l2_delta": float(lora_delta.mean()),
            "minimum_embedding_l2_delta": float(lora_delta.min()),
            "status": "PASS" if lora_changed == len(rows) else "FAIL",
        },
        "hf_vllm_parity": {
            "raw_hidden_cosine_mean": float(raw_cosine.mean()),
            "raw_hidden_cosine_minimum": float(raw_cosine.min()),
            "normalized_representation_cosine_mean": float(representation_cosine.mean()),
            "normalized_representation_cosine_minimum": float(representation_cosine.min()),
            "score_mean_absolute_error": score_difference["mean_absolute_error"],
            "score_maximum_absolute_error": score_difference["maximum_absolute_error"],
            "score_pearson": score_correlation,
            "ranking_pairwise_agreement": agreement,
            "ranking_comparison_count": comparison_count,
        },
        "auxiliary_parity": auxiliary_report,
        "acceptance_gate": acceptance_gate,
        "timing_seconds": {
            "hf_model_load": hf_model_load_seconds,
            "hf_inference": hf_inference_seconds,
            "vllm_engine_load": engine_load_seconds,
            "vllm_warmup": warmup_seconds,
            "vllm_base_inference": base_seconds,
            "vllm_lora_inference": lora_seconds,
        },
        "throughput_questions_per_second": {
            "hf_lora": len(rows) / hf_inference_seconds,
            "vllm_base": len(rows) / base_seconds,
            "vllm_lora": len(rows) / lora_seconds,
        },
        "predictions": str(predictions_path.resolve()),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
