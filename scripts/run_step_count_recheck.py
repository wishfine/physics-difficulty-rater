#!/usr/bin/env python3
"""Blind, resumable Qwen3 review of the independent reasoning step count.

The reviewer input must be V3 label-free question rows.  Existing auxiliary
labels are deliberately rejected so that retrieval labels cannot leak into the
review prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.data.text_only import forbidden_source_label_paths, question_identifier
from physics_difficulty.schema import FEATURE_VALUES


PROMPT_VERSION = "physics_step_count_blind_recheck_v2_step3"
STEP_VALUES = set(FEATURE_VALUES["step_count"])
REVIEWER_FORBIDDEN_KEYS = {
    "teacher_features", "teacher_features_legacy18", "teacher_difficulty_id",
    "teacher_difficulty_level", "raw_difficulty", "difficulty", "label_quality",
    "label_source", "feature_metadata", "feature_schema_version", "label_schema_version",
}
SYSTEM_PROMPT = "你是严谨的初中物理教研专家。"
USER_TEMPLATE = """请判断下面这道题中，初中学生在不看解析的前提下，完成正确解题所需的“有效物理推理步骤”数量。

只计算必须的物理判断、建模、关系推导、关键计算或结论判断；不要把机械代数展开、重复代入、抄写题干、阅读解析算成额外步骤。题目有多个小问时，按完成整题所需的连续有效步骤总量判断。

只能从以下三档中选一档：
- 1-2步
- 3-5步
- 6步以上

请先在内部分析，最终只输出一个 JSON 对象，不要输出其他内容：
{{"step_count":"三档之一"}}

【题目】
{text}
"""
JSON_OBJECT = re.compile(r"\{\s*\"step_count\"\s*:\s*\"([^\"]+)\"\s*\}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_step_count(text: str) -> str | None:
    """Accept only an explicit final JSON step-count value."""
    matches = JSON_OBJECT.findall(str(text or ""))
    if not matches:
        return None
    candidate = matches[-1]
    return candidate if candidate in STEP_VALUES else None


def reviewer_label_paths(row: dict[str, Any]) -> list[str]:
    paths = list(forbidden_source_label_paths(row))

    def visit(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if str(key) in REVIEWER_FORBIDDEN_KEYS:
                    paths.append(path)
                visit(child, path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{prefix}[{index}]")

    visit(row)
    return sorted(set(paths))


def build_prompt(tokenizer: Any, question: dict[str, Any], enable_thinking: bool) -> str:
    text = str(question.get("text") or "").strip()
    if not text:
        raise ValueError(f"question {question_identifier(question)} has empty text")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
        )
    except TypeError:
        if not enable_thinking:
            messages[-1]["content"] += "\n/no_think"
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def configure_vllm_environment() -> None:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def config_hash(args: argparse.Namespace) -> str:
    relevant = {key: value for key, value in vars(args).items() if key != "dry_run"}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    known, _ = bootstrap.parse_known_args(argv)
    defaults = json.loads(Path(known.config).read_text(encoding="utf-8")) if known.config else {}
    parser = argparse.ArgumentParser(parents=[bootstrap])
    parser.add_argument("--input", required=True, help="Label-free V3 reviewer question JSONL")
    parser.add_argument("--raw-votes-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", required="model_path" not in defaults)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples-per-question", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--dry-run", action="store_true")
    known_destinations = {action.dest for action in parser._actions}
    unknown_config_keys = sorted(set(defaults) - known_destinations)
    if unknown_config_keys:
        raise ValueError(f"unknown recheck config keys: {unknown_config_keys}")
    parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not (0 < args.temperature and 0 < args.top_p <= 1):
        raise ValueError("temperature must be > 0 and top-p must be in (0, 1]")
    if args.top_k < -1 or not 0 <= args.min_p <= 1:
        raise ValueError("top-k must be -1 or non-negative and min-p must be in [0, 1]")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("gpu-memory-utilization must be in (0, 1)")
    if args.batch_size < 1 or args.samples_per_question < 1 or args.max_new_tokens < 1:
        raise ValueError("batch-size, samples-per-question and max-new-tokens must be positive")
    if args.enable_thinking and args.max_new_tokens < 64:
        raise ValueError("thinking mode requires max-new-tokens >= 64")

    questions = load_jsonl(Path(args.input))
    if args.max_questions is not None:
        questions = questions[:args.max_questions]
    by_id = {question_identifier(row): row for row in questions}
    if len(by_id) != len(questions):
        raise ValueError("reviewer input has duplicate question ids")
    for question_id, row in by_id.items():
        paths = reviewer_label_paths(row)
        if paths:
            raise ValueError(f"reviewer input question {question_id} is not label-free: {paths}")
        if not str(row.get("text") or "").strip():
            raise ValueError(f"reviewer input question {question_id} has empty text")

    raw_path = Path(args.raw_votes_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    previous_rows = load_jsonl(raw_path) if raw_path.is_file() else []
    current_hash = config_hash(args)
    hashes = {str(row.get("run_config_hash")) for row in previous_rows}
    if hashes and hashes != {current_hash}:
        raise ValueError("raw vote file belongs to another configuration; use a new output file")
    counts: dict[str, int] = {question_id: 0 for question_id in by_id}
    for row in previous_rows:
        question_id = str(row.get("question_id"))
        if question_id not in counts:
            raise ValueError(f"raw vote file contains unknown question id {question_id}")
        if row.get("valid"):
            counts[question_id] += 1

    if args.dry_run:
        print(json.dumps({
            "questions": len(questions), "label_free": True,
            "samples_per_question": args.samples_per_question,
            "prompt_version": PROMPT_VERSION, "config_hash": current_hash,
        }, ensure_ascii=False, indent=2))
        return

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    configure_vllm_environment()
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model_path, tokenizer=args.model_path, trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size, dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens, max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
    )
    pending = [row for question_id, row in by_id.items() if counts[question_id] < args.samples_per_question]
    generated = 0
    started = time.perf_counter()
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        prompts = [build_prompt(tokenizer, row, args.enable_thinking) for row in batch]
        missing = [args.samples_per_question - counts[question_identifier(row)] for row in batch]
        if len(set(missing)) != 1:
            # vLLM shares n across a prompt batch; the common case is a new run.
            # Resumed partial items are handled one prompt at a time below.
            for row, item_missing in zip(batch, missing):
                if item_missing:
                    responses = llm.generate([build_prompt(tokenizer, row, args.enable_thinking)], SamplingParams(
                        n=item_missing, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                        min_p=args.min_p, max_tokens=args.max_new_tokens, seed=args.seed + counts[question_identifier(row)],
                    ), use_tqdm=True)
                    _write_votes(raw_path, row, responses[0], counts, args, current_hash)
                    generated += item_missing
            continue
        responses = llm.generate(prompts, SamplingParams(
            n=missing[0], temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            min_p=args.min_p, max_tokens=args.max_new_tokens, seed=args.seed,
        ), use_tqdm=True)
        for row, response in zip(batch, responses):
            generated += _write_votes(raw_path, row, response, counts, args, current_hash)
        print(json.dumps({"message": "step_count_recheck_progress", "completed_questions": sum(count >= args.samples_per_question for count in counts.values()), "total_questions": len(questions)}, ensure_ascii=False), flush=True)

    all_rows = load_jsonl(raw_path)
    valid = [row for row in all_rows if row.get("valid")]
    elapsed = time.perf_counter() - started
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "schema_version": "step_count_blind_recheck_v1", "input": str(Path(args.input).resolve()),
        "raw_votes_output": str(raw_path.resolve()), "question_count": len(questions),
        "completed_questions": sum(count >= args.samples_per_question for count in counts.values()),
        "samples_per_question": args.samples_per_question, "total_vote_rows": len(all_rows),
        "valid_vote_rows": len(valid), "parse_success_rate": len(valid) / max(1, len(all_rows)),
        "new_votes_generated": generated, "new_generation_wall_seconds": elapsed,
        "teacher_model_path": str(Path(args.model_path).resolve()), "prompt_version": PROMPT_VERSION,
        "images_uploaded": False, "reviewer_input_label_free": True,
        "config_hash": current_hash, "config": vars(args),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_votes(raw_path: Path, question: dict[str, Any], response: Any, counts: dict[str, int], args: argparse.Namespace, run_config_hash: str) -> int:
    question_id = question_identifier(question)
    written = 0
    with raw_path.open("a", encoding="utf-8") as target:
        for output in response.outputs:
            raw_output = str(output.text or "")
            parsed = parse_step_count(raw_output)
            finish_reason = getattr(output, "finish_reason", None)
            valid = parsed is not None and finish_reason != "length"
            target.write(json.dumps({
                "schema_version": "qwen_step_count_recheck_vote_v1", "question_id": question_id,
                "sample_index": counts[question_id], "seed": args.seed,
                "run_config_hash": run_config_hash, "raw_output": raw_output,
                "parsed_step_count": parsed, "valid": valid,
                "finish_reason": finish_reason, "stop_reason": getattr(output, "stop_reason", None),
                "output_token_count": len(getattr(output, "token_ids", []) or []),
                "teacher": {"model": "Qwen3-32B", "model_path": str(Path(args.model_path).resolve()),
                            "prompt_version": PROMPT_VERSION, "thinking": args.enable_thinking,
                            "temperature": args.temperature, "top_p": args.top_p,
                            "top_k": args.top_k, "min_p": args.min_p, "max_new_tokens": args.max_new_tokens},
            }, ensure_ascii=False) + "\n")
            if valid:
                counts[question_id] += 1
            written += 1
    return written


if __name__ == "__main__":
    main()
