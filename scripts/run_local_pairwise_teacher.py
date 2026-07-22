#!/usr/bin/env python3
"""Run resumable, bidirectional Qwen3-32B voting with adaptive sampling."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.pairwise.labels import aggregate_pair_votes

VOTE_PROMPT_VERSION = "physics_pair_vote_v1"
SYSTEM_PROMPT = "你是严谨的初中物理教研专家。"
USER_TEMPLATE = """请比较下面两道题对于初中学生独立解题时的真实难度。
综合考虑物理建模、知识整合、推理深度、信息加工、必要计算和隐含约束。
不要仅根据题目长度、解析长度、数字大小或机械步骤数量判断。
解析只用于判断学生独立解题所需的过程，不是判断阅读解析的难度。

如果题目A更难，只输出 A。
如果题目B更难，只输出 B。
不要输出解释或其他字符。

【题目A】
{text_a}

【题目B】
{text_b}
"""
FINAL_VOTE = re.compile(r"(?:^|\n)\s*(?:(?:最终答案|答案)\s*[:：]?\s*)?(A|B)\s*[。.]?$", re.IGNORECASE)


def load_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_vote(text: str) -> str | None:
    cleaned = str(text or "").strip()
    match = FINAL_VOTE.search(cleaned)
    return match.group(1).upper() if match else None


def ordered_pair(pair: Dict[str, Any], direction: str) -> tuple[str, str, str, str]:
    if direction == "forward":
        return str(pair["question_a_id"]), pair["question_a_text"], str(pair["question_b_id"]), pair["question_b_text"]
    if direction == "backward":
        return str(pair["question_b_id"]), pair["question_b_text"], str(pair["question_a_id"]), pair["question_a_text"]
    raise ValueError(f"unknown direction: {direction}")


def build_prompt(tokenizer: Any, pair: Dict[str, Any], direction: str, enable_thinking: bool) -> str:
    _, first_text, _, second_text = ordered_pair(pair, direction)
    user = USER_TEMPLATE.format(text_a=first_text, text_b=second_text)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    except TypeError:
        # Older tokenizer versions do not expose enable_thinking.  Qwen3 also
        # accepts an explicit /no_think directive in the user message.
        if not enable_thinking:
            messages[-1]["content"] = user + "\n/no_think"
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def winner_from_position(pair: Dict[str, Any], direction: str, vote: str | None) -> str | None:
    if vote not in {"A", "B"}:
        return None
    first_id, _, second_id, _ = ordered_pair(pair, direction)
    return first_id if vote == "A" else second_id


def group_votes(rows: Iterable[Dict[str, Any]]) -> dict[str, dict[str, list[Dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["pair_id"])][str(row["direction"])].append(row)
    return grouped


def valid_count(rows: Iterable[Dict[str, Any]]) -> int:
    return sum(bool(row.get("valid")) for row in rows)


def summarize_vote_rows(rows: Iterable[Dict[str, Any]], generation_seconds: float) -> Dict[str, Any]:
    rows = list(rows)
    valid_rows = [row for row in rows if row.get("valid")]
    output_tokens = sum(int(row.get("output_token_count", 0) or 0) for row in rows)
    valid_output_tokens = sum(int(row.get("output_token_count", 0) or 0) for row in valid_rows)
    return {
        "total_vote_rows": len(rows),
        "valid_votes": len(valid_rows),
        "parse_success_rate": len(valid_rows) / max(1, len(rows)),
        "output_tokens": output_tokens,
        "valid_output_tokens": valid_output_tokens,
        "mean_output_tokens_per_valid_vote": valid_output_tokens / max(1, len(valid_rows)),
        "generation_wall_seconds": generation_seconds,
        "valid_votes_per_second": len(valid_rows) / generation_seconds if generation_seconds > 0 else None,
    }


def desired_votes(stats: Dict[str, Any], initial: int, uncertain: int, maximum: int, uncertainty_low: float, uncertainty_high: float, medium_gap: float, high_gap: float) -> int:
    target = float(stats["soft_target"])
    gap = float(stats["position_bias_gap"])
    if uncertainty_high <= target <= 1 - uncertainty_high or gap > high_gap:
        return maximum
    if uncertainty_low <= target <= 1 - uncertainty_low or gap > medium_gap:
        return uncertain
    return initial


def config_hash(args: argparse.Namespace) -> str:
    relevant = {key: value for key, value in vars(args).items() if key not in {"dry_run"}}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    known, _ = bootstrap.parse_known_args()
    defaults = json.loads(Path(known.config).read_text(encoding="utf-8")) if known.config else {}
    parser = argparse.ArgumentParser(parents=[bootstrap])
    parser.set_defaults(**defaults)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--raw-votes-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", required="model_path" not in defaults)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--initial-samples-per-direction", type=int, default=3)
    parser.add_argument("--uncertain-samples-per-direction", type=int, default=5)
    parser.add_argument("--maximum-samples-per-direction", type=int, default=10)
    parser.add_argument("--uncertainty-low", type=float, default=0.30)
    parser.add_argument("--uncertainty-high", type=float, default=0.40)
    parser.add_argument("--medium-position-gap", type=float, default=0.15)
    parser.add_argument("--high-position-gap", type=float, default=0.30)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mode-name", default="nonthinking")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.temperature or not 0 < args.top_p <= 1:
        raise ValueError("stochastic voting requires temperature > 0 and top-p in (0, 1]")
    if args.top_k < -1 or not 0 <= args.min_p <= 1:
        raise ValueError("top-k must be -1 or non-negative and min-p must be in [0, 1]")
    if args.enable_thinking and args.max_new_tokens < 64:
        raise ValueError("thinking mode requires max-new-tokens >= 64 to reduce truncated votes")
    if not 1 <= args.initial_samples_per_direction <= args.uncertain_samples_per_direction <= args.maximum_samples_per_direction:
        raise ValueError("sample counts must satisfy initial <= uncertain <= maximum")
    if not 0 <= args.uncertainty_low <= args.uncertainty_high <= 0.5:
        raise ValueError("uncertainty thresholds must satisfy 0 <= low <= high <= 0.5")

    pairs = load_jsonl(Path(args.pairs))
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]
    pair_by_id = {str(pair["pair_id"]): pair for pair in pairs}
    if len(pair_by_id) != len(pairs):
        raise ValueError("pair IDs must be unique")

    raw_path = Path(args.raw_votes_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = load_jsonl(raw_path) if raw_path.is_file() else []
    current_config_hash = config_hash(args)
    previous_hashes = {str(row.get("run_config_hash")) for row in existing_rows}
    if previous_hashes and previous_hashes != {current_config_hash}:
        raise ValueError("raw vote file was created with a different or legacy teacher configuration; use a new output file")
    unknown = {str(row["pair_id"]) for row in existing_rows} - set(pair_by_id)
    if unknown:
        raise ValueError(f"raw vote file contains pairs absent from current candidate file: {len(unknown)}")
    grouped = group_votes(existing_rows)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.dry_run:
        if not pairs:
            raise ValueError("pair file is empty")
        preview = {
            "pair_id": pairs[0]["pair_id"],
            "forward_prompt": build_prompt(tokenizer, pairs[0], "forward", args.enable_thinking),
            "backward_prompt": build_prompt(tokenizer, pairs[0], "backward", args.enable_thinking),
            "teacher_mode": args.mode_name,
            "sampling": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "min_p": args.min_p, "max_new_tokens": args.max_new_tokens},
            "config_hash": config_hash(args),
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )

    generated_rows = 0
    new_generation_seconds = 0.0
    # Continue with a fresh sampling seed after interruption; otherwise an
    # invalid stochastic output could be reproduced forever on every restart.
    round_index = max((int(row.get("sampling_round", 0)) for row in existing_rows), default=0)
    rounds_this_run = 0
    warnings: list[str] = []
    while True:
        round_index += 1
        rounds_this_run += 1
        tasks: list[tuple[Dict[str, Any], str, int]] = []
        for pair in pairs:
            pair_id = str(pair["pair_id"])
            direction_rows = grouped[pair_id]
            forward_valid = valid_count(direction_rows.get("forward", []))
            backward_valid = valid_count(direction_rows.get("backward", []))
            desired = args.initial_samples_per_direction
            if forward_valid >= args.initial_samples_per_direction and backward_valid >= args.initial_samples_per_direction:
                try:
                    stats = aggregate_pair_votes(direction_rows["forward"] + direction_rows["backward"])
                    desired = desired_votes(
                        stats,
                        args.initial_samples_per_direction,
                        args.uncertain_samples_per_direction,
                        args.maximum_samples_per_direction,
                        args.uncertainty_low,
                        args.uncertainty_high,
                        args.medium_position_gap,
                        args.high_position_gap,
                    )
                except ValueError:
                    desired = args.initial_samples_per_direction
            for direction, count in (("forward", forward_valid), ("backward", backward_valid)):
                missing = max(0, desired - count)
                if missing:
                    tasks.append((pair, direction, missing))
        if not tasks:
            break
        if rounds_this_run > 8:
            warnings.append("Stopped after eight adaptive rounds because some prompts still lacked enough valid votes")
            break

        # vLLM accepts one SamplingParams object for a prompt batch, so group
        # tasks by the number of requested generations.
        for generation_count in sorted({task[2] for task in tasks}):
            matching = [task for task in tasks if task[2] == generation_count]
            for batch_start in range(0, len(matching), args.batch_size):
                batch = matching[batch_start:batch_start + args.batch_size]
                prompts = [build_prompt(tokenizer, pair, direction, args.enable_thinking) for pair, direction, _ in batch]
                sampling = SamplingParams(
                    n=generation_count,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    max_tokens=args.max_new_tokens,
                    seed=args.seed + round_index,
                )
                generation_started = time.perf_counter()
                responses = llm.generate(prompts, sampling, use_tqdm=True)
                new_generation_seconds += time.perf_counter() - generation_started
                if len(responses) != len(batch):
                    raise RuntimeError("vLLM returned a different number of prompt responses")
                with raw_path.open("a", encoding="utf-8") as target:
                    for (pair, direction, _), response in zip(batch, responses):
                        pair_id = str(pair["pair_id"])
                        existing_indices = [int(row.get("sample_index", -1)) for row in grouped[pair_id][direction]]
                        start_index = max(existing_indices, default=-1) + 1
                        for offset, candidate in enumerate(response.outputs):
                            raw_output = candidate.text
                            vote = parse_vote(raw_output)
                            finish_reason = getattr(candidate, "finish_reason", None)
                            output_token_count = len(getattr(candidate, "token_ids", []) or [])
                            valid = vote is not None and finish_reason != "length"
                            row = {
                                "schema_version": "qwen_pair_vote_v2",
                                "pair_id": pair_id,
                                "split": pair["split"],
                                "question_a_id": str(pair["question_a_id"]),
                                "question_b_id": str(pair["question_b_id"]),
                                "direction": direction,
                                "sample_index": start_index + offset,
                                "seed": args.seed + round_index,
                                "sampling_round": round_index,
                                "run_config_hash": current_config_hash,
                                "raw_output": raw_output,
                                "parsed_vote": vote,
                                "winner_question_id": winner_from_position(pair, direction, vote) if valid else None,
                                "valid": valid,
                                "output_token_count": output_token_count,
                                "finish_reason": finish_reason,
                                "stop_reason": getattr(candidate, "stop_reason", None),
                                "teacher": {
                                    "model": "Qwen3-32B",
                                    "model_path": str(Path(args.model_path).resolve()),
                                    "mode": args.mode_name,
                                    "prompt_version": VOTE_PROMPT_VERSION,
                                    "temperature": args.temperature,
                                    "top_p": args.top_p,
                                    "top_k": args.top_k,
                                    "min_p": args.min_p,
                                    "max_new_tokens": args.max_new_tokens,
                                    "thinking": args.enable_thinking,
                                },
                            }
                            target.write(json.dumps(row, ensure_ascii=False) + "\n")
                            grouped[pair_id][direction].append(row)
                            generated_rows += 1
                        target.flush()
        print(json.dumps({"message": "teacher_sampling_round_complete", "round": round_index, "generated_votes": generated_rows, "remaining_pairs_to_check": len(pairs)}, ensure_ascii=False), flush=True)

    all_rows = [row for directions in grouped.values() for rows in directions.values() for row in rows]
    previous_manifest_path = Path(args.manifest)
    previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8")) if previous_manifest_path.is_file() else {}
    previous_seconds = float(previous_manifest.get("generation_wall_seconds", 0.0) or 0.0) if previous_manifest.get("config_hash") == current_config_hash else 0.0
    vote_summary = summarize_vote_rows(all_rows, previous_seconds + new_generation_seconds)
    completed = 0
    for pair_id in pair_by_id:
        directions = grouped[pair_id]
        if valid_count(directions.get("forward", [])) >= args.initial_samples_per_direction and valid_count(directions.get("backward", [])) >= args.initial_samples_per_direction:
            completed += 1
    manifest = {
        "schema_version": "local_pairwise_teacher_run_v2",
        "pairs": str(Path(args.pairs).resolve()),
        "raw_votes_output": str(raw_path.resolve()),
        "teacher_model_path": str(Path(args.model_path).resolve()),
        "prompt_version": VOTE_PROMPT_VERSION,
        "teacher_mode": args.mode_name,
        "config_hash": current_config_hash,
        "images_uploaded": False,
        "pairs_requested": len(pairs),
        "pairs_completed_minimum": completed,
        **vote_summary,
        "new_votes_generated": generated_rows,
        "new_generation_wall_seconds": new_generation_seconds,
        "warnings": warnings,
        "config": vars(args),
    }
    manifest_path = previous_manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
