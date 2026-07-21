#!/usr/bin/env python3
"""Train the V2 multi-task physics difficulty model with epoch-level resume."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from physics_difficulty.data.dataset import DifficultyDataset
from physics_difficulty.models.qwen_difficulty import QwenDifficultyRater


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    known, _ = bootstrap.parse_known_args()
    defaults = json.loads(Path(known.config).read_text(encoding="utf-8")) if known.config else {}
    parser = argparse.ArgumentParser(parents=[bootstrap])
    parser.set_defaults(**defaults)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--checkpoint_every_epochs", type=float, default=0.25)
    parser.add_argument("--log_every_optimizer_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature_loss_weight", type=float, default=0.3)
    parser.add_argument("--ordinal_loss_weight", type=float, default=0.2)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    output_dir: Path,
    checkpoint_name: str,
    next_epoch: int,
    next_micro_step: int,
    optimizer_step: int,
    epoch_loss_sum: float,
    epoch_micro_steps: int,
    args: argparse.Namespace,
) -> Path:
    raw = model.module if hasattr(model, "module") else model
    checkpoint = output_dir / checkpoint_name
    checkpoint.mkdir(parents=True, exist_ok=True)
    raw.backbone.save_pretrained(checkpoint / "adapter")
    tokenizer.save_pretrained(checkpoint / "tokenizer")
    torch.save({"norm": raw.norm.state_dict(), "difficulty_head": raw.difficulty_head.state_dict(), "feature_heads": raw.feature_heads.state_dict()}, checkpoint / "difficulty_heads.pt")
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
    state = {
        "next_epoch": next_epoch,
        "next_micro_step": next_micro_step,
        "optimizer_step": optimizer_step,
        "epoch_loss_sum": epoch_loss_sum,
        "epoch_micro_steps": epoch_micro_steps,
        "seed": args.seed,
    }
    (checkpoint / "trainer_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return checkpoint


def make_epoch_loader(
    dataset: DifficultyDataset,
    args: argparse.Namespace,
    epoch: int,
    world_size: int,
    global_rank: int,
) -> tuple[DataLoader, object]:
    """Build a reproducibly shuffled loader so a mid-epoch resume sees the same batches."""
    if world_size > 1:
        # Pass rank/world size explicitly. Relying on DistributedSampler's
        # implicit process-group lookup can silently give each rank the full
        # dataset when launcher environment is unusual.
        sampler: object = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True,
            seed=args.seed,
        )
        sampler.set_epoch(epoch)
        return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=dataset.collate_fn), sampler
    generator = torch.Generator()
    generator.manual_seed(args.seed + epoch)
    sampler = RandomSampler(dataset, generator=generator)
    return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=dataset.collate_fn), sampler


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    distributed = "WORLD_SIZE" in os.environ
    if distributed:
        torch.distributed.init_process_group("nccl")
        rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        rank, global_rank, world_size = 0, 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    main_process = global_rank == 0
    output_dir = Path(args.output_dir)
    resume_dir = Path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    if resume_dir and not (resume_dir / "trainer_state.json").is_file():
        raise ValueError(f"Invalid checkpoint directory: {resume_dir}")

    tokenizer_source = resume_dir / "tokenizer" if resume_dir and (resume_dir / "tokenizer").is_dir() else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32
    base = AutoModel.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True, device_map={"": rank} if torch.cuda.is_available() else None)
    if args.gradient_checkpointing:
        base.config.use_cache = False
        base.gradient_checkpointing_enable()
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
    from peft import LoraConfig, PeftModel, get_peft_model
    if resume_dir:
        backbone = PeftModel.from_pretrained(base, resume_dir / "adapter", is_trainable=True)
    else:
        # Use module suffixes, not classes, for PEFT target matching.
        targets = sorted({name.split(".")[-1] for name, module in base.named_modules() if isinstance(module, torch.nn.Linear)})
        if not targets:
            raise RuntimeError("No linear modules found for LoRA")
        backbone = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=targets, lora_dropout=0.05, bias="none", task_type="FEATURE_EXTRACTION"))
    model = QwenDifficultyRater(backbone).to(device)
    if resume_dir:
        head_state = torch.load(resume_dir / "difficulty_heads.pt", map_location=device)
        model.norm.load_state_dict(head_state["norm"])
        model.difficulty_head.load_state_dict(head_state["difficulty_head"])
        model.feature_heads.load_state_dict(head_state["feature_heads"])
    if distributed:
        # Losses are intentionally computed outside forward so they can be weighted
        # per sample. DDP therefore needs unused-parameter detection for auxiliary
        # heads and LoRA modules whose graph is not visible inside forward.
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    if args.checkpoint_every_epochs <= 0:
        raise ValueError("checkpoint_every_epochs must be positive")
    if args.log_every_optimizer_steps <= 0:
        raise ValueError("log_every_optimizer_steps must be positive")

    dataset = DifficultyDataset(args.train_file, tokenizer, args.max_length)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.learning_rate, weight_decay=0.01)
    micro_batches_per_epoch = math.ceil(len(dataset) / (args.batch_size * world_size))
    steps_per_epoch = max(1, math.ceil(micro_batches_per_epoch / args.gradient_accumulation_steps))
    total_steps = steps_per_epoch * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    start_epoch, resume_micro_step, optimizer_step = 0, 0, 0
    resumed_epoch_loss_sum, resumed_epoch_micro_steps = 0.0, 0
    if resume_dir:
        optimizer.load_state_dict(torch.load(resume_dir / "optimizer.pt", map_location=device))
        scheduler.load_state_dict(torch.load(resume_dir / "scheduler.pt", map_location=device))
        state = json.loads((resume_dir / "trainer_state.json").read_text(encoding="utf-8"))
        start_epoch = int(state["next_epoch"])
        resume_micro_step = int(state.get("next_micro_step", 0))
        optimizer_step = int(state["optimizer_step"])
        resumed_epoch_loss_sum = float(state.get("epoch_loss_sum", 0.0))
        resumed_epoch_micro_steps = int(state.get("epoch_micro_steps", 0))

    if main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "training_metrics.jsonl"
        if not resume_dir:
            metrics_path.write_text("", encoding="utf-8")
        (output_dir / "training_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"message": "Training reports loss only; run evaluate_difficulty.py separately for model selection.", "start_epoch": start_epoch, "resume_micro_step": resume_micro_step, "total_epochs": args.num_train_epochs, "checkpoint_every_epochs": args.checkpoint_every_epochs, "optimizer_steps_per_epoch": steps_per_epoch}, ensure_ascii=False), flush=True)

    for epoch in range(start_epoch, args.num_train_epochs):
        loader, _ = make_epoch_loader(dataset, args, epoch, world_size, global_rank)
        epoch_resume_micro_step = resume_micro_step if epoch == start_epoch else 0
        epoch_loss_sum = resumed_epoch_loss_sum if epoch == start_epoch else 0.0
        epoch_micro_steps = resumed_epoch_micro_steps if epoch == start_epoch else 0
        optimizer_updates_in_epoch = math.ceil(epoch_resume_micro_step / args.gradient_accumulation_steps)
        checkpoint_every_steps = max(1, math.ceil(steps_per_epoch * args.checkpoint_every_epochs))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_micro_steps = 0
        last_log_time = time.perf_counter()
        last_log_update = optimizer_updates_in_epoch
        if main_process:
            print(json.dumps({"message": "Starting epoch", "epoch": epoch + 1, "world_size": world_size, "per_gpu_batch_size": args.batch_size, "gradient_accumulation_steps": args.gradient_accumulation_steps, "micro_batches": len(loader), "optimizer_steps": steps_per_epoch, "checkpoint_every_optimizer_steps": checkpoint_every_steps}, ensure_ascii=False), flush=True)
        for micro_step, batch in enumerate(loader, 1):
            if micro_step <= epoch_resume_micro_step:
                continue
            outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            weights, labels = batch["sample_weights"].to(device), batch["difficulty_labels"].to(device)
            difficulty_loss = weighted_mean(F.cross_entropy(outputs["difficulty_logits"].float(), labels, reduction="none"), weights)
            expected = torch.softmax(outputs["difficulty_logits"].float(), -1).mul(torch.arange(5, device=device)).sum(-1)
            ordinal_loss = weighted_mean(F.smooth_l1_loss(expected, labels.float(), reduction="none"), weights)
            feature_losses = []
            for name, logits in outputs["feature_logits"].items():
                targets = batch["feature_labels"][name].to(device)
                per_sample = F.cross_entropy(logits.float(), targets, reduction="none")
                feature_losses.append(weighted_mean(per_sample, weights))
            loss = difficulty_loss + args.ordinal_loss_weight * ordinal_loss + args.feature_loss_weight * torch.stack(feature_losses).mean()
            (loss / args.gradient_accumulation_steps).backward()
            epoch_loss_sum += loss.item()
            epoch_micro_steps += 1
            accumulated_micro_steps += 1
            is_accumulation_boundary = accumulated_micro_steps % args.gradient_accumulation_steps == 0
            is_last_micro_batch = micro_step == len(loader)
            if is_accumulation_boundary or is_last_micro_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); optimizer_step += 1
                optimizer_updates_in_epoch += 1
                if main_process and optimizer_updates_in_epoch % args.log_every_optimizer_steps == 0:
                    elapsed = max(time.perf_counter() - last_log_time, 1e-6)
                    updates_since_log = optimizer_updates_in_epoch - last_log_update
                    print(json.dumps({
                        "epoch": epoch + 1,
                        "epoch_progress": round(micro_step / len(loader), 4),
                        "optimizer_step": optimizer_step,
                        "last_loss": round(loss.item(), 6),
                        "optimizer_updates_per_second": round(updates_since_log / elapsed, 4),
                    }, ensure_ascii=False), flush=True)
                    last_log_time = time.perf_counter()
                    last_log_update = optimizer_updates_in_epoch
                should_save_mid_epoch = (
                    optimizer_updates_in_epoch % checkpoint_every_steps == 0
                    and not is_last_micro_batch
                )
                if should_save_mid_epoch and main_process:
                    checkpoint = save_checkpoint(
                        model, tokenizer, optimizer, scheduler, output_dir,
                        f"checkpoint-epoch-{epoch + 1}-step-{optimizer_step}",
                        epoch, micro_step, optimizer_step, epoch_loss_sum, epoch_micro_steps, args,
                    )
                    print(f"Saved resumable checkpoint: {checkpoint}", flush=True)

        epoch_metrics = {"epoch": epoch + 1, "mean_loss": epoch_loss_sum / max(1, epoch_micro_steps), "optimizer_step": optimizer_step}
        if main_process:
            with (output_dir / "training_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")
            print(json.dumps(epoch_metrics, ensure_ascii=False), flush=True)
            checkpoint = save_checkpoint(
                model, tokenizer, optimizer, scheduler, output_dir,
                f"checkpoint-epoch-{epoch + 1}",
                epoch + 1, 0, optimizer_step, 0.0, 0, args,
            )
            print(f"Saved resumable checkpoint: {checkpoint}", flush=True)
        if distributed:
            torch.distributed.barrier()
        resume_micro_step, resumed_epoch_loss_sum, resumed_epoch_micro_steps = 0, 0.0, 0
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
