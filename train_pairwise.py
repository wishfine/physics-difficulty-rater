#!/usr/bin/env python3
"""Train a resumable Qwen LoRA scalar rater with soft Bradley-Terry loss."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from physics_difficulty.data.pairwise_dataset import PairwiseDifficultyDataset
from physics_difficulty.models.qwen_pairwise import QwenPairwiseRater
from physics_difficulty.pairwise.losses import auxiliary_loss_weight, normalized_auxiliary_loss


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    known, _ = bootstrap.parse_known_args()
    defaults = json.loads(Path(known.config).read_text(encoding="utf-8")) if known.config else {}
    parser = argparse.ArgumentParser(parents=[bootstrap])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1, help="pairs per GPU micro-batch")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="LoRA learning rate")
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--checkpoint-every-epochs", type=float, default=0.25)
    parser.add_argument("--log-every-optimizer-steps", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--score-regularization-weight", type=float, default=1e-4)
    parser.add_argument("--auxiliary-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--auxiliary-loss-weight", type=float, default=0.1)
    parser.add_argument("--auxiliary-warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    known_destinations = {action.dest for action in parser._actions}
    unknown_config_keys = sorted(set(defaults) - known_destinations)
    if unknown_config_keys:
        raise ValueError(f"unknown pairwise training config keys: {unknown_config_keys}")
    # Apply JSON defaults after registering arguments; argparse otherwise lets
    # each add_argument(default=...) silently overwrite the config value.
    parser.set_defaults(**defaults)
    return parser.parse_args()


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def make_loader(dataset: PairwiseDifficultyDataset, args: argparse.Namespace, epoch: int, world_size: int, rank: int) -> DataLoader:
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        sampler.set_epoch(epoch)
    else:
        generator = torch.Generator().manual_seed(args.seed + epoch)
        sampler = RandomSampler(dataset, generator=generator)
    return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=dataset.collate_fn)


def save_checkpoint(model: torch.nn.Module, tokenizer: Any, optimizer: torch.optim.Optimizer, scheduler: Any, output_dir: Path, name: str, trainer_state: dict[str, Any], args: argparse.Namespace) -> Path:
    raw = model.module if hasattr(model, "module") else model
    checkpoint = output_dir / name
    checkpoint.mkdir(parents=True, exist_ok=True)
    raw.backbone.save_pretrained(checkpoint / "adapter")
    tokenizer.save_pretrained(checkpoint / "tokenizer")
    head_state = {"norm": raw.norm.state_dict(), "score_head": raw.score_head.state_dict()}
    if raw.auxiliary_features:
        head_state["auxiliary_heads"] = raw.auxiliary_heads.state_dict()
    torch.save(head_state, checkpoint / "pairwise_head.pt")
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
    (checkpoint / "trainer_state.json").write_text(json.dumps(trainer_state, ensure_ascii=False, indent=2), encoding="utf-8")
    (checkpoint / "pairwise_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    return checkpoint


def main() -> None:
    args = parse_args()
    for value, name in ((args.max_length, "max-length"), (args.batch_size, "batch-size"), (args.gradient_accumulation_steps, "gradient-accumulation-steps"), (args.log_every_optimizer_steps, "log-every-optimizer-steps")):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.checkpoint_every_epochs <= 0 or not 0 <= args.warmup_ratio < 1:
        raise ValueError("checkpoint interval and warmup ratio are invalid")
    if args.auxiliary_loss_weight < 0 or not 0 <= args.auxiliary_warmup_ratio <= 1:
        raise ValueError("auxiliary loss settings are invalid")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        torch.distributed.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = global_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_process = global_rank == 0
    output_dir = Path(args.output_dir)
    resume_dir = Path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    if resume_dir and not (resume_dir / "trainer_state.json").is_file():
        raise ValueError(f"invalid resume checkpoint: {resume_dir}")

    tokenizer_source = resume_dir / "tokenizer" if resume_dir and (resume_dir / "tokenizer").is_dir() else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32
    base = AutoModel.from_pretrained(
        args.model_path,
        dtype=dtype,
        trust_remote_code=True,
        device_map={"": local_rank} if torch.cuda.is_available() else None,
    )
    if args.gradient_checkpointing:
        base.config.use_cache = False
        base.gradient_checkpointing_enable()
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()

    from peft import LoraConfig, PeftModel, get_peft_model
    if resume_dir:
        backbone = PeftModel.from_pretrained(base, resume_dir / "adapter", is_trainable=True)
    else:
        targets = sorted({name.split(".")[-1] for name, module in base.named_modules() if isinstance(module, torch.nn.Linear)})
        if not targets:
            raise RuntimeError("no linear modules found for LoRA")
        backbone = get_peft_model(base, LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        ))
    model = QwenPairwiseRater(backbone, auxiliary_features=args.auxiliary_features).to(device)
    if resume_dir:
        state = torch.load(resume_dir / "pairwise_head.pt", map_location=device)
        model.norm.load_state_dict(state["norm"])
        model.score_head.load_state_dict(state["score_head"])
        if args.auxiliary_features:
            if "auxiliary_heads" not in state:
                raise ValueError("cannot resume V2 training from a V1 checkpoint")
            model.auxiliary_heads.load_state_dict(state["auxiliary_heads"])

    head_parameters = list(model.norm.parameters()) + list(model.score_head.parameters()) + list(model.auxiliary_heads.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in head_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone_parameters, "lr": args.learning_rate},
        {"params": head_parameters, "lr": args.head_learning_rate},
    ], weight_decay=args.weight_decay)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    dataset = PairwiseDifficultyDataset(args.train_file, tokenizer, args.max_length, require_auxiliary_features=args.auxiliary_features)
    micro_batches_per_epoch = math.ceil(len(dataset) / (args.batch_size * world_size))
    optimizer_steps_per_epoch = max(1, math.ceil(micro_batches_per_epoch / args.gradient_accumulation_steps))
    total_steps = optimizer_steps_per_epoch * args.num_train_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)

    start_epoch = resume_micro_step = optimizer_step = 0
    resumed_loss_sum = 0.0
    resumed_micro_count = 0
    if resume_dir:
        optimizer.load_state_dict(torch.load(resume_dir / "optimizer.pt", map_location=device))
        scheduler.load_state_dict(torch.load(resume_dir / "scheduler.pt", map_location=device))
        trainer_state = json.loads((resume_dir / "trainer_state.json").read_text(encoding="utf-8"))
        start_epoch = int(trainer_state["next_epoch"])
        resume_micro_step = int(trainer_state.get("next_micro_step", 0))
        optimizer_step = int(trainer_state["optimizer_step"])
        resumed_loss_sum = float(trainer_state.get("epoch_loss_sum", 0.0))
        resumed_micro_count = int(trainer_state.get("epoch_micro_count", 0))

    if main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "training_metrics.jsonl"
        if not resume_dir:
            metrics_path.write_text("", encoding="utf-8")
        (output_dir / "training_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({
            "message": "Training soft Bradley-Terry with optional normalized auxiliary supervision; run evaluate_pairwise.py separately.",
            "records": len(dataset),
            "world_size": world_size,
            "effective_pair_batch_size": args.batch_size * world_size * args.gradient_accumulation_steps,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "start_epoch": start_epoch,
            "resume_micro_step": resume_micro_step,
            "auxiliary_features": args.auxiliary_features,
            "auxiliary_loss_weight": args.auxiliary_loss_weight if args.auxiliary_features else 0.0,
        }, ensure_ascii=False), flush=True)

    for epoch in range(start_epoch, args.num_train_epochs):
        loader = make_loader(dataset, args, epoch, world_size, global_rank)
        epoch_resume = resume_micro_step if epoch == start_epoch else 0
        loss_sum = resumed_loss_sum if epoch == start_epoch else 0.0
        micro_count = resumed_micro_count if epoch == start_epoch else 0
        updates_in_epoch = math.ceil(epoch_resume / args.gradient_accumulation_steps)
        checkpoint_interval = max(1, math.ceil(optimizer_steps_per_epoch * args.checkpoint_every_epochs))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        last_log_time = time.perf_counter()
        last_log_update = updates_in_epoch
        if main_process:
            print(json.dumps({"message": "Starting pairwise epoch", "epoch": epoch + 1, "micro_batches": len(loader), "checkpoint_every_optimizer_steps": checkpoint_interval}, ensure_ascii=False), flush=True)
        for micro_step, batch in enumerate(loader, 1):
            if micro_step <= epoch_resume:
                continue
            outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), int(batch["pair_count"]))
            targets = batch["soft_targets"].to(device)
            weights = batch["sample_weights"].to(device)
            pair_loss = weighted_mean(F.binary_cross_entropy_with_logits(outputs["pair_logits"].float(), targets, reduction="none"), weights)
            score_regularization = (outputs["score_a"].float().square().mean() + outputs["score_b"].float().square().mean()) / 2
            aux_loss = None
            current_aux_weight = 0.0
            if args.auxiliary_features:
                aux_loss = normalized_auxiliary_loss(
                    outputs["auxiliary_logits_a"], outputs["auxiliary_logits_b"],
                    {name: value.to(device) for name, value in batch["auxiliary_targets_a"].items()},
                    {name: value.to(device) for name, value in batch["auxiliary_targets_b"].items()},
                    batch["auxiliary_weights_a"].to(device), batch["auxiliary_weights_b"].to(device),
                    dataset.feature_class_weights,
                )
                current_aux_weight = auxiliary_loss_weight(
                    optimizer_step, total_steps, args.auxiliary_loss_weight, args.auxiliary_warmup_ratio,
                )
            loss = pair_loss + args.score_regularization_weight * score_regularization
            if aux_loss is not None:
                loss = loss + current_aux_weight * aux_loss
            (loss / args.gradient_accumulation_steps).backward()
            loss_sum += float(loss.item())
            micro_count += 1
            accumulated += 1
            boundary = accumulated % args.gradient_accumulation_steps == 0
            is_last = micro_step == len(loader)
            if boundary or is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                updates_in_epoch += 1
                if main_process and updates_in_epoch % args.log_every_optimizer_steps == 0:
                    elapsed = max(time.perf_counter() - last_log_time, 1e-6)
                    probabilities = torch.sigmoid(outputs["pair_logits"].float()).detach()
                    brier = weighted_mean((probabilities - targets).square(), weights).item()
                    print(json.dumps({
                        "epoch": epoch + 1,
                        "epoch_progress": round(micro_step / len(loader), 4),
                        "optimizer_step": optimizer_step,
                        "last_loss": round(float(loss.item()), 6),
                        "last_pair_loss": round(float(pair_loss.item()), 6),
                        "last_auxiliary_loss": round(float(aux_loss.item()), 6) if aux_loss is not None else None,
                        "auxiliary_loss_weight": round(current_aux_weight, 6),
                        "auxiliary_contribution_ratio": round(
                            current_aux_weight * float(aux_loss.item()) / max(float(pair_loss.item()), 1e-8), 6
                        ) if aux_loss is not None else 0.0,
                        "last_brier": round(float(brier), 6),
                        "optimizer_updates_per_second": round((updates_in_epoch - last_log_update) / elapsed, 4),
                    }, ensure_ascii=False), flush=True)
                    last_log_time = time.perf_counter()
                    last_log_update = updates_in_epoch
                if main_process and updates_in_epoch % checkpoint_interval == 0 and not is_last:
                    state = {"next_epoch": epoch, "next_micro_step": micro_step, "optimizer_step": optimizer_step, "epoch_loss_sum": loss_sum, "epoch_micro_count": micro_count, "seed": args.seed}
                    checkpoint = save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir, f"checkpoint-epoch-{epoch + 1}-step-{optimizer_step}", state, args)
                    print(f"Saved resumable pairwise checkpoint: {checkpoint}", flush=True)

        metrics = {"epoch": epoch + 1, "mean_loss": loss_sum / max(1, micro_count), "optimizer_step": optimizer_step}
        if main_process:
            with (output_dir / "training_metrics.jsonl").open("a", encoding="utf-8") as target:
                target.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            state = {"next_epoch": epoch + 1, "next_micro_step": 0, "optimizer_step": optimizer_step, "epoch_loss_sum": 0.0, "epoch_micro_count": 0, "seed": args.seed}
            checkpoint = save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir, f"checkpoint-epoch-{epoch + 1}", state, args)
            print(f"Saved resumable pairwise checkpoint: {checkpoint}", flush=True)
        if distributed:
            torch.distributed.barrier()
        resume_micro_step = 0
        resumed_loss_sum = 0.0
        resumed_micro_count = 0

    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
