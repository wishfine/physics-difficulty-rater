#!/usr/bin/env python3
"""Train the V2 multi-task physics difficulty model."""
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from physics_difficulty.data.dataset import DifficultyDataset
from physics_difficulty.models.qwen_difficulty import QwenDifficultyRater
from physics_difficulty.schema import MULTI_LABEL_FEATURES

def parse_args():
    first = argparse.ArgumentParser(add_help=False); first.add_argument("--config")
    known, _ = first.parse_known_args(); defaults = json.load(open(known.config, encoding="utf-8")) if known.config else {}
    parser = argparse.ArgumentParser(parents=[first]); parser.set_defaults(**defaults)
    parser.add_argument("--model_path", required=True); parser.add_argument("--train_file", required=True); parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=1024); parser.add_argument("--batch_size", type=int, default=1); parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5); parser.add_argument("--num_train_epochs", type=int, default=3); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature_loss_weight", type=float, default=.3); parser.add_argument("--ordinal_loss_weight", type=float, default=.2); parser.add_argument("--bf16", action="store_true", default=True)
    return parser.parse_args()

def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)

def main():
    args = parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    distributed = "WORLD_SIZE" in os.environ
    if distributed:
        torch.distributed.init_process_group("nccl"); rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(rank); device = f"cuda:{rank}"
    else: rank, device = 0, ("cuda" if torch.cuda.is_available() else "cpu")
    main_process = not distributed or int(os.environ["RANK"]) == 0
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32
    backbone = AutoModel.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True, device_map={"": rank} if torch.cuda.is_available() else None)
    from peft import LoraConfig, get_peft_model
    targets = sorted({name.split(".")[-1] for name, module in backbone.named_modules() if isinstance(module, torch.nn.Linear)})
    backbone = get_peft_model(backbone, LoraConfig(r=8, lora_alpha=16, target_modules=targets, lora_dropout=.05, bias="none", task_type="FEATURE_EXTRACTION"))
    model = QwenDifficultyRater(backbone).to(device)
    if distributed: model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    dataset = DifficultyDataset(args.train_file, tokenizer, args.max_length)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None, collate_fn=dataset.collate_fn)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=.01)
    steps_per_epoch = max(1, (len(loader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps)
    steps = steps_per_epoch * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(.1 * steps), steps)
    optimizer.zero_grad(); optimizer_step = 0
    metrics_path = Path(args.output_dir) / "training_metrics.jsonl"
    if main_process:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("", encoding="utf-8")
    for epoch in range(args.num_train_epochs):
        if sampler: sampler.set_epoch(epoch)
        model.train(); total_loss = 0.0
        accumulated_micro_steps = 0
        for micro_step, batch in enumerate(loader, 1):
            outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device)); weights = batch["sample_weights"].to(device)
            labels = batch["difficulty_labels"].to(device)
            difficulty_loss = weighted_mean(F.cross_entropy(outputs["difficulty_logits"].float(), labels, reduction="none"), weights)
            expected = torch.softmax(outputs["difficulty_logits"].float(), -1).mul(torch.arange(5, device=device)).sum(-1)
            ordinal_loss = weighted_mean(F.smooth_l1_loss(expected, labels.float(), reduction="none"), weights)
            feature_losses = []
            for name, logits in outputs["feature_logits"].items():
                targets = batch["feature_labels"][name].to(device)
                if name in MULTI_LABEL_FEATURES:
                    per_sample = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none").mean(dim=-1)
                else:
                    per_sample = F.cross_entropy(logits.float(), targets, reduction="none")
                feature_losses.append(weighted_mean(per_sample, weights))
            loss = difficulty_loss + args.ordinal_loss_weight * ordinal_loss + args.feature_loss_weight * torch.stack(feature_losses).mean()
            (loss / args.gradient_accumulation_steps).backward(); total_loss += loss.item(); accumulated_micro_steps += 1
            if micro_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); scheduler.step(); optimizer.zero_grad(); optimizer_step += 1
        # Flush a partial final accumulation window instead of losing its gradients.
        if accumulated_micro_steps % args.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); scheduler.step(); optimizer.zero_grad(); optimizer_step += 1
        epoch_metrics = {"epoch": epoch + 1, "mean_loss": total_loss / max(1, len(loader)), "optimizer_step": optimizer_step}
        if main_process:
            print(json.dumps(epoch_metrics, ensure_ascii=False))
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")
    if main_process:
        raw = model.module if hasattr(model, "module") else model; output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
        raw.backbone.save_pretrained(output / "adapter"); tokenizer.save_pretrained(output / "tokenizer")
        torch.save({"norm": raw.norm.state_dict(), "difficulty_head": raw.difficulty_head.state_dict(), "feature_heads": raw.feature_heads.state_dict()}, output / "difficulty_heads.pt")
        (output / "training_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    if distributed: torch.distributed.destroy_process_group()

if __name__ == "__main__": main()
