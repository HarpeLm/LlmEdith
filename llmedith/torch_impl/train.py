"""Pré-entraînement PyTorch (RTX 4060 Ti, GPU Colab et Kaggle, 2×T4 via torchrun).

Exemples :
    python -m llmedith.torch_impl.train --config configs/tiny.yaml
    torchrun --nproc_per_node 2 -m llmedith.torch_impl.train --config configs/small.yaml --resume
    python -m llmedith.torch_impl.train --config configs/tiny.yaml --override train.optimizer=adamw
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from llmedith import hub
from llmedith.config import Config
from llmedith.data import ShardLoader, iter_val_batches
from llmedith.schedule import in_decay_phase, wsd_lr_factor
from llmedith.torch_impl.model import LlmEdith
from llmedith.torch_impl.muon import build_optimizers, native_bf16

# Pic bf16 dense approximatif (TFLOPs, accumulation fp32), pour le MFU
PEAK_TFLOPS = {"4060 Ti": 44.0, "T4": 65.0, "P100": 19.0, "L4": 121.0, "A100": 312.0, "H100": 989.0}


def setup_device():
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if native_bf16(device) else torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    return ddp, rank, world, device, dtype


def model_state(model) -> dict:
    sd = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    if model.cfg.tie_embeddings:
        sd.pop("lm_head.weight", None)
    return sd


def load_model_state(model, path: str) -> None:
    missing, unexpected = model.load_state_dict(load_file(path), strict=False)
    missing = [m for m in missing if not (m == "lm_head.weight" and model.cfg.tie_embeddings)]
    assert not missing and not unexpected, (missing, unexpected)


def save_checkpoint(ckpt_dir, model, optimizers, scaler, step, loader, anneal_loader, cfg):
    os.makedirs(ckpt_dir, exist_ok=True)
    save_file(model_state(model), os.path.join(ckpt_dir, "model.safetensors"))
    torch.save({"optimizers": [o.state_dict() for o in optimizers],
                "scaler": scaler.state_dict() if scaler is not None else None},
               os.path.join(ckpt_dir, "optim.pt"))
    meta = {"step": step, "loader": loader.state_dict(),
            "anneal_loader": anneal_loader.state_dict() if anneal_loader else None,
            "config": cfg.to_dict(), "framework": "torch"}
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--resume", action="store_true", help="reprend depuis le dernier checkpoint (local, sinon Hub)")
    ap.add_argument("--init_from", default="", help="dossier de checkpoint dont on ne charge que les poids")
    ap.add_argument("--hub_repo", default=os.environ.get("LLMEDITH_HUB_REPO", ""))
    ap.add_argument("--max_steps", type=int, default=0, help="arrête après N steps (tests, sessions courtes)")
    ap.add_argument("--max_minutes", type=float, default=0, help="arrête proprement avant la fin du quota")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config, args.override)
    mc, tc, dc = cfg.model, cfg.train, cfg.data
    ddp, rank, world, device, dtype = setup_device()
    master = rank == 0
    torch.manual_seed(tc.seed + rank)
    run_dir = os.path.join(args.out_dir, cfg.name)
    ckpt_dir = os.path.join(run_dir, "latest")

    model = LlmEdith(mc).to(device)
    model.gradient_checkpointing = tc.remat
    optimizers = build_optimizers(model, tc)
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None

    loader = ShardLoader(dc.train_pattern, tc.global_batch_seqs, tc.seq_len, dc.hub_repo)
    anneal_loader = ShardLoader(dc.anneal_pattern, tc.global_batch_seqs, tc.seq_len, dc.hub_repo) if dc.anneal_pattern else None
    step = 0
    rewarm_from = None  # step où l'optimiseur a été réinitialisé : on refait un court warmup

    if args.resume:
        if not os.path.exists(os.path.join(ckpt_dir, "meta.json")) and master:
            hub.pull_folder(args.hub_repo, f"{cfg.name}/latest", args.out_dir)
        if ddp:
            dist.barrier()
        if os.path.exists(os.path.join(ckpt_dir, "meta.json")):
            meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))
            load_model_state(model, os.path.join(ckpt_dir, "model.safetensors"))
            if meta.get("framework") == "torch":
                opt = torch.load(os.path.join(ckpt_dir, "optim.pt"), map_location=device, weights_only=False)
                for o, s in zip(optimizers, opt["optimizers"]):
                    o.load_state_dict(s)
                if scaler is not None and opt["scaler"]:
                    scaler.load_state_dict(opt["scaler"])
            else:
                rewarm_from = meta["step"]
            step = meta["step"]
            loader.load_state_dict(meta["loader"])
            if anneal_loader and meta.get("anneal_loader"):
                anneal_loader.load_state_dict(meta["anneal_loader"])
            if master:
                print(f"Reprise au step {step}")
    elif args.init_from:
        load_model_state(model, os.path.join(args.init_from, "model.safetensors"))

    train_model = LossWrapper(model, tc)
    if args.compile:
        train_model = torch.compile(train_model)
    if ddp:
        train_model = torch.nn.parallel.DistributedDataParallel(train_model, device_ids=[device.index])

    assert tc.global_batch_seqs % (tc.micro_batch_seqs * world) == 0, "global_batch_seqs doit être divisible par micro×world"
    accum = tc.global_batch_seqs // (tc.micro_batch_seqs * world)
    total_steps = tc.total_steps
    autocast = (torch.autocast(device.type, dtype=dtype) if dtype != torch.float32 else contextlib.nullcontext())
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
    peak = next((v for k, v in PEAK_TFLOPS.items() if k in gpu_name), None)
    flops_per_step = mc.flops_per_token(tc.seq_len) * tc.tokens_per_step

    if master:
        os.makedirs(run_dir, exist_ok=True)
        print(f"LlmEdith {cfg.name} : {model.num_params() / 1e6:.1f}M paramètres | {gpu_name} | {dtype} | "
              f"world={world} accum={accum} | {total_steps} steps × {tc.tokens_per_step} tokens")
        log_f = open(os.path.join(run_dir, "log.csv"), "a", newline="")
        log = csv.writer(log_f)
        if args.wandb:
            import wandb
            wandb.init(project="llmedith", name=cfg.name, config=cfg.to_dict(), resume="allow")

    def evaluate() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for x, y in iter_val_batches(dc.val_pattern, tc.micro_batch_seqs, tc.seq_len, tc.eval_tokens, dc.hub_repo):
                with autocast:
                    _, ce = model.loss(torch.from_numpy(x).to(device), torch.from_numpy(y).long().to(device),
                                       chunk=tc.loss_chunk)
                losses.append(ce.item())
        model.train()
        return float(np.mean(losses))

    model.train()
    t_start = t_ckpt = time.time()
    while step < total_steps:
        t0 = time.time()
        decay = in_decay_phase(step, total_steps, tc.decay_frac)
        src = anneal_loader if (decay and anneal_loader) else loader
        x_np, y_np = src.next_batch()
        rows = slice(rank * tc.global_batch_seqs // world, (rank + 1) * tc.global_batch_seqs // world)
        x = torch.from_numpy(x_np[rows]).to(device, non_blocking=True)
        y = torch.from_numpy(y_np[rows]).long().to(device, non_blocking=True)

        lr = tc.lr * wsd_lr_factor(step, total_steps, tc.warmup_steps, tc.decay_frac, tc.min_lr_frac)
        if rewarm_from is not None:
            lr *= min(1.0, (step - rewarm_from + 1) / max(1, tc.warmup_steps // 4))
        for o in optimizers:
            for g in o.param_groups:
                g["lr"] = lr

        loss_acc = 0.0
        for i in range(accum):
            sl = slice(i * tc.micro_batch_seqs, (i + 1) * tc.micro_batch_seqs)
            sync = (not ddp) or i == accum - 1
            ctx = contextlib.nullcontext() if sync else train_model.no_sync()
            with ctx, autocast:
                loss, ce = train_model(x[sl], y[sl])
                loss = loss / accum
            (scaler.scale(loss) if scaler else loss).backward()
            loss_acc += ce.item() / accum

        if scaler:
            for o in optimizers:
                scaler.unscale_(o)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip).item()
        for o in optimizers:
            (scaler.step(o) if scaler else o.step())
        if scaler:
            scaler.update()
        for o in optimizers:
            o.zero_grad(set_to_none=True)
        step += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        if master and (step % tc.log_every == 0 or step == 1):
            tps = tc.tokens_per_step / dt
            mfu = flops_per_step / dt / (peak * 1e12 * world) if peak else float("nan")
            print(f"step {step}/{total_steps} | loss {loss_acc:.4f} | lr {lr:.2e} | gnorm {gnorm:.2f} | "
                  f"{tps / 1e3:.1f}k tok/s | MFU {mfu:.1%}")
            log.writerow([step, "train", round(loss_acc, 5), lr, round(gnorm, 4), int(tps)])
            log_f.flush()
            if args.wandb:
                wandb.log({"train/loss": loss_acc, "lr": lr, "gnorm": gnorm, "tok_s": tps, "mfu": mfu}, step=step)

        if step % tc.eval_every == 0 or step == total_steps:
            val = evaluate()
            if master:
                print(f"== step {step} | val loss {val:.4f} | ppl {math.exp(val):.2f}")
                log.writerow([step, "val", round(val, 5), lr, "", ""])
                log_f.flush()
                if args.wandb:
                    wandb.log({"val/loss": val}, step=step)

        elapsed_min = (time.time() - t_start) / 60
        stop = (args.max_steps and step >= args.max_steps) or (args.max_minutes and elapsed_min >= args.max_minutes)
        if master and ((time.time() - t_ckpt) / 60 >= tc.ckpt_every_minutes or stop or step == total_steps):
            save_checkpoint(ckpt_dir, model, optimizers, scaler, step, loader, anneal_loader, cfg)
            hub.push_folder(args.hub_repo, ckpt_dir, f"{cfg.name}/latest")
            t_ckpt = time.time()
            print(f"Checkpoint sauvegardé (step {step})")
        if stop:
            break

    if master:
        hub.wait_uploads()
        log_f.close()
    if ddp:
        dist.destroy_process_group()


class LossWrapper(torch.nn.Module):
    """DDP et torch.compile passent par forward() : on y branche directement la loss."""

    def __init__(self, model, tc):
        super().__init__()
        self.model, self.tc = model, tc

    def forward(self, x, y):
        return self.model.loss(x, y, z_loss=self.tc.z_loss, chunk=self.tc.loss_chunk)


if __name__ == "__main__":
    main()
