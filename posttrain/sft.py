"""SFT (fine-tuning supervisé) de LlmEdith sur des conversations en anglais (smoltalk par défaut).

    python -m posttrain.sft --base runs/main/latest --out runs/sft --epochs 2
    python -m convert.to_hf --ckpt runs/sft/latest --out exports/llmedith-chat --chat
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import time

import numpy as np
import torch

from llmedith.config import TrainConfig
from llmedith.schedule import wsd_lr_factor
from llmedith.torch_impl.muon import build_optimizers
from posttrain.common import ChatFormatter, load_checkpoint, pick_device_dtype, save_checkpoint


def pack(examples, seq_len: int, pad: int):
    """Regroupe plusieurs conversations dans chaque séquence de seq_len+1 tokens (peu de padding)."""
    rows, cur_ids, cur_mask = [], [], []
    for ids, mask in examples:
        ids, mask = ids[:seq_len + 1], mask[:seq_len + 1]
        if len(cur_ids) + len(ids) > seq_len + 1:
            rows.append((cur_ids, cur_mask))
            cur_ids, cur_mask = [], []
        cur_ids += ids
        cur_mask += mask
    if cur_ids:
        rows.append((cur_ids, cur_mask))
    X = np.full((len(rows), seq_len + 1), pad, dtype=np.int64)
    M = np.zeros((len(rows), seq_len + 1), dtype=np.float32)
    for i, (ids, mask) in enumerate(rows):
        X[i, :len(ids)], M[i, :len(mask)] = ids, mask
    return X, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="checkpoint pré-entraîné (dossier avec model.safetensors)")
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--tokenizer", default="tokenizer/llmedith-32k/tokenizer.json")
    ap.add_argument("--dataset", default="HuggingFaceTB/smoltalk")
    ap.add_argument("--dataset_config", default="all")
    ap.add_argument("--max_examples", type=int, default=400_000)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_seqs", type=int, default=32)
    ap.add_argument("--micro_batch_seqs", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--optimizer", default="muon")
    args = ap.parse_args()

    device, dtype = pick_device_dtype()
    model, meta = load_checkpoint(args.base, device)
    model.gradient_checkpointing = True
    fmt = ChatFormatter(args.tokenizer)

    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.dataset_config, split="train", streaming=True)
    ds = ds.shuffle(seed=0, buffer_size=20_000).take(args.max_examples)
    print(f"Tokenisation de {args.max_examples} conversations …")
    examples = [fmt.encode(ex["messages"]) for ex in ds]
    random.Random(0).shuffle(examples)
    X, M = pack(examples, args.seq_len, fmt.pad)
    print(f"{len(X)} séquences packées de {args.seq_len} tokens ; {M.sum() / 1e6:.1f}M tokens de réponse entraînés")

    tc = TrainConfig(lr=args.lr, optimizer=args.optimizer, weight_decay=0.0)
    optimizers = build_optimizers(model, tc)
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None
    autocast = torch.autocast(device.type, dtype=dtype) if dtype != torch.float32 else contextlib.nullcontext()
    steps_per_epoch = len(X) // args.batch_seqs
    total = max(1, int(steps_per_epoch * args.epochs))
    accum = args.batch_seqs // args.micro_batch_seqs
    model.train()
    step, t0 = 0, time.time()
    while step < total:
        perm = np.random.RandomState(step).permutation(len(X))
        for b in range(steps_per_epoch):
            if step >= total:
                break
            idx = perm[b * args.batch_seqs:(b + 1) * args.batch_seqs]
            lr = args.lr * wsd_lr_factor(step, total, max(1, total // 50), 0.5)
            for o in optimizers:
                for g in o.param_groups:
                    g["lr"] = lr
            loss_acc = 0.0
            for i in range(accum):
                sl = idx[i * args.micro_batch_seqs:(i + 1) * args.micro_batch_seqs]
                x = torch.from_numpy(X[sl, :-1]).to(device)
                y = torch.from_numpy(X[sl, 1:]).to(device)
                m = torch.from_numpy(M[sl, 1:]).to(device)
                with autocast:
                    loss, ce = model.loss(x, y, mask=m)
                ((scaler.scale(loss / accum)) if scaler else loss / accum).backward()
                loss_acc += ce.item() / accum
            if scaler:
                for o in optimizers:
                    scaler.unscale_(o)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for o in optimizers:
                scaler.step(o) if scaler else o.step()
                o.zero_grad(set_to_none=True)
            if scaler:
                scaler.update()
            step += 1
            if step % 10 == 0 or step == 1:
                print(f"sft step {step}/{total} | loss {loss_acc:.4f} | lr {lr:.2e} | {time.time() - t0:.0f}s")
            if step % 500 == 0 or step == total:
                save_checkpoint(os.path.join(args.out, "latest"), model, meta, step, "sft")
    print(f"SFT terminé → {args.out}/latest (perplexité réponses ≈ {math.exp(loss_acc):.2f})")


if __name__ == "__main__":
    main()
