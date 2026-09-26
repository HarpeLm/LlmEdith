"""DPO (Direct Preference Optimization, Rafailov et al. 2023) sur UltraFeedback, après le SFT.

    python -m posttrain.dpo --base runs/sft/latest --out runs/dpo
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import os
import time

import torch
import torch.nn.functional as F

from posttrain.common import ChatFormatter, load_checkpoint, pick_device_dtype, save_checkpoint


def seq_logprobs(model, ids, mask):
    """Somme des log-probabilités des tokens de réponse, pour chaque séquence ([B])."""
    x, y, m = ids[:, :-1], ids[:, 1:], mask[:, 1:]
    logits = model(x)
    lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, y[..., None]).squeeze(-1)
    return (lp * m).sum(-1)


def collate(pairs, pad):
    L = max(len(ids) for ids, _ in pairs)
    ids = torch.full((len(pairs), L), pad, dtype=torch.long)
    mask = torch.zeros((len(pairs), L))
    for i, (a, b) in enumerate(pairs):
        ids[i, :len(a)], mask[i, :len(b)] = torch.tensor(a), torch.tensor(b, dtype=torch.float32)
    return ids, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="checkpoint après SFT")
    ap.add_argument("--out", default="runs/dpo")
    ap.add_argument("--tokenizer", default="tokenizer/llmedith-32k/tokenizer.json")
    ap.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    ap.add_argument("--split", default="train_prefs")
    ap.add_argument("--max_examples", type=int, default=60_000)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro_batch", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-6)
    args = ap.parse_args()

    device, dtype = pick_device_dtype()
    policy, meta = load_checkpoint(args.base, device)
    ref = copy.deepcopy(policy).eval().requires_grad_(False)
    fmt = ChatFormatter(args.tokenizer)
    autocast = torch.autocast(device.type, dtype=dtype) if dtype != torch.float32 else contextlib.nullcontext()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    ds = ds.shuffle(seed=0, buffer_size=20_000).take(args.max_examples)
    data = []
    for ex in ds:
        c, r = fmt.encode(ex["chosen"]), fmt.encode(ex["rejected"])
        if len(c[0]) <= args.max_len and len(r[0]) <= args.max_len:
            data.append((c, r))
    print(f"{len(data)} paires de préférences")

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    total = len(data) // args.batch
    accum = args.batch // args.micro_batch
    policy.train()
    t0 = time.time()
    for step in range(total):
        lr = args.lr * min(1.0, (step + 1) / max(1, total // 10)) * (1 - step / total)
        for g in opt.param_groups:
            g["lr"] = lr
        stats = {"loss": 0.0, "acc": 0.0, "margin": 0.0}
        for i in range(accum):
            chunk = data[step * args.batch + i * args.micro_batch: step * args.batch + (i + 1) * args.micro_batch]
            ids, mask = collate([c for c, _ in chunk] + [r for _, r in chunk], fmt.pad)
            ids, mask = ids.to(device), mask.to(device)
            n = len(chunk)
            with autocast:
                pl = seq_logprobs(policy, ids, mask)
                with torch.no_grad():
                    rl = seq_logprobs(ref, ids, mask)
            logits = args.beta * ((pl[:n] - rl[:n]) - (pl[n:] - rl[n:]))
            loss = -F.logsigmoid(logits).mean()
            (loss / accum).backward()
            stats["loss"] += loss.item() / accum
            stats["acc"] += (logits > 0).float().mean().item() / accum
            stats["margin"] += logits.mean().item() / accum
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % 10 == 0 or step == 0:
            print(f"dpo step {step + 1}/{total} | loss {stats['loss']:.4f} | acc préférences {stats['acc']:.2f} | "
                  f"marge {stats['margin']:.3f} | {time.time() - t0:.0f}s")
        if (step + 1) % 200 == 0 or step + 1 == total:
            save_checkpoint(os.path.join(args.out, "latest"), policy, meta, step + 1, "dpo")


if __name__ == "__main__":
    main()
