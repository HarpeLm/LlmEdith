"""Pré-entraînement JAX sur TPU (Kaggle TPU VM v5e-8 / v3-8). Fonctionne aussi sur GPU et CPU.

Parallélisme de données : les paramètres sont répliqués sur les 8 cœurs, le batch est découpé
entre eux. Pour un modèle de ~300M, cela tient largement en mémoire HBM.

    python -m llmedith.jax_impl.train --config configs/main.yaml --resume --max_minutes 530
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from safetensors.numpy import load_file, save_file

from llmedith import hub
from llmedith.config import Config
from llmedith.convert import hf_to_jax, jax_to_hf
from llmedith.data import ShardLoader, iter_val_batches
from llmedith.jax_impl.model import init_params, loss_fn
from llmedith.jax_impl.muon import build_optimizer
from llmedith.schedule import in_decay_phase, wsd_lr_factor

# Pic bf16 par device JAX (TFLOPs)
PEAK_TFLOPS = {"TPU v3": 61.5, "TPU v4": 275.0, "TPU v5 lite": 197.0, "TPU v5e": 197.0,
               "TPU v5p": 459.0, "TPU v6": 918.0}


def save_checkpoint(ckpt_dir, params, opt_state, step, loader, anneal_loader, cfg):
    os.makedirs(ckpt_dir, exist_ok=True)
    host_params = jax.device_get(params)
    save_file({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in jax_to_hf(host_params).items()},
              os.path.join(ckpt_dir, "model.safetensors"))
    leaves = [np.asarray(x) for x in jax.tree.leaves(jax.device_get(opt_state))]
    np.savez(os.path.join(ckpt_dir, "opt_state_jax.npz"), *leaves)
    meta = {"step": step, "loader": loader.state_dict(),
            "anneal_loader": anneal_loader.state_dict() if anneal_loader else None,
            "config": cfg.to_dict(), "framework": "jax"}
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init_from", default="")
    ap.add_argument("--hub_repo", default=os.environ.get("LLMEDITH_HUB_REPO", ""))
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--max_minutes", type=float, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config, args.override)
    mc, tc, dc = cfg.model, cfg.train, cfg.data
    dtype = jnp.dtype(args.dtype)
    devices = jax.devices()
    n_dev = len(devices)
    mesh = Mesh(np.array(devices), ("data",))
    replicated = NamedSharding(mesh, P())
    batch_sharding = NamedSharding(mesh, P(None, "data", None))
    run_dir = os.path.join(args.out_dir, cfg.name)
    ckpt_dir = os.path.join(run_dir, "latest")

    micro_global = tc.micro_batch_seqs * n_dev
    assert tc.global_batch_seqs % micro_global == 0, "global_batch_seqs doit être divisible par micro×devices"
    accum = tc.global_batch_seqs // micro_global
    total_steps = tc.total_steps

    params = init_params(mc, jax.random.PRNGKey(tc.seed))
    tx = build_optimizer(tc, params)
    opt_state = tx.init(params)
    loader = ShardLoader(dc.train_pattern, tc.global_batch_seqs, tc.seq_len, dc.hub_repo)
    anneal_loader = ShardLoader(dc.anneal_pattern, tc.global_batch_seqs, tc.seq_len, dc.hub_repo) if dc.anneal_pattern else None
    step = 0
    rewarm_from = None  # step où l'optimiseur a été réinitialisé : on refait un court warmup

    if args.resume and not os.path.exists(os.path.join(ckpt_dir, "meta.json")):
        hub.pull_folder(args.hub_repo, f"{cfg.name}/latest", args.out_dir)
    if args.resume and os.path.exists(os.path.join(ckpt_dir, "meta.json")):
        meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))
        params = hf_to_jax(load_file(os.path.join(ckpt_dir, "model.safetensors")), mc.n_layers, mc.tie_embeddings)
        opt_path = os.path.join(ckpt_dir, "opt_state_jax.npz")
        if meta.get("framework") == "jax" and os.path.exists(opt_path):
            saved = np.load(opt_path)
            leaves = [saved[f"arr_{i}"] for i in range(len(saved.files))]
            opt_state = jax.tree.unflatten(jax.tree.structure(opt_state), leaves)
        else:
            print("Checkpoint PyTorch : on ne reprend que les poids, l'état de l'optimiseur repart de zéro.")
            opt_state = tx.init(params)
            rewarm_from = meta["step"]
        step = meta["step"]
        loader.load_state_dict(meta["loader"])
        if anneal_loader and meta.get("anneal_loader"):
            anneal_loader.load_state_dict(meta["anneal_loader"])
        print(f"Reprise au step {step}")
    elif args.init_from:
        params = hf_to_jax(load_file(os.path.join(args.init_from, "model.safetensors")), mc.n_layers,
                           mc.tie_embeddings)
        opt_state = tx.init(params)

    params = jax.device_put(params, replicated)
    opt_state = jax.device_put(opt_state, replicated)

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(params, opt_state, bx, by, lr):
        def micro(carry, xy):
            g_acc, ce_acc = carry
            (_, ce), g = jax.value_and_grad(
                lambda p: loss_fn(mc, p, xy[0], xy[1], z_loss=tc.z_loss, chunk=tc.loss_chunk,
                                  dtype=dtype, remat=tc.remat), has_aux=True)(params)
            return (jax.tree.map(jnp.add, g_acc, g), ce_acc + ce), None

        zeros = jax.tree.map(jnp.zeros_like, params)
        (grads, ce), _ = jax.lax.scan(micro, (zeros, jnp.float32(0)), (bx, by))
        grads = jax.tree.map(lambda g: g / accum, grads)
        gnorm = optax.global_norm(grads)
        updates, opt_state = tx.update(grads, opt_state, params)
        updates = jax.tree.map(lambda u: lr * u, updates)
        return optax.apply_updates(params, updates), opt_state, ce / accum, gnorm

    @jax.jit
    def eval_step(params, x, y):
        return loss_fn(mc, params, x, y, chunk=tc.loss_chunk, dtype=dtype, remat=False)[1]

    def evaluate() -> float:
        eval_sharding = NamedSharding(mesh, P("data", None))
        losses = [eval_step(params, jax.device_put(x, eval_sharding), jax.device_put(y, eval_sharding))
                  for x, y in iter_val_batches(dc.val_pattern, micro_global, tc.seq_len, tc.eval_tokens, dc.hub_repo)]
        return float(np.mean(jax.device_get(losses)))

    kind = devices[0].device_kind
    peak = next((v for k, v in PEAK_TFLOPS.items() if kind.startswith(k)), None)
    flops_per_step = mc.flops_per_token(tc.seq_len) * tc.tokens_per_step
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"LlmEdith {cfg.name} (JAX) : {n_params / 1e6:.1f}M paramètres | {n_dev}× {kind} | accum={accum} | "
          f"{total_steps} steps × {tc.tokens_per_step} tokens")
    os.makedirs(run_dir, exist_ok=True)
    log_f = open(os.path.join(run_dir, "log.csv"), "a", newline="")
    log = csv.writer(log_f)
    if args.wandb:
        import wandb
        wandb.init(project="llmedith", name=cfg.name, config=cfg.to_dict(), resume="allow")

    t_start = t_ckpt = time.time()
    while step < total_steps:
        t0 = time.time()
        decay = in_decay_phase(step, total_steps, tc.decay_frac)
        src = anneal_loader if (decay and anneal_loader) else loader
        x, y = src.next_batch()
        bx = jax.device_put(x.reshape(accum, micro_global, tc.seq_len), batch_sharding)
        by = jax.device_put(y.reshape(accum, micro_global, tc.seq_len), batch_sharding)
        lr = tc.lr * wsd_lr_factor(step, total_steps, tc.warmup_steps, tc.decay_frac, tc.min_lr_frac)
        if rewarm_from is not None:
            lr *= min(1.0, (step - rewarm_from + 1) / max(1, tc.warmup_steps // 4))
        params, opt_state, ce, gnorm = train_step(params, opt_state, bx, by, jnp.float32(lr))
        step += 1

        if step % tc.log_every == 0 or step == 1:
            ce, gnorm = float(ce), float(gnorm)          # synchronise l'hôte
            dt = (time.time() - t0)
            tps = tc.tokens_per_step / dt
            mfu = flops_per_step / dt / (peak * 1e12 * n_dev) if peak else float("nan")
            print(f"step {step}/{total_steps} | loss {ce:.4f} | lr {lr:.2e} | gnorm {gnorm:.2f} | "
                  f"{tps / 1e3:.1f}k tok/s | MFU {mfu:.1%}", flush=True)
            log.writerow([step, "train", round(ce, 5), lr, round(gnorm, 4), int(tps)])
            log_f.flush()
            if args.wandb:
                wandb.log({"train/loss": ce, "lr": lr, "gnorm": gnorm, "tok_s": tps, "mfu": mfu}, step=step)

        if step % tc.eval_every == 0 or step == total_steps:
            val = evaluate()
            print(f"== step {step} | val loss {val:.4f} | ppl {math.exp(val):.2f}", flush=True)
            log.writerow([step, "val", round(val, 5), lr, "", ""])
            log_f.flush()
            if args.wandb:
                wandb.log({"val/loss": val}, step=step)

        elapsed_min = (time.time() - t_start) / 60
        stop = (args.max_steps and step >= args.max_steps) or (args.max_minutes and elapsed_min >= args.max_minutes)
        if (time.time() - t_ckpt) / 60 >= tc.ckpt_every_minutes or stop or step == total_steps:
            save_checkpoint(ckpt_dir, params, opt_state, step, loader, anneal_loader, cfg)
            hub.push_folder(args.hub_repo, ckpt_dir, f"{cfg.name}/latest")
            t_ckpt = time.time()
            print(f"Checkpoint sauvegardé (step {step})", flush=True)
        if stop:
            break

    hub.wait_uploads()
    log_f.close()


if __name__ == "__main__":
    main()
