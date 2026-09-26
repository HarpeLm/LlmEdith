"""Télécharge (en streaming), mélange et tokenise le corpus anglais en shards uint16.

À lancer sur ton PC : le CPU tokenise pendant que le GPU reste libre. La préparation est reprenable
(l'état de chaque flux est sauvegardé), donc tu peux l'interrompre et la relancer.

    python -m data.prepare --mix train  --total_tokens 25e9 --out shards
    python -m data.prepare --mix anneal --total_tokens 3e9  --out shards --no_val
    python -m data.prepare --upload MonPseudo/llmedith-data --out shards   # envoi sur le HF Hub (privé)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml
from tokenizers import Tokenizer

from llmedith.data import write_shard

DOCS_PER_BATCH = 128


class Source:
    def __init__(self, spec: dict, state: dict | None):
        from datasets import load_dataset
        self.spec = spec
        self.name = spec["name"]
        self.weight = float(spec["weight"])
        ds = load_dataset(spec["path"], spec.get("config"), split=spec.get("split", "train"), streaming=True)
        if state and state.get("ds") is not None:
            ds.load_state_dict(state["ds"])
        self.ds = ds
        self.it = iter(ds)
        self.tokens = state["tokens"] if state else 0

    def next_texts(self, n: int) -> list[str]:
        out, field, min_score = [], self.spec.get("field", "text"), self.spec.get("min_score")
        while len(out) < n:
            try:
                ex = next(self.it)
            except StopIteration:
                print(f"[{self.name}] source épuisée, on recommence")
                self.it = iter(self.ds)
                continue
            if min_score is not None and ex.get("int_score", ex.get("score", min_score)) < min_score:
                continue
            text = ex.get(field)
            if text and len(text) > 50:
                out.append(text)
        return out

    def state(self) -> dict:
        return {"ds": self.ds.state_dict(), "tokens": self.tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix_config", default="configs/data_mix.yaml")
    ap.add_argument("--mix", default="train", choices=["train", "anneal"])
    ap.add_argument("--total_tokens", type=float, default=25e9)
    ap.add_argument("--out", default="shards")
    ap.add_argument("--no_val", action="store_true")
    ap.add_argument("--upload", default="", help="dépôt dataset HF (privé) où envoyer les shards")
    args = ap.parse_args()

    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.upload, repo_type="dataset", private=True, exist_ok=True)
        api.upload_large_folder(repo_id=args.upload, repo_type="dataset", folder_path=args.out,
                                allow_patterns=["*.bin", "*.json"])
        return

    mix = yaml.safe_load(open(args.mix_config))
    tok = Tokenizer.from_file(mix["tokenizer"])
    eot = tok.token_to_id("<|endoftext|>")
    assert tok.get_vocab_size() <= 65536
    shard_tokens, val_tokens = int(float(mix["shard_tokens"])), int(float(mix["val_tokens"]))
    os.makedirs(args.out, exist_ok=True)
    progress_path = os.path.join(args.out, f"progress_{args.mix}.json")
    progress = json.load(open(progress_path)) if os.path.exists(progress_path) else {}

    sources = [Source(s, progress.get("sources", {}).get(s["name"])) for s in mix[args.mix]]
    total_w = sum(s.weight for s in sources)
    written = progress.get("written", 0)
    shard_idx = progress.get("shard_idx", 0)
    need_val = not args.no_val and args.mix == "train" and not progress.get("val_done", False)
    buf, buf_len = [], 0

    def flush(split: str, idx: int, arr: np.ndarray):
        path = os.path.join(args.out, f"{split}_{idx:05d}.bin")
        write_shard(path, arr)
        print(f"-> {path} ({len(arr) / 1e6:.0f}M tokens) | total {written / 1e9:.2f}B | " +
              " ".join(f"{s.name}={s.tokens / max(1, sum(x.tokens for x in sources)):.1%}" for s in sources),
              flush=True)

    def save_progress():
        json.dump({"written": written, "shard_idx": shard_idx, "val_done": not need_val,
                   "sources": {s.name: s.state() for s in sources}}, open(progress_path, "w"))

    split = "anneal" if args.mix == "anneal" else "train"
    while written < args.total_tokens:
        # on tire la source la plus en retard sur son poids cible
        src = min(sources, key=lambda s: s.tokens / (s.weight / total_w))
        encs = tok.encode_batch(src.next_texts(DOCS_PER_BATCH))
        for e in encs:
            ids = e.ids + [eot]
            buf.append(np.asarray(ids, dtype=np.uint16))
            buf_len += len(ids)
            src.tokens += len(ids)
        target = val_tokens if need_val else shard_tokens
        if buf_len >= target:
            arr = np.concatenate(buf)
            out, rest = arr[:target], arr[target:]
            buf, buf_len = [rest], len(rest)
            if need_val:
                flush("val", 0, out)
                need_val = False
            else:
                written += len(out)
                flush(split, shard_idx, out)
                shard_idx += 1
            save_progress()
    print("Terminé.", flush=True)
    os._exit(0)  # les threads de streaming de datasets/pyarrow peuvent bloquer la sortie normale


if __name__ == "__main__":
    main()
