"""Data loader déterministe et reprenable, commun à PyTorch et JAX.

Les shards sont des fichiers binaires bruts de tokens uint16 (vocabulaire ≤ 65 536).
Les documents y sont concaténés et séparés par le token <|endoftext|>.
L'état du loader tient en deux entiers (shard, position). Après une coupure
Colab ou Kaggle, on reprend donc exactement au même token.
"""
from __future__ import annotations

import fnmatch
import glob
import os
import threading

import numpy as np


def list_shards(pattern: str, hub_repo: str = "") -> list[str]:
    """Chemins locaux des shards. Avec `hub_repo`, la liste vient du dépôt dataset HF
    (fichiers à la racine du dépôt, téléchargés dans le dossier du pattern au moment voulu)."""
    if hub_repo:
        from huggingface_hub import list_repo_files
        base, local_dir = os.path.basename(pattern), os.path.dirname(pattern) or "."
        remote = [f for f in list_repo_files(hub_repo, repo_type="dataset") if fnmatch.fnmatch(f, base)]
        files = [os.path.join(local_dir, f) for f in sorted(remote)]
    else:
        files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Aucun shard trouvé pour {pattern!r} {'sur ' + hub_repo if hub_repo else ''}")
    return files


class ShardLoader:
    """Renvoie des batches (x, y) de forme [B, T]. y est x décalé d'un token.

    Avec `hub_repo` : chaque shard est téléchargé à la demande, le suivant est préchargé en
    arrière-plan et les anciens sont supprimés (utile sur Kaggle/Colab où le disque est limité).
    """

    def __init__(self, pattern: str, batch_seqs: int, seq_len: int, hub_repo: str = "", keep_local: int = 3):
        self.hub_repo = hub_repo
        self.files = list_shards(pattern, hub_repo)
        self.B, self.T = batch_seqs, seq_len
        self.keep_local = keep_local
        self.shard_idx = 0
        self.pos = 0
        self.epoch = 0
        self._tokens = None
        self._prefetch: threading.Thread | None = None  # shard chargé paresseusement au premier batch

    def _fetch(self, idx: int) -> str:
        path = self.files[idx]
        if self.hub_repo and not os.path.exists(path):
            from huggingface_hub import hf_hub_download
            hf_hub_download(self.hub_repo, os.path.basename(path), repo_type="dataset",
                            local_dir=os.path.dirname(path) or ".")
        return path

    def _load(self):
        if self._prefetch is not None:
            self._prefetch.join()
        self._tokens = np.memmap(self._fetch(self.shard_idx), dtype=np.uint16, mode="r")
        if self.hub_repo and len(self.files) > 1:
            nxt = (self.shard_idx + 1) % len(self.files)
            self._prefetch = threading.Thread(target=self._fetch, args=(nxt,), daemon=True)
            self._prefetch.start()
            if len(self.files) > self.keep_local:
                old = self.files[(self.shard_idx - self.keep_local + 1) % len(self.files)]
                if os.path.exists(old) and old != self.files[nxt]:
                    os.remove(old)

    def state_dict(self) -> dict:
        return {"shard_idx": self.shard_idx, "pos": self.pos, "epoch": self.epoch}

    def load_state_dict(self, state: dict) -> None:
        self.shard_idx = int(state["shard_idx"]) % len(self.files)
        self.pos, self.epoch = int(state["pos"]), int(state.get("epoch", 0))
        self._tokens = None

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        need = self.B * self.T + 1
        if self._tokens is None:
            self._load()
        if self.pos + need > len(self._tokens):
            self.shard_idx += 1
            if self.shard_idx >= len(self.files):
                self.shard_idx, self.epoch = 0, self.epoch + 1
            self.pos = 0
            self._load()
        buf = np.asarray(self._tokens[self.pos:self.pos + need], dtype=np.int32)
        self.pos += self.B * self.T
        x = buf[:-1].reshape(self.B, self.T)
        y = buf[1:].reshape(self.B, self.T)
        return x, y


def iter_val_batches(pattern: str, batch_seqs: int, seq_len: int, max_tokens: float, hub_repo: str = ""):
    """Batches de validation fixes : toujours les mêmes tokens, pour comparer les runs."""
    loader = ShardLoader(pattern, batch_seqs, seq_len, hub_repo, keep_local=10**9)
    n = max(1, int(max_tokens // (batch_seqs * seq_len)))
    for _ in range(n):
        yield loader.next_batch()


def write_shard(path: str, tokens: np.ndarray) -> None:
    assert tokens.dtype == np.uint16
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tokens.tofile(path)
