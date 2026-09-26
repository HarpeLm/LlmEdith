"""Configuration partagée entre PyTorch et JAX.

Un seul fichier YAML décrit le modèle, l'entraînement et les données,
pour que les deux implémentations entraînent exactement le même réseau.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4          # GQA : n_heads doit être un multiple de n_kv_heads
    head_dim: int = 64
    ffn_dim: int = 3072          # dimension cachée SwiGLU
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    init_std: float = 0.02

    def __post_init__(self):
        assert self.n_heads % self.n_kv_heads == 0, "n_heads doit être un multiple de n_kv_heads"

    def num_params(self) -> int:
        d, h, kv, hd, f = self.d_model, self.n_heads, self.n_kv_heads, self.head_dim, self.ffn_dim
        attn = d * h * hd * 2 + d * kv * hd * 2 + 2 * hd       # q,o + k,v + q_norm,k_norm
        mlp = 3 * d * f
        layer = attn + mlp + 2 * d                              # + 2 RMSNorm
        emb = self.vocab_size * d * (1 if self.tie_embeddings else 2)
        return emb + self.n_layers * layer + d

    def flops_per_token(self, seq_len: int) -> float:
        """FLOPs d'entraînement par token (6N + attention), pour calculer le MFU."""
        n_non_emb = self.num_params() - self.vocab_size * self.d_model
        attn = 12 * self.n_layers * self.n_heads * self.head_dim * seq_len
        return 6 * (n_non_emb + self.vocab_size * self.d_model) + attn


@dataclass
class TrainConfig:
    seq_len: int = 2048
    global_batch_seqs: int = 256       # séquences par step optimiseur (≈ 0,5M tokens à 2048)
    micro_batch_seqs: int = 8          # séquences par passe avant (par device)
    total_tokens: float = 20e9
    # Planning WSD
    lr: float = 2e-3
    min_lr_frac: float = 0.0
    warmup_steps: int = 1000
    decay_frac: float = 0.2            # fraction finale de la décroissance (cooldown 1-sqrt)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    z_loss: float = 1e-4
    # Optimiseurs
    optimizer: str = "muon"            # "muon" ou "adamw"
    adam_betas: tuple = (0.9, 0.95)
    adam_eps: float = 1e-8
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    # Divers
    seed: int = 1337
    log_every: int = 10
    eval_every: int = 500
    eval_tokens: float = 20e6
    ckpt_every_minutes: float = 30.0
    loss_chunk: int = 4096             # tokens par morceau pour la cross-entropy découpée
    remat: bool = True                 # gradient checkpointing par couche

    @property
    def tokens_per_step(self) -> int:
        return self.global_batch_seqs * self.seq_len

    @property
    def total_steps(self) -> int:
        return int(self.total_tokens // self.tokens_per_step)


@dataclass
class DataConfig:
    train_pattern: str = "shards/train_*.bin"
    val_pattern: str = "shards/val_*.bin"
    anneal_pattern: str = ""           # si défini, utilisé pendant la phase de décroissance
    hub_repo: str = ""                 # dépôt dataset HF d'où télécharger les shards
    tokenizer_path: str = "tokenizer/llmedith-32k/tokenizer.json"


@dataclass
class Config:
    name: str = "llmedith"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        for ov in overrides or []:
            key, val = ov.split("=", 1)
            if "." in key:
                section, name = key.split(".", 1)
                raw.setdefault(section, {})[name] = yaml.safe_load(val)
            else:  # clé de premier niveau, ex. name=tiny_muon
                raw[key] = yaml.safe_load(val)
        cfg = cls(
            name=raw.get("name", Path(path).stem),
            model=_build(ModelConfig, raw.get("model", {})),
            train=_build(TrainConfig, raw.get("train", {})),
            data=_build(DataConfig, raw.get("data", {})),
        )
        if isinstance(cfg.train.adam_betas, list):
            cfg.train.adam_betas = tuple(cfg.train.adam_betas)
        return cfg

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _build(klass, values: dict):
    """PyYAML lit « 5.0e8 » comme une chaîne : on convertit selon le type par défaut du champ."""
    defaults = {f.name: f.default for f in dataclasses.fields(klass)}
    out = {}
    for k, v in values.items():
        if k not in defaults:
            raise KeyError(f"Champ inconnu {klass.__name__}.{k}")
        d = defaults[k]
        if isinstance(d, bool):
            v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
        elif isinstance(d, (int, float)) and isinstance(v, str):
            v = type(d)(float(v))
        out[k] = v
    return klass(**out)
