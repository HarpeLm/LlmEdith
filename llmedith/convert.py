"""Conversion des poids entre PyTorch (noms Qwen3/HF) et JAX (paramètres empilés par couche).

On n'utilise que numpy ici : le module sert des deux côtés, sans dépendre de torch ni de jax.
Le format canonique des checkpoints est un `model.safetensors` avec les noms HF/Qwen3.
Un checkpoint écrit par JAX se relit donc en PyTorch, et inversement.
"""
from __future__ import annotations

import numpy as np

# nom JAX (empilé)  ->  suffixe HF dans model.layers.{i}.
LAYER_KEYS = {
    "input_layernorm": "input_layernorm.weight",
    "q_proj": "self_attn.q_proj.weight",
    "k_proj": "self_attn.k_proj.weight",
    "v_proj": "self_attn.v_proj.weight",
    "o_proj": "self_attn.o_proj.weight",
    "q_norm": "self_attn.q_norm.weight",
    "k_norm": "self_attn.k_norm.weight",
    "post_attention_layernorm": "post_attention_layernorm.weight",
    "gate_proj": "mlp.gate_proj.weight",
    "up_proj": "mlp.up_proj.weight",
    "down_proj": "mlp.down_proj.weight",
}


def hf_to_jax(sd: dict[str, np.ndarray], n_layers: int, tie: bool = True) -> dict:
    params = {
        "embed": sd["model.embed_tokens.weight"],
        "norm": sd["model.norm.weight"],
        "layers": {k: np.stack([sd[f"model.layers.{i}.{v}"] for i in range(n_layers)])
                   for k, v in LAYER_KEYS.items()},
    }
    if not tie:
        params["lm_head"] = sd["lm_head.weight"]
    return params


def jax_to_hf(params: dict) -> dict[str, np.ndarray]:
    p = {k: np.asarray(v) for k, v in _flatten_top(params).items()}
    sd = {"model.embed_tokens.weight": p["embed"], "model.norm.weight": p["norm"]}
    n_layers = p["layers/q_proj"].shape[0]
    for k, v in LAYER_KEYS.items():
        stacked = p[f"layers/{k}"]
        for i in range(n_layers):
            sd[f"model.layers.{i}.{v}"] = np.ascontiguousarray(stacked[i])
    if "lm_head" in p:
        sd["lm_head.weight"] = p["lm_head"]
    return sd


def _flatten_top(params: dict) -> dict:
    out = {k: v for k, v in params.items() if k != "layers"}
    out.update({f"layers/{k}": v for k, v in params["layers"].items()})
    return out
