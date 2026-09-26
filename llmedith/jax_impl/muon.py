"""Muon et AdamW pour JAX/Optax. C'est le même algorithme que llmedith/torch_impl/muon.py.

Les matrices des couches sont empilées ([L, out, in]). On applique donc Newton-Schulz
couche par couche avec vmap.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax


def newton_schulz(G, steps: int = 5, eps: float = 1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(jnp.bfloat16)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (jnp.linalg.norm(X.astype(jnp.float32)).astype(X.dtype) + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class MuonState(NamedTuple):
    count: jnp.ndarray
    mu: optax.Updates


def muon(learning_rate, momentum=0.95, ns_steps=5, weight_decay=0.1, nesterov=True):
    def init_fn(params):
        return MuonState(jnp.zeros([], jnp.int32), jax.tree.map(jnp.zeros_like, params))

    def update_fn(grads, state, params):
        mu = jax.tree.map(lambda m, g: momentum * m + (1 - momentum) * g, state.mu, grads)
        g_eff = jax.tree.map(lambda m, g: (1 - momentum) * g + momentum * m, mu, grads) if nesterov else mu
        lr = learning_rate(state.count) if callable(learning_rate) else learning_rate

        def one(g, p):
            ns = lambda x: newton_schulz(x, ns_steps)
            o = jax.vmap(ns)(g) if g.ndim == 3 else ns(g)
            scale = 0.2 * max(g.shape[-2], g.shape[-1]) ** 0.5
            return -lr * (scale * o.astype(p.dtype) + weight_decay * p)

        updates = jax.tree.map(one, g_eff, params)
        return updates, MuonState(state.count + 1, mu)

    return optax.GradientTransformation(init_fn, update_fn)


def param_labels(params: dict) -> dict:
    """Muon pour les matrices des couches, AdamW (avec ou sans weight decay) pour le reste."""
    def label(path, x):
        name = jax.tree_util.keystr(path)
        if "layers" in name and x.ndim == 3:
            return "muon"
        if x.ndim >= 2:            # embed / lm_head
            return "adam_decay"
        return "adam_no_decay"     # normes
    return jax.tree_util.tree_map_with_path(label, params)


def build_optimizer(tc, params):
    """LR unitaire : la boucle d'entraînement multiplie les updates par le LR du planning WSD.
    Le LR ne dépend ainsi que du step global, même après une reprise depuis un checkpoint PyTorch."""
    schedule = 1.0
    b1, b2 = tc.adam_betas
    adam_decay = optax.adamw(schedule, b1=b1, b2=b2, eps=tc.adam_eps, weight_decay=tc.weight_decay)
    adam_nd = optax.adamw(schedule, b1=b1, b2=b2, eps=tc.adam_eps, weight_decay=0.0)
    if tc.optimizer == "muon":
        mu = muon(schedule, tc.muon_momentum, tc.muon_ns_steps, tc.weight_decay)
    else:
        mu = adam_decay
    tx = optax.multi_transform({"muon": mu, "adam_decay": adam_decay, "adam_no_decay": adam_nd},
                               param_labels(params))
    return optax.chain(optax.clip_by_global_norm(tc.grad_clip), tx)
