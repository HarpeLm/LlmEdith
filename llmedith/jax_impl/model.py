"""Modèle LlmEdith en JAX pur, sans Flax : les paramètres sont un simple dict.

Mathématiquement identique à la version PyTorch (le test de parité le vérifie).
Les couches sont empilées ([L, ...]) et parcourues avec `lax.scan`. Le graphe
XLA est ainsi compilé une seule fois pour une couche, quel que soit le nombre de couches.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from llmedith.config import ModelConfig


def init_params(cfg: ModelConfig, key) -> dict:
    d, h, kv, hd, f, L = cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.ffn_dim, cfg.n_layers
    std, out_std = cfg.init_std, cfg.init_std / (2 * L) ** 0.5
    ks = iter(jax.random.split(key, 16))
    normal = lambda shape, s: s * jax.random.normal(next(ks), shape, jnp.float32)
    params = {
        "embed": normal((cfg.vocab_size, d), std),
        "norm": jnp.ones((d,)),
        "layers": {
            "input_layernorm": jnp.ones((L, d)),
            "q_proj": normal((L, h * hd, d), std),
            "k_proj": normal((L, kv * hd, d), std),
            "v_proj": normal((L, kv * hd, d), std),
            "o_proj": normal((L, d, h * hd), out_std),
            "q_norm": jnp.ones((L, hd)),
            "k_norm": jnp.ones((L, hd)),
            "post_attention_layernorm": jnp.ones((L, d)),
            "gate_proj": normal((L, f, d), std),
            "up_proj": normal((L, f, d), std),
            "down_proj": normal((L, d, f), out_std),
        },
    }
    if not cfg.tie_embeddings:
        params["lm_head"] = normal((cfg.vocab_size, d), std)
    return params


def rms_norm(x, w, eps):
    xf = x.astype(jnp.float32)
    xf = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return w * xf.astype(x.dtype)


def rope_cos_sin(T, head_dim, theta, dtype):
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = jnp.outer(jnp.arange(T, dtype=jnp.float32), inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def apply_rope(x, cos, sin):
    # x : [B, T, N, hd] ; cos/sin : [T, hd]
    x1, x2 = jnp.split(x, 2, axis=-1)
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return x * cos[None, :, None, :] + rotated * sin[None, :, None, :]


def linear(x, w):
    return jnp.einsum("...i,oi->...o", x, w)


def block(cfg: ModelConfig, x, p, cos, sin):
    B, T, _ = x.shape
    h = rms_norm(x, p["input_layernorm"], cfg.norm_eps)
    q = linear(h, p["q_proj"]).reshape(B, T, cfg.n_heads, cfg.head_dim)
    k = linear(h, p["k_proj"]).reshape(B, T, cfg.n_kv_heads, cfg.head_dim)
    v = linear(h, p["v_proj"]).reshape(B, T, cfg.n_kv_heads, cfg.head_dim)
    q = apply_rope(rms_norm(q, p["q_norm"], cfg.norm_eps), cos, sin)
    k = apply_rope(rms_norm(k, p["k_norm"], cfg.norm_eps), cos, sin)
    y = jax.nn.dot_product_attention(q, k, v, is_causal=True)
    x = x + linear(y.reshape(B, T, -1), p["o_proj"])
    h = rms_norm(x, p["post_attention_layernorm"], cfg.norm_eps)
    return x + linear(jax.nn.silu(linear(h, p["gate_proj"])) * linear(h, p["up_proj"]), p["down_proj"])


def hidden_states(cfg: ModelConfig, params, idx, dtype=jnp.bfloat16, remat: bool = False):
    p = jax.tree.map(lambda a: a.astype(dtype), params)
    x = p["embed"][idx]
    cos, sin = rope_cos_sin(idx.shape[1], cfg.head_dim, cfg.rope_theta, dtype)
    fn = lambda carry, lp: (block(cfg, carry, lp, cos, sin), None)
    if remat:
        fn = jax.checkpoint(fn, prevent_cse=False)
    x, _ = jax.lax.scan(fn, x, p["layers"])
    return rms_norm(x, p["norm"], cfg.norm_eps)


def logits_fn(cfg: ModelConfig, params, idx, dtype=jnp.float32):
    h = hidden_states(cfg, params, idx, dtype)
    W = params.get("lm_head", params["embed"]).astype(dtype)
    return linear(h, W).astype(jnp.float32)


def loss_fn(cfg: ModelConfig, params, idx, targets, *, z_loss=0.0, chunk=4096, dtype=jnp.bfloat16,
            remat=True, mask=None):
    """Cross-entropy découpée (scan + remat) : les logits [N, V] ne sont jamais tous en mémoire."""
    h = hidden_states(cfg, params, idx, dtype, remat).reshape(-1, cfg.d_model)
    t = targets.reshape(-1)
    m = jnp.ones_like(t, jnp.float32) if mask is None else mask.reshape(-1).astype(jnp.float32)
    W = params.get("lm_head", params["embed"]).astype(dtype)
    N = h.shape[0]
    chunk = min(chunk, N)
    pad = (-N) % chunk
    if pad:
        h = jnp.pad(h, ((0, pad), (0, 0)))
        t = jnp.pad(t, (0, pad))
        m = jnp.pad(m, (0, pad))
    hc, tc, mc = (a.reshape(-1, chunk, *a.shape[1:]) for a in (h, t, m))

    @jax.checkpoint
    def body(carry, xs):
        hh, tt, mm = xs
        logits = linear(hh, W).astype(jnp.float32)
        lse = jax.nn.logsumexp(logits, axis=-1)
        tgt = jnp.take_along_axis(logits, tt[:, None], axis=-1)[:, 0]
        ce, z = carry
        return (ce + jnp.sum((lse - tgt) * mm), z + jnp.sum(lse * lse * mm)), None

    (ce_sum, z_sum), _ = jax.lax.scan(body, (jnp.float32(0), jnp.float32(0)), (hc, tc, mc))
    denom = jnp.maximum(m.sum(), 1.0)
    ce = ce_sum / denom
    return ce + z_loss * z_sum / denom, ce
