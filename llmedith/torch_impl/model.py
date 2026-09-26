"""Modèle LlmEdith en PyTorch.

Transformer décodeur moderne :
- pre-norm RMSNorm, QK-norm, RoPE, GQA, SwiGLU ;
- pas de biais, embeddings liés.

Les noms des paramètres sont identiques à ceux de `Qwen3ForCausalLM` (HF Transformers).
L'export vers HF est donc un simple renommage de fichier.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from llmedith.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rope_cos_sin(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)          # convention « rotate_half » (Llama/Qwen)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x, cos, sin):
    # x : [B, H, T, hd]
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, h, kv, hd = cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(d, h * hd, bias=False)
        self.k_proj = nn.Linear(d, kv * hd, bias=False)
        self.v_proj = nn.Linear(d, kv * hd, bias=False)
        self.o_proj = nn.Linear(h * hd, d, bias=False)
        self.q_norm = RMSNorm(hd, cfg.norm_eps)
        self.k_norm = RMSNorm(hd, cfg.norm_eps)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        c = self.cfg
        q = self.q_norm(self.q_proj(x).view(B, T, c.n_heads, c.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, c.n_kv_heads, c.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, c.n_heads * c.head_dim))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)


class LlmEdith(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = Backbone(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.gradient_checkpointing = False
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        std = self.cfg.init_std
        out_std = std / math.sqrt(2 * self.cfg.n_layers)   # projections de sortie réduites (GPT-2 / OLMo)
        for name, p in self.named_parameters():
            if p.ndim == 1:
                p.fill_(1.0)
            elif name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, 0.0, out_std)
            else:
                nn.init.normal_(p, 0.0, std)

    def hidden_states(self, idx):
        B, T = idx.shape
        x = self.model.embed_tokens(idx)
        cos, sin = rope_cos_sin(T, self.cfg.head_dim, self.cfg.rope_theta, idx.device, x.dtype)
        for layer in self.model.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, cos, sin, use_reentrant=False)
            else:
                x = layer(x, cos, sin)
        return self.model.norm(x)

    def forward(self, idx):
        """Renvoie les logits [B, T, V]. Utilisé pour l'inférence et les tests."""
        return self.lm_head(self.hidden_states(idx)).float()

    def loss(self, idx, targets, z_loss: float = 0.0, chunk: int = 4096, mask=None):
        """Cross-entropy découpée en morceaux : on ne matérialise jamais les logits [B*T, V].

        `mask` (optionnel, [B, T]) : 1 pour les tokens comptés dans la loss (sert au SFT).
        Renvoie (loss totale, cross-entropy seule).
        """
        h = self.hidden_states(idx).flatten(0, 1)
        t = targets.flatten()
        m = torch.ones_like(t, dtype=torch.float32) if mask is None else mask.flatten().float()
        W = self.lm_head.weight
        ce_sum = h.new_zeros((), dtype=torch.float32)
        z_sum = h.new_zeros((), dtype=torch.float32)
        for s in range(0, h.shape[0], chunk):
            if self.training:
                ce, z = checkpoint(_chunk_ce, h[s:s + chunk], W, t[s:s + chunk], m[s:s + chunk], use_reentrant=False)
            else:
                ce, z = _chunk_ce(h[s:s + chunk], W, t[s:s + chunk], m[s:s + chunk])
            ce_sum, z_sum = ce_sum + ce, z_sum + z
        denom = m.sum().clamp_min(1.0)
        ce = ce_sum / denom
        return ce + z_loss * z_sum / denom, ce

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _chunk_ce(h, W, t, m):
    logits = (h @ W.t()).float()
    lse = torch.logsumexp(logits, dim=-1)
    tgt = logits.gather(-1, t[:, None]).squeeze(-1)
    return ((lse - tgt) * m).sum(), (lse.pow(2) * m).sum()


@torch.no_grad()
def generate(model: LlmEdith, idx, max_new_tokens: int, temperature: float = 0.8, top_p: float = 0.95,
             eos_id: int | None = None):
    """Génération simple sans cache KV (pour l'inférence rapide, passer par l'export HF)."""
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.cfg.max_seq_len:])[:, -1]
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            sp, si = probs.sort(descending=True)
            keep = sp.cumsum(-1) - sp < top_p
            sp = sp * keep
            nxt = si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1))
        idx = torch.cat([idx, nxt], dim=1)
        if eos_id is not None and (nxt == eos_id).all():
            break
    return idx
