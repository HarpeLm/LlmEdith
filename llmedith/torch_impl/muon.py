"""Optimiseur Muon (MomentUm Orthogonalized by Newton-Schulz).

- K. Jordan et al. 2024 (modded-nanogpt) ;
- mise à l'échelle « Moonlight » (Liu et al. 2025) : update × 0,2·sqrt(max(m, n)),
  ce qui donne un RMS comparable à AdamW. On peut alors partager le même LR
  et le même weight decay.

Muon s'applique uniquement aux matrices 2D internes au Transformer.
Les embeddings, le lm_head et les normes restent sous AdamW.
"""
import torch


def native_bf16(device: torch.device) -> bool:
    """bf16 matériel : GPU Ampere+ (RTX 30/40, A100, L4…). Les T4 et P100 l'émulent très lentement."""
    if device.type == "cuda":
        return torch.cuda.get_device_capability(device)[0] >= 8
    return device.type in ("cpu", "mps")


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Orthogonalisation approchée de G (itération quintique, en bf16 si le matériel le permet)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16 if native_bf16(G.device) else torch.float32)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-3, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.1):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                update = newton_schulz(g, group["ns_steps"]).to(p.dtype)
                scale = 0.2 * max(p.size(0), p.size(1)) ** 0.5
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"] * scale)


def split_params(model):
    """Sépare les paramètres : (matrices pour Muon, reste avec weight decay, reste sans weight decay)."""
    muon, adam_decay, adam_no_decay = [], [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim == 2 and "embed_tokens" not in name and "lm_head" not in name:
            muon.append(p)
        elif p.ndim == 2:
            adam_decay.append(p)
        else:
            adam_no_decay.append(p)
    return muon, adam_decay, adam_no_decay


def build_optimizers(model, tc) -> list[torch.optim.Optimizer]:
    muon, adam_decay, adam_no_decay = split_params(model)
    fused = torch.cuda.is_available()
    adam_groups = [
        {"params": adam_decay, "weight_decay": tc.weight_decay},
        {"params": adam_no_decay, "weight_decay": 0.0},
    ]
    if tc.optimizer == "muon":
        adam = torch.optim.AdamW(adam_groups, lr=tc.lr, betas=tc.adam_betas, eps=tc.adam_eps, fused=fused)
        return [Muon(muon, lr=tc.lr, momentum=tc.muon_momentum, ns_steps=tc.muon_ns_steps,
                     weight_decay=tc.weight_decay), adam]
    adam_groups[0]["params"] = adam_decay + muon
    return [torch.optim.AdamW(adam_groups, lr=tc.lr, betas=tc.adam_betas, eps=tc.adam_eps, fused=fused)]

