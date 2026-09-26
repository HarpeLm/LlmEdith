"""Tests de cohérence : PyTorch ↔ Qwen3 (HF) ↔ JAX, masque causal, sur-apprentissage, reprise."""
import numpy as np
import pytest
import torch

from llmedith.config import ModelConfig, TrainConfig
from llmedith.torch_impl.model import LlmEdith

CFG = ModelConfig(vocab_size=512, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16, ffn_dim=128,
                  max_seq_len=64)


def make_model(seed=0):
    torch.manual_seed(seed)
    m = LlmEdith(CFG)
    with torch.no_grad():  # normes ≠ 1 pour que le test vérifie vraiment leur usage
        for n, p in m.named_parameters():
            if p.ndim == 1:
                p.uniform_(0.5, 1.5)
    return m.eval()


def tokens(B=2, T=32, seed=0):
    return torch.from_numpy(np.random.RandomState(seed).randint(0, CFG.vocab_size, (B, T)))


def test_shapes_and_params():
    m = make_model()
    assert m(tokens()).shape == (2, 32, CFG.vocab_size)
    assert m.num_params() == CFG.num_params()


def test_causal():
    m = make_model()
    x = tokens()
    x2 = x.clone()
    x2[:, 20:] = (x2[:, 20:] + 1) % CFG.vocab_size
    with torch.no_grad():
        a, b = m(x), m(x2)
    assert torch.allclose(a[:, :20], b[:, :20], atol=1e-5)
    assert not torch.allclose(a[:, 20:], b[:, 20:], atol=1e-3)


def test_chunked_loss_matches_full():
    m = make_model()
    x, y = tokens(), tokens(seed=1)
    with torch.no_grad():
        ref = torch.nn.functional.cross_entropy(m(x).flatten(0, 1), y.flatten())
        _, ce = m.loss(x, y, chunk=7)
    assert torch.allclose(ref, ce, atol=1e-5)


def test_matches_hf_qwen3():
    transformers = pytest.importorskip("transformers")
    from convert.to_hf import hf_config
    m = make_model()
    hf = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(**hf_config(CFG))).eval()
    sd = {k: v for k, v in m.state_dict().items()}
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    assert not unexpected and all("lm_head" in k for k in missing), (missing, unexpected)
    x = tokens()
    with torch.no_grad():
        np.testing.assert_allclose(m(x).numpy(), hf(x).logits.float().numpy(), atol=1e-4, rtol=1e-4)


def test_matches_jax():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from llmedith.convert import hf_to_jax, jax_to_hf
    from llmedith.jax_impl.model import logits_fn, loss_fn
    m = make_model()
    sd = {k: v.numpy() for k, v in m.state_dict().items() if k != "lm_head.weight"}
    params = hf_to_jax(sd, CFG.n_layers)
    x = tokens()
    with torch.no_grad():
        ref = m(x).numpy()
    out = np.asarray(logits_fn(CFG, params, jnp.asarray(x.numpy()), dtype=jnp.float32))
    np.testing.assert_allclose(ref, out, atol=1e-4, rtol=1e-4)
    # loss découpée identique
    y = tokens(seed=1)
    with torch.no_grad():
        _, ce_t = m.loss(x, y, chunk=16)
    _, ce_j = loss_fn(CFG, params, jnp.asarray(x.numpy()), jnp.asarray(y.numpy()), chunk=16, dtype=jnp.float32)
    assert abs(float(ce_t) - float(ce_j)) < 1e-4
    # aller-retour de conversion
    back = jax_to_hf(params)
    assert set(back) == set(sd) and all(np.array_equal(back[k], sd[k]) for k in sd)


@pytest.mark.parametrize("optimizer", ["muon", "adamw"])
def test_overfit_one_batch_torch(optimizer):
    from llmedith.torch_impl.muon import build_optimizers
    torch.manual_seed(0)
    m = LlmEdith(CFG).train()
    tc = TrainConfig(optimizer=optimizer, lr=1e-2, weight_decay=0.0)
    opts = build_optimizers(m, tc)
    x, y = tokens(), tokens(seed=1)
    for _ in range(150):
        loss, ce = m.loss(x, y)
        loss.backward()
        for o in opts:
            o.step()
            o.zero_grad()
    assert ce.item() < 0.5, ce.item()


def test_overfit_one_batch_jax():
    jax = pytest.importorskip("jax")
    import optax
    jnp = jax.numpy
    from llmedith.jax_impl.model import init_params, loss_fn
    from llmedith.jax_impl.muon import build_optimizer
    tc = TrainConfig(optimizer="muon", weight_decay=0.0)
    params = init_params(CFG, jax.random.PRNGKey(0))
    tx = build_optimizer(tc, params)
    state = tx.init(params)
    x, y = jnp.asarray(tokens().numpy()), jnp.asarray(tokens(seed=1).numpy())

    @jax.jit
    def step(params, state):
        (_, ce), g = jax.value_and_grad(lambda p: loss_fn(CFG, p, x, y, dtype=jnp.float32), has_aux=True)(params)
        u, state = tx.update(g, state, params)
        return optax.apply_updates(params, jax.tree.map(lambda a: 1e-2 * a, u)), state, ce

    for _ in range(150):
        params, state, ce = step(params, state)
    assert float(ce) < 0.5, float(ce)
