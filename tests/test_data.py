import numpy as np

from llmedith.data import ShardLoader, write_shard
from llmedith.schedule import wsd_lr_factor


def _make_shards(tmp_path, n=3, size=1000):
    for i in range(n):
        write_shard(str(tmp_path / f"train_{i:05d}.bin"), (np.arange(size) + i * size).astype(np.uint16))
    return str(tmp_path / "train_*.bin")


def test_loader_shift_and_resume(tmp_path):
    pattern = _make_shards(tmp_path)
    a = ShardLoader(pattern, 2, 16)
    x, y = a.next_batch()
    assert (y[:, :-1] == x[:, 1:]).all() and y[0, -1] == x[1, 0]
    for _ in range(40):          # traverse plusieurs shards
        a.next_batch()
    state = a.state_dict()
    expected = [a.next_batch()[0] for _ in range(5)]
    b = ShardLoader(pattern, 2, 16)
    b.load_state_dict(state)
    for e in expected:
        assert np.array_equal(b.next_batch()[0], e)


def test_wsd():
    f = [wsd_lr_factor(s, 100, 10, 0.2) for s in range(100)]
    assert f[0] == 0.1 and f[9] == 1.0 and f[50] == 1.0 and f[80] == 1.0
    assert f[99] < 0.25 and all(f[i] >= f[i + 1] for i in range(80, 99))


def test_loader_hub_lazy(tmp_path, monkeypatch):
    """Mode Hub : téléchargement à la demande, préchargement, suppression des vieux shards, reprise identique."""
    import shutil
    import huggingface_hub
    remote = tmp_path / "remote"
    remote.mkdir()
    _make_shards(remote, n=6, size=500)
    downloads = []
    monkeypatch.setattr(huggingface_hub, "list_repo_files",
                        lambda repo, repo_type=None: sorted(p.name for p in remote.iterdir()))

    def fake_download(repo, filename, repo_type=None, local_dir="."):
        downloads.append(filename)
        shutil.copy(remote / filename, f"{local_dir}/{filename}")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    local = tmp_path / "local"
    local.mkdir()
    ref = ShardLoader(str(remote / "train_*.bin"), 2, 16)
    hub = ShardLoader(str(local / "train_*.bin"), 2, 16, hub_repo="me/data", keep_local=2)
    for _ in range(60):
        assert np.array_equal(ref.next_batch()[0], hub.next_batch()[0])
    assert len(list(local.iterdir())) <= 3
    resumed = ShardLoader(str(local / "train_*.bin"), 2, 16, hub_repo="me/data")
    resumed.load_state_dict(hub.state_dict())
    assert np.array_equal(resumed.next_batch()[0], ref.next_batch()[0])
