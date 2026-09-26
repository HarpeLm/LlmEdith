from llmedith.config import Config


def test_overrides_top_level_and_sections():
    cfg = Config.load("configs/tiny.yaml", ["name=tiny_adamw", "train.optimizer=adamw",
                                            "data.hub_repo=Moi/data"])
    assert cfg.name == "tiny_adamw"
    assert cfg.train.optimizer == "adamw"
    assert cfg.data.hub_repo == "Moi/data"
