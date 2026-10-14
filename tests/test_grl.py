"""Real GodotRocketLanding runs; skipped unless the `grl` package and its game binary are installed."""

import dataclasses
import functools
from pathlib import Path
import numpy as np
import pytest
from world_model.config import Config
from world_model.data import collect

grl = pytest.importorskip("grl")
try:
    grl.find_binary()
except FileNotFoundError:
    pytest.skip("GodotRocketLanding binary not found", allow_module_level=True)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke_grl_landing.json"


@pytest.mark.parametrize("policy", ["random", "demonstrator"])
def test_grl_collection(tmp_path, policy):
    cfg = dataclasses.replace(Config.load(CONFIG), episodes=2, max_steps=6)
    make_policy = None
    if policy == "demonstrator":
        from grl.policies import demonstrator

        make_policy = functools.partial(demonstrator, noise=0.1)
    manifest = collect(cfg, tmp_path / "data", make_policy=make_policy)
    assert manifest["complete"] and manifest["action_spec"]["kind"] == "box"
    for episode in manifest["episodes"]:
        pixels = np.load(tmp_path / "data" / episode["name"] / "pixels.npy")
        assert pixels.shape == (episode["steps"] + 1, cfg.image_size, cfg.image_size, 3)
        assert pixels.std() > 0
