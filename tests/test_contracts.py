import dataclasses
import functools
import json
from pathlib import Path
import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete
from world_model.config import Config
from world_model.envs import ActionCodec, step
from world_model.data import Windows, collect, split_episodes
from world_model.example_env import expert


@pytest.mark.parametrize(
    "space",
    [
        Discrete(3, start=2),
        Box(np.array([[-2.0, 1.0]], dtype=np.float32), np.array([[3.0, 8.0]], dtype=np.float32)),
    ],
)
def test_action_round_trip(space):
    codec = ActionCodec.from_space(space)
    space.seed(0)
    for _ in range(10):
        action = space.sample()
        assert np.allclose(action, codec.decode(codec.encode(action)), atol=1e-6)


def test_unbounded_actions_rejected():
    with pytest.raises(ValueError, match="finite"):
        ActionCodec.from_space(Box(-np.inf, np.inf, (1,), dtype=np.float32))


def test_repeat_stops_at_truncation():
    class FakeEnv:
        calls = 0

        def step(self, action):
            self.calls += 1
            return None, 2.0, False, self.calls == 2, {}

    env = FakeEnv()
    assert step(env, 0, 5) == (4.0, False, True)
    assert env.calls == 2


def test_collection_alignment_and_split(tmp_path, tiny_config):
    manifest = collect(tiny_config, tmp_path / "data")
    train, val = split_episodes(manifest, 0, 0.25)
    assert {e["name"] for e in train}.isdisjoint({e["name"] for e in val})
    windows = Windows(tmp_path / "data", manifest["episodes"], tiny_config.history)
    assert len(windows) == 4 * (8 - 3 + 1)
    assert windows[0]["pixels"].shape == (4, 16, 16, 3)
    for episode in manifest["episodes"]:
        root = tmp_path / "data" / episode["name"]
        assert len(np.load(root / "pixels.npy")) == len(np.load(root / "actions.npy")) + 1
        assert not np.load(root / "terminated.npy").any()
        assert np.load(root / "truncated.npy")[-1]
    assert json.loads((tmp_path / "data" / "manifest.json").read_text())["complete"]


def test_expert_policy_reads_simulator_state(tmp_path, tiny_config):
    cfg = dataclasses.replace(
        tiny_config,
        env_id="DotReach-v0",
        factory="world_model.example_env:make_env",
        env_kwargs={"max_steps": 60},
        max_steps=60,
    )
    make_policy = functools.partial(expert, noise=0.0)
    manifest = collect(cfg, tmp_path / "data", make_policy=make_policy)
    assert manifest["policy"] == "expert"
    for episode in manifest["episodes"]:
        root = tmp_path / "data" / episode["name"]
        assert np.load(root / "terminated.npy")[-1]  # reached the goal before the time limit
        assert len(np.load(root / "pixels.npy")) == episode["steps"] + 1
        assert np.abs(np.load(root / "actions.npy")).max() <= 1.0
    with pytest.raises(ValueError, match="either"):
        collect(cfg, tmp_path / "other", policy=object(), make_policy=make_policy)


def test_bundled_configs_load():
    for path in sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.json")):
        assert Config.load(path).max_steps > 0
