import dataclasses
import json
import subprocess
import sys
import numpy as np
import pytest
import torch
from world_model.data import collect
from world_model.evaluate import evaluate, evaluate_model
from world_model.envs import frame, make_env
from world_model.planner import MPCPolicy
from world_model.train import load_checkpoint, train


def test_custom_factory_cli_from_another_directory(tmp_path, tiny_config):
    cfg = dataclasses.replace(
        tiny_config, env_id="DotReach-v0", factory="world_model.example_env:make_env"
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg.to_dict()))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "world_model.cli",
            "collect",
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "custom"),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    manifest = json.loads((tmp_path / "custom" / "manifest.json").read_text())
    assert manifest["complete"] and len(manifest["episodes"]) == cfg.episodes


def test_expert_policy_cli(tmp_path, tiny_config):
    cfg = dataclasses.replace(
        tiny_config, env_id="DotReach-v0", factory="world_model.example_env:make_env"
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg.to_dict()))
    command = [
        sys.executable,
        "-m",
        "world_model.cli",
        "collect",
        "--config",
        str(config_path),
        "--output",
        str(tmp_path / "expert"),
        "--policy",
        "world_model.example_env:expert",
        "--policy-kwargs",
        '{"noise": 0.1, "seed": 3}',
    ]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    manifest = json.loads((tmp_path / "expert" / "manifest.json").read_text())
    assert manifest["complete"] and manifest["policy"] == "world_model.example_env:expert"
    command[7] = str(tmp_path / "both")
    rejected = subprocess.run(
        command + ["--checkpoint", "unused.pt"], cwd=tmp_path, capture_output=True, text=True
    )
    assert rejected.returncode != 0 and "mutually exclusive" in rejected.stderr


def test_train_reload_resume_and_mpc(tmp_path, tiny_config):
    cfg = tiny_config
    data, run = tmp_path / "data", tmp_path / "run"
    collect(cfg, data)
    best = train(cfg, data, run, "cpu")
    model, loaded_cfg, saved = load_checkpoint(best)
    assert loaded_cfg == cfg and saved["epoch"] == 0
    assert all(torch.isfinite(p).all() for p in model.parameters())
    training = json.loads((run / "split.json").read_text())["training"]
    actions = np.concatenate([np.load(data / name / "actions.npy") for name in training])
    assert np.allclose(saved["action_prior"]["mean"], actions.mean(0), atol=1e-6)
    assert np.allclose(saved["action_prior"]["std"], actions.std(0), atol=1e-6)
    policy = MPCPolicy.from_checkpoint(best, seed=5)
    assert torch.allclose(policy.cem.prior_mean, torch.tensor(actions.mean(0)))
    env = make_env(cfg)
    try:
        env.reset(seed=99)
        for _ in range(5):
            action = policy.act(frame(env, cfg.image_size))
            assert env.action_space.contains(action)
            env.step(action)
        policy.reset()
        assert not policy.actions and not policy.embeddings
    finally:
        env.close()
    result = evaluate(best, episodes=1, device="cpu")
    assert np.isfinite(result["mean_return"]) and result["planner"]["action_prior"] == "data"
    assert "success_rate" not in result  # Pendulum has no is_success
    offline = evaluate_model(best, data, horizon=2, max_windows=2)
    assert len(offline["latent_mse_by_step"]) == 2
    train(dataclasses.replace(cfg, epochs=2), data, run, "cpu", resume=run / "last.pt")
    _, _, last = load_checkpoint(run / "last.pt")
    assert last["epoch"] == 1


def test_checkpoint_without_action_prior_needs_uniform_planning(tmp_path, tiny_config):
    cfg = dataclasses.replace(
        tiny_config, env_id="DotReach-v0", factory="world_model.example_env:make_env"
    )
    collect(cfg, tmp_path / "data")
    best = train(cfg, tmp_path / "data", tmp_path / "run", "cpu")
    result = evaluate(best, episodes=2, device="cpu")
    assert [set(row) >= {"success"} for row in result["episodes"]] == [True, True]
    assert result["success_rate"] == np.mean([row["success"] for row in result["episodes"]])
    saved = torch.load(best, weights_only=True)
    del saved["action_prior"]  # as saved before action priors existed
    torch.save(saved, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="uniform"):
        MPCPolicy.from_checkpoint(tmp_path / "old.pt")
    policy = MPCPolicy.from_checkpoint(tmp_path / "old.pt", action_prior="uniform")
    assert torch.equal(policy.cem.prior_mean, torch.zeros(2))
    assert torch.equal(policy.cem.prior_std, torch.ones(2))


def test_outcome_losses_do_not_change_representation(tiny_config):
    from world_model.model import WorldModel

    model = WorldModel(tiny_config, 1)
    batch = {
        "pixels": torch.randint(0, 256, (4, 4, 16, 16, 3), dtype=torch.uint8),
        "actions": torch.randn(4, 3, 1),
        "rewards": torch.randn(4, 3),
        "terminated": torch.zeros(4, 3),
    }
    losses = model.losses(batch, tiny_config.sigreg_weight)
    (losses["reward"] + losses["termination"]).backward()
    assert all(p.grad is None for p in model.jepa.parameters())
    assert any(p.grad is not None for p in model.outcome.parameters())


def test_preloading_does_not_change_training(tmp_path, tiny_config):
    collect(tiny_config, tmp_path / "data")
    runs = {}
    for name, limit in (("streamed", 0.0), ("preloaded", 4.0)):
        cfg = dataclasses.replace(tiny_config, preload_max_gb=limit)
        model, _, _ = load_checkpoint(train(cfg, tmp_path / "data", tmp_path / name, "cpu"))
        runs[name] = model.state_dict()
    assert all(torch.equal(runs["streamed"][k], runs["preloaded"][k]) for k in runs["streamed"])
    # preload_max_gb only changes how batches are read, so it may differ on resume
    cfg = dataclasses.replace(tiny_config, epochs=2, preload_max_gb=0.0)
    train(cfg, tmp_path / "data", tmp_path / "preloaded", "cpu", tmp_path / "preloaded" / "last.pt")
