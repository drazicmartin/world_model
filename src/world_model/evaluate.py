"""Real-environment policy evaluation, plus offline multi-step latent evaluation."""

import json
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from .envs import ActionCodec, frame, make_env, step
from .planner import MPCPolicy
from .train import check_dataset, load_checkpoint, resolve_device
from .data import split_episodes
from .progress import Progress, stage


def evaluate(
    checkpoint,
    episodes=10,
    seed=100000,
    device="auto",
    random_policy=False,
    video=None,
    planner_overrides=None,
):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    device = resolve_device(device)
    stage(
        f"Policy evaluation | loading={checkpoint} | device={device} | "
        f"policy={'random' if random_policy else 'CEM MPC'} | episodes={episodes}"
    )
    policy = MPCPolicy.from_checkpoint(checkpoint, device, seed, **(planner_overrides or {}))
    stage(
        f"Environment={policy.cfg.env_id} | horizon={policy.cfg.horizon} | "
        f"population={policy.cfg.population} | iterations={policy.cfg.iterations}"
    )
    env = make_env(policy.cfg)
    writer = None
    rows = []
    try:
        if ActionCodec.from_space(env.action_space).spec != policy.codec.spec:
            raise ValueError("Environment action space differs from checkpoint")
        if video:
            import imageio.v2 as imageio

            writer = imageio.get_writer(video, fps=env.metadata.get("render_fps", 30))
        with Progress(episodes, "Evaluating policy", "episode") as progress:
            for episode in range(episodes):
                env.reset(seed=seed + episode)
                env.action_space.seed(seed + episode)
                policy.reset()
                total, latencies = 0.0, []
                for t in range(policy.cfg.max_steps):
                    pixels = frame(env, policy.cfg.image_size)
                    if writer and episode == 0:
                        writer.append_data(pixels)
                    start = time.perf_counter()
                    action = env.action_space.sample() if random_policy else policy.act(pixels)
                    latencies.append(time.perf_counter() - start)
                    reward, terminated, truncated, info = step(
                        env, action, policy.cfg.action_repeat, return_info=True
                    )
                    total += reward
                    progress.update(
                        0, episode=episode + 1, step=t + 1, episode_return=f"{total:.2f}"
                    )
                    if terminated or truncated:
                        break
                rows.append(
                    {
                        "seed": seed + episode,
                        "return": total,
                        "steps": t + 1,
                        "terminated": terminated,
                        "truncated": truncated
                        or (t + 1 == policy.cfg.max_steps and not terminated),
                        "mean_action_seconds": float(np.mean(latencies)),
                    }
                )
                if "is_success" in info:  # Gymnasium's success flag, e.g. a GRL landing
                    rows[-1]["success"] = bool(info["is_success"])
                progress.update(1, last_return=f"{total:.2f}")
    finally:
        env.close()
        if writer:
            writer.close()
    returns = np.array([row["return"] for row in rows])
    successes = [row["success"] for row in rows if "success" in row]
    success = f" | success_rate={np.mean(successes):.2f}" if successes else ""
    stage(
        f"Policy evaluation complete | mean_return={returns.mean():.2f}{success} | "
        f"episodes={episodes}"
    )
    return {
        "policy": "random" if random_policy else "cem",
        "checkpoint": str(checkpoint),
        "planner": {
            k: getattr(policy.cfg, k)
            for k in ("horizon", "population", "iterations", "action_prior")
        },
        "episodes": rows,
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "standard_error": float(returns.std(ddof=1) / np.sqrt(episodes)) if episodes > 1 else None,
        **({"success_rate": float(np.mean(successes))} if successes else {}),
    }


@torch.inference_mode()
def evaluate_model(checkpoint, data, device="auto", horizon=10, max_windows=100):
    """Roll out recorded actions on held-out episodes, without teacher forcing."""
    if horizon < 1 or max_windows < 1:
        raise ValueError("horizon and max_windows must be positive")
    device = resolve_device(device)
    stage(
        f"Model evaluation | loading={checkpoint} | device={device} | "
        f"horizon={horizon} | max_windows={max_windows} | data={data}"
    )
    model, cfg, saved = load_checkpoint(checkpoint, device)
    manifest_text = (Path(data) / "manifest.json").read_text()
    manifest = json.loads(manifest_text)
    check_dataset(cfg, manifest)
    if hashlib.sha256(manifest_text.encode()).hexdigest() != saved["dataset_fingerprint"]:
        raise ValueError("Use the training dataset to reproduce its held-out episode split")
    if manifest["action_spec"] != saved["action_spec"]:
        raise ValueError("Dataset action specification differs from checkpoint")
    _, episodes = split_episodes(manifest, cfg.seed, cfg.validation_fraction)
    total_windows = min(
        max_windows,
        sum(len(range(0, e["steps"] - cfg.history - horizon + 2, horizon)) for e in episodes),
    )
    latent, persistence, reward, termination = [], [], [], []
    with Progress(total_windows, "Evaluating model rollouts", "window") as progress:
        for episode in episodes:
            root = Path(data) / episode["name"]
            pixels = np.load(root / "pixels.npy", mmap_mode="r")
            actions = np.load(root / "actions.npy", mmap_mode="r")
            rewards = np.load(root / "rewards.npy", mmap_mode="r")
            terms = np.load(root / "terminated.npy", mmap_mode="r")
            for start in range(0, len(actions) - cfg.history - horizon + 2, horizon):
                offset = start + cfg.history - 1
                p = torch.tensor(np.array(pixels[start : offset + horizon + 1]), device=device)[
                    None
                ]
                truth = model.encode(p)
                z = truth[:, : cfg.history]
                a = torch.tensor(np.array(actions[start:offset]), device=device)[None]
                le, pe, re, te = [], [], [], []
                for t in range(horizon):
                    current = torch.tensor(np.array(actions[offset + t]), device=device)[None, None]
                    a = torch.cat([a, current], 1)[:, -cfg.history :]
                    prediction = model.predict(z[:, -cfg.history :], a)[:, -1:]
                    r, done = model.outcomes(z[:, -1:], current, prediction)
                    target = truth[:, cfg.history + t : cfg.history + t + 1]
                    le.append((prediction - target).square().mean().item())
                    pe.append(
                        (truth[:, cfg.history - 1 : cfg.history] - target).square().mean().item()
                    )
                    re.append((r.item() - float(rewards[offset + t])) ** 2)
                    te.append((done.sigmoid().item() - float(terms[offset + t])) ** 2)
                    z = torch.cat([z, prediction], 1)[:, -cfg.history :]
                    a = a[:, -(cfg.history - 1) :] if cfg.history > 1 else a[:, :0]
                latent.append(le)
                persistence.append(pe)
                reward.append(re)
                termination.append(te)
                progress.update(latent_mse=f"{le[-1]:.4f}")
                if len(latent) >= max_windows:
                    break
            if len(latent) >= max_windows:
                break
    if not latent:
        raise ValueError("No validation episodes long enough for requested rollout horizon")
    stage(f"Model evaluation complete | {len(latent)} windows | horizon={horizon}")
    return {
        "windows": len(latent),
        "horizon": horizon,
        "latent_mse_by_step": np.mean(latent, 0).tolist(),
        # Baseline: the last observed embedding, i.e. "nothing changes". A useful model is well below it.
        "persistence_mse_by_step": np.mean(persistence, 0).tolist(),
        "reward_rmse_by_step": np.sqrt(np.mean(reward, 0)).tolist(),
        "termination_brier_by_step": np.mean(termination, 0).tolist(),
    }
