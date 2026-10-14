"""Watch a trained GodotRocketLanding checkpoint fly in the game window.

python -m examples.grl_watch --checkpoint runs/grl/best.pt --seed 7 --episodes 3 --zoom 1.0

The planner still sees the small frames it was trained on (drawn from the game state); the window shows the
game itself at full resolution, in real time, holding still while the planner thinks. Seeds give new start
positions and pad placements; another `--level` than the training one is outside the training data. GRL
always launches the game headless on Linux (grl/process.py), so the window only appears on Windows.
"""

import argparse
import dataclasses
import time
from world_model.envs import frame, make_env, step
from world_model.planner import MPCPolicy
from world_model.train import resolve_device
from world_model.progress import stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=200000, help="first episode seed")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--level", help="GRL level (default: the training level)")
    parser.add_argument("--zoom", type=float, default=1.0, help="camera zoom, 0.4 (far) to 2.0")
    parser.add_argument("--max-steps", type=int, help="default: the training episode length")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds to show each outcome")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    policy = MPCPolicy.from_checkpoint(args.checkpoint, resolve_device(args.device))
    env_kwargs = dict(
        policy.cfg.env_kwargs, show_window=True, config={"camera": {"zoom": args.zoom}}
    )
    if args.level:
        env_kwargs["level_name"] = args.level
    env = make_env(dataclasses.replace(policy.cfg, env_kwargs=env_kwargs))
    max_steps = args.max_steps or policy.cfg.max_steps
    stage(f"Watching | level={env_kwargs.get('level_name')} | device={policy.device} | CEM MPC")
    try:
        for seed in range(args.seed, args.seed + args.episodes):
            env.reset(seed=seed)
            policy.reset()
            total, steps, terminated = 0.0, 0, False
            while steps < max_steps and not terminated:
                action = policy.act(frame(env, policy.cfg.image_size))
                reward, terminated, truncated = step(env, action, policy.cfg.action_repeat)
                total += reward
                steps += 1
                terminated = terminated or truncated
            outcome = env.unwrapped.last_state.get("game_state", "running")
            print(f"seed {seed}: {outcome} after {steps} steps, return {total:.1f}")
            time.sleep(args.pause)
    finally:
        env.close()


if __name__ == "__main__":
    main()
