# GodotRocketLanding

[GodotRocketLanding](https://github.com/drazicmartin/GodotRocketLanding) (GRL) is a Godot rocket-landing
simulator driven over a WebSocket. Its `grl` package provides `GRLEnv`, a Gymnasium environment that launches
one headless game per environment instance, and an `rgb_array` render mode that this pipeline trains on.

## Setup

Install GRL into the same venv. Its repository contains the exported game (Godot 4.7: `GRL.x86_64` on Linux,
`GRL.exe` on Windows, stored with Git LFS, plus `GRL.pck`):

```bash
git clone https://github.com/drazicmartin/GodotRocketLanding.git
python -m pip install -e ./GodotRocketLanding        # websockets, gymnasium, numpy, pillow
```

The binary is found in `$GRL_BINARY`, the GRL repository root, then the current directory. Each environment
starts its own game on a free local port and stops it on `env.close()`; no display is needed. A missing
executable bit is fixed automatically; a Git LFS pointer instead of `GRL.exe` fails with a hint to run
`git lfs pull`.

## Frames

The game runs headless, so GRL draws `render()` frames in Python from the state the game returns:
deterministic, and identical on every machine. The camera follows the rocket with the local vertical up and
shows the sky (darker with altitude), the planet surface (striped so sideways motion is visible), the landing
pad, the rocket (tilt, legs, flames for the last action, explosion on crash) and a velocity marker. A HUD strip
shows the pad direction and distance and propellant/integrity/legs bars; the right edge is an altitude gauge.
Set `env_kwargs.render_size` to `image_size` so frames are drawn at the training resolution instead of being
resized.

## Configs

| Config | Purpose |
|---|---|
| `configs/grl_landing.json` | `random_level_easy`, 64×64 frames, `frame_skip` 5 (6 control steps/s), 500 steps (~83 s) |
| `configs/smoke_grl_landing.json` | Tiny CPU plumbing check (8 episodes × 40 steps, 32×32, 2 epochs) |

`frame_skip` holds each action for that many physics ticks (1/30 s) inside the game, in one round trip; prefer
it over `action_repeat`. `GRLEnv`'s own `max_steps` counts physics ticks: keep it at `max_steps × frame_skip`
(2500 = 500 × 5) so both limits agree. Actions are `Box(4)` in `[-1, 1]` (main engine, left RCS, right RCS,
legs `> 0` deploys). Rewards come from `GRLEnv.get_reward`: +100 for a landing, −100 for a crash, otherwise a
small penalty for the distance to the pad and the speed. A landing (terminated) needs the legs down, both feet
on the pad and the rocket nearly still for 0.5 s. Other levels (`level_1` … `level_4`, `random_level_hard`, …),
frame skips or render sizes need their own data and checkpoint; `env_kwargs` are checked automatically.

## Data: demonstrations, not only random actions

Uniform random actions average half thrust, about twice what hovering needs: the rocket climbs away or
crashes and the data never shows a landing, so a model trained on it cannot plan one. Collect
demonstrations with GRL's scripted lander instead (it reads the game state; only frames, actions and rewards
are saved):

```bash
world-model collect --config configs/grl_landing.json --output data/grl \
  --policy grl.policies:demonstrator --policy-kwargs '{"noise": 0.1}'
world-model train --config configs/grl_landing.json --data data/grl --output runs/grl
world-model eval-model --checkpoint runs/grl/best.pt --data data/grl --horizon 10 \
  --output runs/grl/model_eval.json
world-model eval --checkpoint runs/grl/best.pt --episodes 10 --output runs/grl/mpc_eval.json
world-model eval --checkpoint runs/grl/best.pt --episodes 10 --random --output runs/grl/random_eval.json
```

On seeds 0–29 of `random_level_easy` (500 steps, `frame_skip` 5) the lander lands 28 times without noise,
22 with `noise=0.1` (the default) and 11 with `noise=0.2`; the others time out or crash, which also covers
those outcomes. Random-action datasets (`collect` without `--policy`) and different noise levels can be
combined into one dataset as described in [design](design.md#better-data).

Smoke check (seconds):

```bash
world-model collect --config configs/smoke_grl_landing.json --output runs/grl_smoke/data \
  --policy grl.policies:demonstrator
world-model train --config configs/smoke_grl_landing.json --data runs/grl_smoke/data \
  --output runs/grl_smoke/train --device cpu
world-model eval --checkpoint runs/grl_smoke/train/best.pt --episodes 2 --device cpu \
  --output runs/grl_smoke/eval.json
```

`tests/test_grl.py` runs a short random and demonstrator collection against the real game; it is skipped when
`grl` or its binary is not installed (as in CI).

## Limitations

CEM plans 15 steps (2.5 s at `frame_skip` 5) and has no terminal value, while a landing takes tens of seconds
and ends with a precise, slow touchdown (the feet have no sideways friction, so the rocket must arrive without
horizontal speed). Expect MPC to improve on random actions in the shaped reward long before it lands; landing
reliably is not claimed. Compare MPC and random actions on the same seeds, as for the other environments.
