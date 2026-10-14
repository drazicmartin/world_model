# world_model

**Train [LeWorldModel](https://github.com/lucas-maes/le-wm) from Gymnasium-rendered images and control environments with cross-entropy method (CEM) model predictive control (MPC).**

The pipeline is explicit: collect RGB trajectories → learn latent dynamics → fit reward/termination heads → evaluate imagined rollouts → run closed-loop MPC. Simulator state observations are never model inputs. Train a separate checkpoint per environment.

This repository includes the **actual upstream `JEPA`, `ARPredictor`, action embedder, projection MLPs, and `SIGReg`**, pinned at commit [`8edfeb3`](https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac). A small unmodified source snapshot is included so a normal clone and package install work without submodule setup. See [upstream integration](docs/upstream.md) for provenance and the differences from the paper's experimental setup.

> **Validation status:** CPU end-to-end smoke runs exercise collection, two training epochs, checkpoint loading, CEM evaluation, and latent rollout evaluation on three real environments. These tiny models are not trained benchmark policies. Default training recipes require longer runs and task-specific tuning; solved-task performance is not claimed. See [validation](docs/validation.md).

## Quick start

**Recommended: Python 3.11.** GitHub CI runs the tests and three-environment smoke pipeline on Python 3.11. Python 3.12 was also tested locally; the package declares support for Python 3.10–3.12, but Python 3.10 has not been validated here. Python 3.13 and newer are outside the supported range.

Install Python 3.11 and Git before starting. Authenticate with GitHub to clone this private repository.

```bash
git clone https://github.com/haile-leul/world_model.git
cd world_model
python3.11 -m venv .venv                 # Windows: py -3.11 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python --version                        # should report Python 3.11.x
python -m pip install --upgrade pip
python -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"

# Actual small training/evaluation runs in all three environments:
python scripts/smoke.py
```

For an NVIDIA GPU, install the appropriate PyTorch 2.6.0 build for your driver instead of the CPU wheel, then install this package. Commands accept `--device auto` (CUDA when available), `cpu`, or `cuda:0`. macOS CPU works in principle; only Linux CPU has been exercised here. MuJoCo/Box2D/custom environments need their own simulator dependencies.

No model downloads, WandB account, or upstream training framework are needed. The ViT is initialized from scratch. A clean venv avoids unrelated torchvision/transformers version conflicts.

## Progress and terminal output

Every command announces what it will run, including the environment, relevant device/settings, and input/output locations. Progress is enabled by default and supports Windows PowerShell.

- **Collection:** episode progress with the current episode/step and last episode return; completion confirms where data was saved.
- **Training:** setup and dataset sizes, reward-statistics progress, epoch numbers, separate training/validation batch bars with running loss, checkpoint-save messages, and a compact epoch summary. Full metrics remain in `metrics.jsonl`.
- **Policy evaluation:** episode progress, current step/return, and the final mean return.
- **Model evaluation:** completed rollout windows and the latest final-step latent error.
- **Smoke script:** announces each environment and runs the same stage reporting throughout.
- **Inference example:** checkpoint loading, environment settings, and step progress. An episode that ends early retains its actual step count.

Interactive progress bars show counts, percentage, elapsed time, rate, and estimated remaining time. Episode ETA is estimated from completed episodes and can vary with episode length. Bars refresh at most every half-second; they advance after work completes, so one slow simulator step or model batch can pause the display. Redirected/noninteractive output emits start/end messages and progress at most every 10 seconds rather than repeated terminal control sequences. Status output goes to stderr; evaluation JSON remains on stdout and in the requested result file.

Suppress stage/progress output for an individual command with `--no-progress`, for example:

```bash
world-model train --config configs/pendulum.json --data data/pendulum --output runs/pendulum --no-progress
```

For scripts or Python API calls, set `WORLD_MODEL_PROGRESS=0`. In PowerShell: `$env:WORLD_MODEL_PROGRESS = "0"`; restore with `Remove-Item Env:WORLD_MODEL_PROGRESS`. Final CLI result output is still printed.

## Train a useful-sized model

```bash
world-model collect --config configs/pendulum.json --output data/pendulum
world-model train --config configs/pendulum.json --data data/pendulum --output runs/pendulum

# Held-out multi-step dynamics/reward prediction:
world-model eval-model --checkpoint runs/pendulum/best.pt --data data/pendulum \
  --horizon 10 --output runs/pendulum/model_eval.json

# Closed-loop MPC and a random baseline on identical fresh seeds:
world-model eval --checkpoint runs/pendulum/best.pt --episodes 10 --seed 100000 \
  --output runs/pendulum/mpc_eval.json
world-model eval --checkpoint runs/pendulum/best.pt --episodes 10 --seed 100000 \
  --random --output runs/pendulum/random_eval.json
```

These are starting recipes, not tuned hyperparameters. CPU smoke configs are intentionally much smaller. Collection/run/result paths must be new, except when explicitly resuming. Data, checkpoints, and videos are gitignored.

| Config | Environment | Action space | Main caveat |
|---|---|---|---|
| `cartpole.json` | `CartPole-v1` | Discrete | Constant per-step rewards make accurate termination prediction important |
| `pendulum.json` | `Pendulum-v1` | Continuous Box | Motion needs frame history; reward model must distinguish velocity |
| `mountain_car.json` | `MountainCarContinuous-v0` | Continuous Box | Random data rarely contains successful trajectories; use exploratory/expert data |
| `custom.json` | Included `DotReach` example | Continuous Box | Demonstrates the factory interface |
| `grl_landing.json` | [GodotRocketLanding](docs/godot-rocket-landing.md) `GRL/Landing-v0` | Continuous Box | Needs the `grl` package and game; collect demonstrations with `--policy grl.policies:demonstrator` |

Change the JSON files to set collection size, image size, model dimensions, batch size, optimizer, training epochs, and CEM settings. Unknown fields fail early. Training loads datasets of up to `preload_max_gb` (default 4) into memory, where each batch is gathered in one indexing step; larger datasets stream from disk, using `workers` loader processes. Defaults are defined in [`config.py`](src/world_model/config.py); the fully resolved config is saved in every run and checkpoint.

To resume, increase `epochs` in the same config (it means **total** epochs) and use:

```bash
world-model train --config configs/pendulum.json --data data/pendulum \
  --output runs/pendulum --resume runs/pendulum/last.pt
```

Resume restores model, optimizer, epoch, RNG, and loader state. Use `last.pt` for uninterrupted training continuity. `best.pt` minimizes held-out prediction + normalized reward + termination loss, **not policy return**. It is not a substitute for evaluating the controller. Keep the original dataset immutable; its manifest fingerprint protects the split and resume path.

## Inference

```bash
python -m examples.inference --checkpoint runs/pendulum/best.pt
```

The public API is small:

```python
from world_model.planner import MPCPolicy
from world_model.envs import frame, make_env, step

policy = MPCPolicy.from_checkpoint("runs/pendulum/best.pt", device="cpu")
env = make_env(policy.cfg)
try:
    env.reset(seed=123)
    policy.reset()                         # required at every episode boundary
    for _ in range(policy.cfg.max_steps):
        action = policy.act(frame(env, policy.cfg.image_size))
        reward, terminated, truncated = step(env, action, policy.cfg.action_repeat)
        if terminated or truncated:
            break
finally:
    env.close()
```

`act()` consumes one resized `uint8` RGB frame and returns an action in the environment's original units. For a non-simulator caller, provide `(image_size, image_size, 3)` pixels with the same viewpoint and resizing as training. A policy instance owns one environment's history and must not be shared between parallel episodes.

LeWM predicts **embeddings, not decoded images**. For open-loop inference, use `WorldModel.encode`, `predict`, and `imagine`; [`evaluate_model`](src/world_model/evaluate.py) shows a complete recorded-action rollout. There is no pixel decoder or value function in this implementation.

## CEM MPC

At each environment step, CEM samples action sequences, rolls out latent dynamics, ranks predicted discounted returns, and refits to the best sequences. Only the first action is executed; the next observed frame corrects the state estimate before planning again.

- Continuous actions: bounded Gaussian sampling in normalized `[-1, 1]` coordinates.
- Discrete actions: categorical sampling and elite-frequency updates, without continuous rounding.
- Candidate rollouts are vectorized and chunked; the current frame is encoded once per control step.
- Previous CEM means/probabilities warm-start the next plan; history and plans reset per episode.
- Rewards are weighted by predicted survival. Time-limit truncation never becomes a terminal training target.

Override planning cost/quality tradeoffs without retraining:

```bash
world-model eval --checkpoint runs/pendulum/best.pt --horizon 20 \
  --population 512 --elites 64 --iterations 6 --candidate-chunk 64 \
  --output runs/pendulum/larger_planner.json
```

Longer horizons can amplify model error. Increase population for search quality and reduce `candidate_chunk` for memory. Changing `action_repeat` or image preprocessing requires compatible data and retraining.

## Your own environments

Any environment meeting the documented contract can use the same pipeline. It must expose Gymnasium's reset/step API, RGB rendering, and a `Discrete` or finite floating-point `Box` action space. Other spaces need an explicit wrapper. Older Gym's four-return API needs an adapter; it is not silently guessed.

See the complete [custom environment guide](docs/custom-environments.md) and runnable [`DotReach`](examples/custom_env.py):

```bash
world-model collect --config configs/custom.json --output data/custom
world-model train --config configs/custom.json --data data/custom --output runs/custom
world-model eval --checkpoint runs/custom/best.pt --output runs/custom/eval.json
```

When random actions never reach the interesting states, collect with a scripted expert instead: `--policy MODULE:FACTORY` calls `FACTORY(env, **policy_kwargs)` once the environment exists, and the returned object's `reset()` / `act(pixels)` choose the actions. The expert may read simulator state; only frames, actions and rewards are saved. For example, `world-model collect --config configs/custom.json --output data/custom_expert --policy world_model.example_env:expert --policy-kwargs '{"noise": 0.2}'`.

[GodotRocketLanding](docs/godot-rocket-landing.md) is a complete external example: a Godot rocket-landing simulator whose `grl` package provides the environment, its pixel rendering and a demonstrator for `--policy`.

## Understand the implementation

| File | Responsibility |
|---|---|
| `envs.py` | Factory loading, RGB resizing, action normalization, action repeat |
| `data.py` | Collection, episode split, lazy memory-mapped sequence windows |
| `model.py` | Upstream LeWM construction, loss, reward/termination heads, imagined returns |
| `train.py` | Optimizer, held-out metrics, checkpoints, resume |
| `planner.py` | CEM search and stateful inference policy |
| `evaluate.py` | Offline model evaluation and real environment evaluation |
| `_vendor/lewm/` | Unmodified pinned upstream source and MIT license |

Read [design and data format](docs/design.md) for tensor alignment, losses, limitations, and how to supply expert trajectories. See [upstream attribution](docs/upstream.md) for the original paper.

## Development

```bash
pytest -q
ruff check .
ruff format --check .
```

Tests cover action round trips, episode boundaries, termination/truncation handling, CEM optimization, rollout alignment, detached outcome heads, training, reload, resume, closed-loop inference, and scripted-expert collection. `tests/test_grl.py` collects from the real GodotRocketLanding game when the `grl` package and binary are installed, and is skipped otherwise. GitHub Actions runs tests and three-environment smoke training on CPU. Optional videos: `pip install -e '.[video]'`, then add `--video runs/pendulum/episode.mp4` to `eval`.
