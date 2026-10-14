# Validation performed

Executed on 2026-10-06 using Python 3.12, PyTorch 2.6.0+cpu, Gymnasium 1.1.1, and Transformers 4.49.0. No GPU was available. See `requirements-tested-cpu.txt` for the full dependency snapshot.

## Tests

`pytest -q`: **11 passed**. Tests exercise bounded/multidimensional Box and nonzero-start Discrete action round trips, action repeat at truncation, real Pendulum image collection, nonoverlapping episode splits, continuous and categorical CEM optimization, action/latent rollout timing, termination survival weighting, detached outcome heads, a real optimizer run, checkpoint reload/resume, closed-loop inference, and custom-factory CLI collection from a different working directory.

`ruff check .` and `ruff format --check .` are also required by CI. Vendored upstream files are intentionally excluded from formatting to preserve their exact source bytes.

## Actual smoke training and evaluation

Command: `python scripts/smoke.py --output runs/verified_smoke`.

Each environment used 8 random collection episodes, at most 40 transitions per episode, two epochs, 32×32 RGB, a one-layer 48-dimensional ViT and one-layer predictor. CEM used horizon 3, 16 candidates, four elites, and two iterations. Evaluation used the two fresh seeds 100000 and 100001 with the same 40-step cap. These runs validate execution, **not task solving or statistically meaningful policy improvement**.

| Environment | Mean MPC return | Mean random return |
|---|---:|---:|
| CartPole-v1 | 9.50 | 24.50 |
| Pendulum-v1 | -211.01 | -299.99 |
| MountainCarContinuous-v0 | -3.88 | -1.57 |

Full per-episode returns, rollout errors, planner settings, and measured latencies are in [`smoke-results.json`](smoke-results.json). The environment horizon caps differ from full benchmarks. MPC was worse than random on CartPole and MountainCar in these tiny runs; the Pendulum difference across two seeds is not sufficient evidence of reliable improvement. Low held-out latent variation in these short runs also warrants longer training and representation diagnostics.

The first smoke attempt exposed an incomplete manifest during final model evaluation. Collection was changed to finalize the manifest atomically, and the complete three-environment smoke run was repeated successfully. The environment emitted CPU-info/NNPACK fallback warnings; computation completed. Pytest also reports a pygame/pkg_resources deprecation warning.

The bundled DotReach custom environment also completed one epoch of training and a reloaded MPC evaluation. Its factory is installed with the package, so CLI collection does not depend on the current working directory.

## Not yet established

- Long-run performance of the default model recipes or solved-task scores.
- GPU throughput, CUDA resume determinism, or multi-device training (not implemented).
- Generalization to an arbitrary custom simulator or unseen visual distribution.
- Reliability of long imagined horizons outside the collection distribution.
- Cross-platform execution beyond the tested Linux CPU environment.

Datasets and checkpoints were generated locally and are excluded from source control. Reproduce them with the smoke script, then use the default recipes and larger datasets for actual experiments. GitHub Actions is configured to run the same smoke pipeline; its remote status is independent of the local results above.

## Progress reporting update

The progress-reporting change passed **14 tests**, lint/format checks, and the full three-environment smoke pipeline. Tests cover disabled output, accurate partial counts, and interrupted-stage reporting in addition to the existing model tests. Collection, training/validation batches, policy evaluation, and held-out model evaluation were exercised with redirected status logs; the inference example was also checked in a terminal. Windows PowerShell was not available for local testing; bars use ASCII characters and tqdm's terminal handling.

## GodotRocketLanding integration

Executed on 2026-10-14 using Python 3.11, PyTorch 2.6.0 (CPU only), Gymnasium 1.1.1 and GodotRocketLanding at commit `bdf492d` (Godot 4.7.2 Linux binary, headless, 4-core CPU, no GPU).

- `pytest -q`: **19 passed**, including both real-game tests in `tests/test_grl.py`; with `grl` absent they are skipped (17 passed, 1 skipped). `ruff check .` and `ruff format --check .` pass. `scripts/smoke.py` reproduced the three-environment returns above exactly.
- `configs/smoke_grl_landing.json`: random and `--policy grl.policies:demonstrator` collection, training, CEM and random evaluation and `eval-model` all completed through the CLI.

A larger CPU run used `configs/grl_landing.json` with 60 episodes, a smaller model (`dim` 96, 4 encoder and 3 predictor layers, `mlp_dim` 384, 128 SIGReg projections, batch 32, `lr` 0.0002), 6 epochs and CEM with 128 candidates, 16 elites and 3 iterations:

- Collection with `--policy grl.policies:demonstrator --policy-kwargs '{"noise": 0.1}'` took 70 s: 21,285 transitions, 46 landings, 12 time limits and 2 crashes.
- Training took about 1.8 minutes per epoch. The best held-out metric (0.369) came at epoch 3; validation loss rose afterwards, so 54 training episodes are too few for longer training.
- `eval-model --horizon 10` (100 held-out windows): latent MSE grew from 0.016 to 0.340 over the 10 steps; reward RMSE was 0.31–0.38 for steps 1–9 and 10.0 at step 10, where a window includes a ±100 terminal reward; termination Brier error stayed at or below 0.01.
- Closed-loop evaluation on seeds 100000–100003 (500-step cap):

| Seed | MPC return (steps) | Random return (steps) |
|---|---:|---:|
| 100000 | -9,899 (500, time limit) | -21,272 (500, time limit) |
| 100001 | -10,683 (500, time limit) | -22,164 (500, time limit) |
| 100002 | -965 (182, terminated) | -321 (73, terminated) |
| 100003 | -238 (58, terminated) | -233 (57, terminated) |
| Mean | -5,446 | -10,997 |

MPC roughly halved the penalty where random actions fly away from the pad, but it did not land within 500 steps; `eval` does not record whether a terminated episode crashed or landed. Four seeds are not evidence of reliable improvement, and landing with MPC is not established. Planning took about 3 s per step. The headless game keeps one core busy while it waits for the next step, which slowed the planner on this 4-core machine.

## In-memory batch loading

Loader throughput on the 60-episode GodotRocketLanding dataset above (19,091 training windows, 237 MB, batch 64, no model, 4-core CPU): the previous per-window loading with an eight-episode cache gave 15 batches/s (52 with four workers), because shuffled batches reopened most episodes. Streaming with the 64-episode cache gives 105 batches/s (156 with four workers); preloading takes 2.1 s and then gives 1,010 batches/s without workers. Tests check that preloaded and streamed batches are identical and that both train to identical weights; `scripts/smoke.py` reproduced the three-environment returns above exactly, and `pytest -q` passed (21 tests).
