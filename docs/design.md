# Design and data contract

## A transition has one meaning everywhere

For transition `t`, the stored tuple is `(pixels[t], actions[t], rewards[t], terminated[t], truncated[t], pixels[t+1])`. Render once after reset, then after every action (or configured action repeat). `pixels` has length `T+1`; every other array has length `T`. Terminal frames are captured before reset. A separate folder is used for each episode, so no training window crosses a reset.

`action_repeat=k` holds the action for up to k simulator steps, stops immediately on termination/truncation, and sums those rewards. Discounting is per resulting model transition, not per underlying simulator tick. Collection `max_steps` is also counted in model transitions. Defaults use repeat 1.

## Data layout

```
data/pendulum/
  manifest.json
  episode_000000/
    pixels.npy          # uint8 [T+1,H,W,3]
    actions.npy         # float32 [T,A], normalized Box or one-hot Discrete
    rewards.npy         # float32 [T], unnormalized reward sums
    terminated.npy      # bool [T]
    truncated.npy       # bool [T]
  episode_000001/
    ...
```

The manifest includes format version, full collection config, action bounds/categories, episode lengths/seeds/returns, and a completion flag. It is atomically finalized after collection. Interrupted datasets remain incomplete and cannot train accidentally. Episode files are uncompressed for efficient mmap access. Training copies the arrays into memory when they total at most `preload_max_gb` (default 4) and gathers each batch with one vectorised index (`Windows.__getitems__`), so no loader workers are needed; larger datasets are streamed from the memory-mapped files through a 64-episode per-worker cache. Both paths produce identical batches, so `preload_max_gb` may change on resume. Frames stay uint8 on disk and are normalized to `[-1,1]` only for encoding.

Uncompressed 64×64 RGB requires roughly 1.23 GB per 100,000 frames. Collection holds one episode's frames in memory. Very long episodes should be capped or use a streaming collector extension. DataLoader `workers` default to zero for portable execution and only apply when streaming; increase them after checking storage throughput.

## Training

Whole episodes are deterministically split before constructing windows. This avoids train/validation leakage from adjacent windows of the same episode. A batch contains `history+1` frames and `history` aligned actions/rewards/flags. Batch size must be at least two for a meaningful cross-sample SIGReg statistic.

1. ViT encodes pixels; the upstream projector yields `z_0 ... z_L`.
2. Upstream action-conditioned causal predictor maps `(z_0 ... z_{L-1}, a_0 ... a_{L-1})` to predictions for `z_1 ... z_L`.
3. The representation objective is the upstream MSE + `sigreg_weight × SIGReg`. Targets are **not detached**, matching upstream training.
4. A small outcome head takes `(z_t, a_t, predicted z_{t+1})`. It predicts normalized reward and a termination logit. Features are detached, so reward/termination supervision cannot change LeWM's representation objective.
5. Total loss adds outcome MSE and binary cross entropy. Reward normalization uses training episodes only and is saved in model buffers. Global gradient clipping is 1.0.

No ground-truth state, future image reconstruction, pretrained image backbone, or teacher/expert state is used by the model. A single model is trained for a single environment and action specification. The outcome head is a practical adapter for reward-based RL; it is not claimed as part of the original LeWM paper.

## Inference alignment

With L observed embeddings, there are exactly L−1 previously executed actions. The first candidate action is appended to that history and predicts the next state. Predicted states and candidate actions are then appended autoregressively. Each predictor call keeps at most `history` observations and the matching actions. No fake past actions or repeated initial frames are introduced on reset; the causal predictor starts with one frame.

The transition reward is counted before multiplying survival by `1-sigmoid(termination_logit)`. The objective is expected discounted reward within a finite horizon. There is no terminal value bootstrap. Truncation ends real episodes but is not modeled as an absorbing event. The planner does not explicitly know remaining time to a TimeLimit.

## Evaluating honestly

`eval-model` reuses the checkpoint's held-out episode split on the original immutable dataset. It reports multi-step latent MSE, raw reward RMSE, and termination Brier error under recorded actions. Lower latent MSE alone is not proof of a useful model: collapsed representations also predict easily. Inspect `latent_std`, reward error, termination error, and real control returns together.

`eval` reports every seed's return/length/termination and action latency, aggregate mean, population standard deviation, and estimated standard error. The default evaluation seed block differs from default collection seeds; if changing seeds, ensure the blocks do not overlap. Use identical evaluation seeds for random and MPC comparisons, multiple independent training seeds, and enough test episodes before claiming improvement.

Offline validation measures behavior covered by the dataset. MPC can exploit prediction errors outside that coverage. Three frames may not make an arbitrary visual environment Markov; camera occlusion, hidden goals, and velocity ambiguity can require more history or additional observations. This adapter intentionally uses only RGB inputs.

## Better data

Uniform random actions are a convenient bootstrap, not a universal exploration strategy. MountainCarContinuous is especially challenging because reward mostly penalizes action until the goal is reached. A model trained without successful states cannot reliably plan toward that unseen outcome. CartPole's constant reward needs a calibrated failure model. Pendulum benefits from broad angle and velocity coverage.

You can collect with a trained checkpoint using `collect --checkpoint ...` into a new dataset, or with a scripted expert using `collect --policy MODULE:FACTORY [--policy-kwargs JSON]`. In Python, pass a policy object to `collect(cfg, path, policy)` implementing `reset()` and `act(pixels)` (plus `codec`, checked when present), or `collect(cfg, path, make_policy=factory)` to build it from the environment once that exists, which is what `--policy` does. An expert collector may access simulator state to choose actions, but only its RGB frames/actions/rewards enter the saved training data; the manifest records which policy collected them. `world_model.example_env:expert` and GodotRocketLanding's `grl.policies:demonstrator` are examples. To import your own trajectories, write the array schema above and a manifest matching a freshly collected sample. Ensure all actions are encoded through `ActionCodec`, bounds and rendering match, each episode ends at its true boundary, and completion is set only after every array is written.

For mixed exploration/expert datasets, create a new dataset directory, copy whole episode folders under unique names, concatenate their metadata into one new manifest, and keep configuration/action specs identical. Do not append data in place to a dataset used for checkpoint resume: its split and normalization would change. Then train a new run. Iterative dataset aggregation is deliberately explicit rather than an automatic self-improvement loop.

## Reproducibility scope

Core dependency versions and upstream source are pinned; a tested Linux CPU dependency snapshot is included in `requirements-tested-cpu.txt`. The checkpoint embeds config, action schema, upstream revision, dataset manifest fingerprint, and RNG states. Exact reproducibility requires unchanged array files, matching dependencies, hardware/backend and environment code; the manifest fingerprint is not a content hash of all frame arrays. GPU kernels and some custom simulators may be nondeterministic.

Checkpoints use `state_dict` plus tensors/primitives and `weights_only=True` loading. They are not upstream pickled `_object.ckpt` files. Output directories are explicit, avoiding dependence on a user's home/cache paths. No training artifacts are automatically uploaded to GitHub.
