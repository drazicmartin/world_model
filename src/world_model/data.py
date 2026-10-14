"""Episode directories of .npy arrays; no reset-crossing windows."""

from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
from .envs import ActionCodec, frame, make_env, step
from .progress import Progress, stage, track

KEYS = ("pixels", "actions", "rewards", "terminated")
CACHE_EPISODES = 64  # memory-mapped episodes kept open per worker when streaming


def write_manifest(root, manifest):
    temporary = root / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(root / "manifest.json")


def collect(cfg, root, policy=None, make_policy=None):
    """Random actions by default, or `policy` (reset(), act(pixels)), or `make_policy(env)`.

    `make_policy` builds the policy once the environment exists, so a scripted expert can read simulator
    state to choose actions. Only frames, actions, rewards and flags are saved.
    """
    if policy is not None and make_policy is not None:
        raise ValueError("Pass either policy or make_policy, not both")
    if make_policy is not None:
        func = getattr(make_policy, "func", make_policy)  # functools.partial
        label = getattr(make_policy, "__name__", getattr(func, "__name__", "custom"))
    else:
        label = "random" if policy is None else type(policy).__name__
    stage(
        f"Collection | env={cfg.env_id} | episodes={cfg.episodes} | "
        f"max_steps/episode={cfg.max_steps} | policy={label} | output={root}"
    )
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    env = make_env(cfg)
    try:
        codec = ActionCodec.from_space(env.action_space)
        if make_policy is not None:
            policy = make_policy(env)
        if getattr(policy, "codec", None) is not None and policy.codec.spec != codec.spec:
            raise ValueError("Collection policy action space differs from environment")
        manifest = {
            "format": 1,
            "config": cfg.to_dict(),
            "action_spec": codec.spec,
            "policy": label,
            "episodes": [],
            "complete": False,
        }
        write_manifest(root, manifest)
        with Progress(cfg.episodes, "Collecting episodes", "episode") as progress:
            for episode in range(cfg.episodes):
                seed = cfg.seed + episode
                env.reset(seed=seed)
                env.action_space.seed(seed)
                if policy is not None:
                    policy.reset()
                pixels = [frame(env, cfg.image_size)]
                actions, rewards, terms, truncs = [], [], [], []
                for t in range(cfg.max_steps):
                    action = env.action_space.sample() if policy is None else policy.act(pixels[-1])
                    reward, term, trunc = step(env, action, cfg.action_repeat)
                    trunc = trunc or (t == cfg.max_steps - 1 and not term)
                    actions.append(codec.encode(action))
                    rewards.append(reward)
                    terms.append(term)
                    truncs.append(trunc)
                    pixels.append(frame(env, cfg.image_size))
                    progress.update(0, episode=episode + 1, step=t + 1)
                    if term or trunc:
                        break
                name = f"episode_{episode:06d}"
                folder = root / name
                folder.mkdir()
                for key, value, dtype in (
                    ("pixels", pixels, np.uint8),
                    ("actions", actions, np.float32),
                    ("rewards", rewards, np.float32),
                    ("terminated", terms, np.bool_),
                    ("truncated", truncs, np.bool_),
                ):
                    np.save(folder / f"{key}.npy", np.asarray(value, dtype=dtype))
                manifest["episodes"].append(
                    {
                        "name": name,
                        "steps": len(actions),
                        "seed": seed,
                        "return": float(sum(rewards)),
                    }
                )
                progress.update(1, last_return=f"{sum(rewards):.2f}")
        manifest["complete"] = True
        write_manifest(root, manifest)
    finally:
        env.close()
    stage(f"Collection complete | {len(manifest['episodes'])} episodes saved to {root}")
    return manifest


class Windows:
    """History windows: `history + 1` frames with the `history` aligned actions, rewards and flags.

    Streams memory-mapped episodes through a bounded per-worker cache, or after `preload()` gathers from
    contiguous in-memory copies. `dataset[i]` is one window; `dataset.__getitems__(indices)` (used by
    DataLoader) is a whole stacked batch, gathered with one vectorised index once preloaded. Both modes
    return identical values.
    """

    def __init__(self, root, episodes, history):
        self.root, self.episodes, self.history = Path(root), episodes, history
        self.counts = np.array([max(0, e["steps"] - history + 1) for e in episodes])
        self.ends = self.counts.cumsum()
        self.cache = OrderedDict()
        self.arrays = None
        if not len(self.ends) or self.ends[-1] == 0:
            raise ValueError("No full history windows; collect longer/more episodes")

    def __len__(self):
        return int(self.ends[-1])

    def open(self, name):
        return {key: np.load(self.root / name / f"{key}.npy", mmap_mode="r") for key in KEYS}

    def nbytes(self):
        """Uncompressed size of all arrays, i.e. the memory `preload()` allocates."""
        sample = self.open(next(e["name"] for e in self.episodes if e["steps"]))
        frames = sum(e["steps"] + 1 for e in self.episodes)
        steps = frames - len(self.episodes)
        per_step = sum(sample[key][0].nbytes for key in KEYS[1:])
        return frames * sample["pixels"][0].nbytes + steps * per_step

    def preload(self):
        """Copy every episode into one contiguous array per key."""
        sample = self.open(next(e["name"] for e in self.episodes if e["steps"]))
        frames = sum(e["steps"] + 1 for e in self.episodes)
        steps = frames - len(self.episodes)
        arrays = {
            key: np.empty(((frames if key == "pixels" else steps),) + v.shape[1:], v.dtype)
            for key, v in sample.items()
        }
        frame_starts, step_starts = [], []
        frame = step = 0
        for episode in track(self.episodes, "Loading dataset into memory", "episode"):
            for key, value in self.open(episode["name"]).items():
                offset = frame if key == "pixels" else step
                arrays[key][offset : offset + len(value)] = value
            windows = np.arange(max(0, episode["steps"] - self.history + 1))
            frame_starts.append(frame + windows)
            step_starts.append(step + windows)
            frame += episode["steps"] + 1
            step += episode["steps"]
        self.frame_starts = np.concatenate(frame_starts)
        self.step_starts = np.concatenate(step_starts)
        self.arrays = arrays
        self.cache.clear()

    def __getitem__(self, index):
        if self.arrays is not None:
            return {key: value[0] for key, value in self.__getitems__([index]).items()}
        episode = int(np.searchsorted(self.ends, index, side="right"))
        start = int(index - (self.ends[episode - 1] if episode else 0))
        name = self.episodes[episode]["name"]
        if name not in self.cache:
            self.cache[name] = self.open(name)
            if len(self.cache) > CACHE_EPISODES:
                self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        arrays = self.cache[name]
        return {
            key: np.array(value[start : start + self.history + (key == "pixels")])
            for key, value in arrays.items()
        }

    def __getitems__(self, indices):
        if self.arrays is None:
            items = [self[int(i)] for i in indices]
            return {key: np.stack([item[key] for item in items]) for key in KEYS}
        indices = np.asarray(indices, dtype=np.int64)
        frames = self.frame_starts[indices][:, None] + np.arange(self.history + 1)
        steps = self.step_starts[indices][:, None] + np.arange(self.history)
        return {
            key: value[frames if key == "pixels" else steps] for key, value in self.arrays.items()
        }


def split_episodes(manifest, seed, fraction):
    episodes = manifest["episodes"]
    if len(episodes) < 2:
        raise ValueError("At least two episodes required for train/validation split")
    order = np.random.default_rng(seed).permutation(len(episodes))
    nval = min(len(episodes) - 1, max(1, round(len(episodes) * fraction)))
    return ([episodes[i] for i in order[nval:]], [episodes[i] for i in order[:nval]])
