"""Episode directories of memory-mapped .npy arrays; no reset-crossing windows."""

from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
from .envs import ActionCodec, frame, make_env, step
from .progress import Progress, stage


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
    """Lazy mmap loading with a per-worker bounded episode cache."""

    def __init__(self, root, episodes, history):
        self.root, self.episodes, self.history = Path(root), episodes, history
        self.counts = np.array([max(0, e["steps"] - history + 1) for e in episodes])
        self.ends = self.counts.cumsum()
        self.cache = OrderedDict()
        if not len(self.ends) or self.ends[-1] == 0:
            raise ValueError("No full history windows; collect longer/more episodes")

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        episode = int(np.searchsorted(self.ends, index, side="right"))
        start = int(index - (self.ends[episode - 1] if episode else 0))
        name = self.episodes[episode]["name"]
        if name not in self.cache:
            self.cache[name] = {
                key: np.load(self.root / name / f"{key}.npy", mmap_mode="r")
                for key in ("pixels", "actions", "rewards", "terminated")
            }
            if len(self.cache) > 8:
                self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        arrays = self.cache[name]
        return {
            key: np.array(value[start : start + self.history + (key == "pixels")])
            for key, value in arrays.items()
        }


def split_episodes(manifest, seed, fraction):
    episodes = manifest["episodes"]
    if len(episodes) < 2:
        raise ValueError("At least two episodes required for train/validation split")
    order = np.random.default_rng(seed).permutation(len(episodes))
    nval = min(len(episodes) - 1, max(1, round(len(episodes) * fraction)))
    return ([episodes[i] for i in order[nval:]], [episodes[i] for i in order[:nval]])
