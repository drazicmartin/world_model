"""Only this module touches the real simulator. Observations are deliberately ignored."""

import importlib
import os
import numpy as np
from PIL import Image


def make_env(cfg):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if cfg.factory:
        module, name = cfg.factory.split(":", 1)
        env = getattr(importlib.import_module(module), name)(
            render_mode="rgb_array", **cfg.env_kwargs
        )
    else:
        import gymnasium as gym

        env = gym.make(cfg.env_id, render_mode="rgb_array", **cfg.env_kwargs)
    return env


def frame(env, size):
    pixels = np.asarray(env.render())
    if pixels.ndim != 3 or pixels.shape[-1] != 3 or pixels.dtype != np.uint8:
        raise ValueError("render() must return uint8 RGB with shape (H, W, 3)")
    return np.asarray(
        Image.fromarray(pixels).resize((size, size), Image.Resampling.BILINEAR)
    ).copy()


class ActionCodec:
    """Discrete -> one hot; finite Box -> flattened [-1, 1]."""

    def __init__(self, spec):
        self.spec = spec
        self.discrete = spec["kind"] == "discrete"
        self.dim = spec["n"] if self.discrete else len(spec["low"])

    @classmethod
    def from_space(cls, space):
        from gymnasium.spaces import Box, Discrete

        if isinstance(space, Discrete):
            return cls({"kind": "discrete", "n": int(space.n), "start": int(space.start)})
        if isinstance(space, Box):
            low, high = space.low.ravel(), space.high.ravel()
            if not (np.isfinite(low).all() and np.isfinite(high).all() and (high > low).all()):
                raise ValueError("Box actions must have finite, strictly ordered bounds")
            if not np.issubdtype(space.dtype, np.floating):
                raise ValueError("Box actions must be floating point")
            return cls(
                {
                    "kind": "box",
                    "low": low.tolist(),
                    "high": high.tolist(),
                    "shape": list(space.shape),
                    "dtype": str(space.dtype),
                }
            )
        raise ValueError("Supported actions: Discrete or bounded floating-point Box; wrap others")

    def encode(self, action):
        if self.discrete:
            return np.eye(self.dim, dtype=np.float32)[int(action) - self.spec["start"]]
        low, high = np.array(self.spec["low"]), np.array(self.spec["high"])
        return (2 * (np.asarray(action).ravel() - low) / (high - low) - 1).astype(np.float32)

    def decode(self, encoded):
        encoded = np.asarray(encoded)
        if self.discrete:
            return int(encoded.argmax()) + self.spec["start"]
        low, high = np.array(self.spec["low"]), np.array(self.spec["high"])
        return (
            (low + (encoded.clip(-1, 1) + 1) * (high - low) / 2)
            .reshape(self.spec["shape"])
            .astype(self.spec["dtype"])
        )


def step(env, action, repeat, return_info=False):
    """One model transition equals repeat simulator steps; never cross a reset.

    With `return_info`, the last simulator step's info dict is returned as a fourth value."""
    total = 0.0
    for _ in range(repeat):
        _, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    if return_info:
        return total, bool(terminated), bool(truncated), info
    return total, bool(terminated), bool(truncated)
