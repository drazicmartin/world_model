"""A minimal Gymnasium environment with pixels and bounded continuous actions."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DotReach(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode="rgb_array", max_steps=100):
        if render_mode != "rgb_array":
            raise ValueError("DotReach supports rgb_array only")
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = self.np_random.uniform(-0.8, 0.8, 2).astype(np.float32)
        self.steps = 0
        return self.position.copy(), {}

    def step(self, action):
        self.position = np.clip(self.position + 0.05 * np.asarray(action), -1, 1).astype(np.float32)
        self.steps += 1
        distance = float(np.linalg.norm(self.position))
        terminated = distance < 0.08
        truncated = self.steps >= self.max_steps and not terminated
        return self.position.copy(), -distance, terminated, truncated, {"is_success": terminated}

    def render(self):
        canvas = np.full((64, 64, 3), 245, dtype=np.uint8)
        canvas[29:35, 29:35] = [40, 180, 80]
        x, y = np.rint((self.position + 1) * 29 + 2).astype(int)
        canvas[y - 2 : y + 3, x - 2 : x + 3] = [30, 70, 220]
        return canvas


def make_env(render_mode="rgb_array", **kwargs):
    return DotReach(render_mode=render_mode, **kwargs)


class DotExpert:
    """Scripted expert for `collect --policy`: steers using the true position, which is never saved."""

    def __init__(self, env, noise=0.0, seed=0):
        self.env = env.unwrapped
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, pixels):
        action = -self.env.position / 0.05 + self.rng.normal(0.0, self.noise, 2)
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def expert(env, noise=0.0, seed=0):
    return DotExpert(env, noise, seed)
