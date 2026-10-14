"""Vectorized CEM for continuous and categorical actions; receding-horizon control."""

import numpy as np
import torch
from .envs import ActionCodec


class CEM:
    """`prior` is the per-dimension mean/std of the encoded training actions: every plan starts there and
    the warm start pads with it. None is the uniform prior: mean 0 and std 1, or equal probabilities."""

    def __init__(self, cfg, codec, device="cpu", seed=0, prior=None):
        self.cfg, self.codec, self.device = cfg, codec, torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        if prior is None:
            mean = torch.full((codec.dim,), 1 / codec.dim if codec.discrete else 0.0)
            std = torch.ones(codec.dim)
        else:
            mean = torch.tensor(prior["mean"], dtype=torch.float32)
            std = torch.tensor(prior["std"], dtype=torch.float32).clamp_min(cfg.min_std)
            if mean.shape != (codec.dim,) or std.shape != (codec.dim,):
                raise ValueError("Action prior does not match the action dimension")
            if codec.discrete:  # one-hot means are category frequencies
                mean = mean.clamp_min(0.01)
                mean /= mean.sum()
        self.prior_mean, self.prior_std = mean.to(self.device), std.to(self.device)
        self.reset()

    def reset(self):
        self.mean = self.prior_mean.expand(self.cfg.horizon, -1).clone()
        self.probs = self.mean.clone()

    @torch.no_grad()
    def plan(self, score_fn):
        c, a = self.cfg, self.codec
        mean, probs = self.mean.clone(), self.probs.clone()
        std = self.prior_std.expand(c.horizon, -1).clone()
        best_score, best = -float("inf"), None
        for _ in range(c.iterations):
            if a.discrete:
                indices = torch.multinomial(
                    probs, c.population, replacement=True, generator=self.generator
                ).T
                candidates = torch.nn.functional.one_hot(indices, a.dim).float()
            else:
                noise = torch.randn(
                    c.population, c.horizon, a.dim, device=self.device, generator=self.generator
                )
                candidates = (mean + std * noise).clamp(-1, 1)
                candidates[0] = mean
            scores = score_fn(candidates)
            if scores.shape != (c.population,) or not torch.isfinite(scores).all():
                raise ValueError("Planner scores must be finite, one per candidate")
            values, idx = scores.topk(c.elites)
            elites = candidates[idx]
            if values[0].item() > best_score:
                best_score, best = values[0].item(), elites[0].clone()
            if a.discrete:
                probs = c.smoothing * probs + (1 - c.smoothing) * elites.mean(0)
                probs = probs.clamp_min(0.01)
                probs /= probs.sum(-1, keepdim=True)
            else:
                mean = c.smoothing * mean + (1 - c.smoothing) * elites.mean(0)
                std = (
                    c.smoothing * std + (1 - c.smoothing) * elites.std(0, unbiased=False)
                ).clamp_min(c.min_std)
        # Warm start the next MPC call. Never share state across episodes.
        self.mean = torch.cat([mean[1:], self.prior_mean[None]])
        self.probs = torch.cat([probs[1:], self.prior_mean[None]])
        return best[0]


class MPCPolicy:
    """Public inference API: reset(); act(uint8 RGB resized to checkpoint image_size)."""

    def __init__(self, model, cfg, action_spec, device="cpu", seed=0, action_prior=None):
        self.model, self.cfg = model.to(device).eval(), cfg
        self.device = torch.device(device)
        self.codec = ActionCodec(action_spec)
        if cfg.action_prior == "data" and action_prior is None:
            raise ValueError(
                "action_prior='data' needs the training-action statistics that train() saves in "
                "checkpoints; retrain, or plan with action_prior='uniform' (eval --action-prior uniform)"
            )
        prior = action_prior if cfg.action_prior == "data" else None
        self.cem = CEM(cfg, self.codec, device, seed, prior)
        self.reset()

    @classmethod
    def from_checkpoint(cls, path, device="cpu", seed=0, **planner_overrides):
        from .train import load_checkpoint

        model, cfg, checkpoint = load_checkpoint(path, device)
        allowed = {
            "horizon",
            "population",
            "elites",
            "iterations",
            "discount",
            "smoothing",
            "min_std",
            "candidate_chunk",
            "action_prior",
        }
        if set(planner_overrides) - allowed:
            raise ValueError("Only planner settings can be overridden during inference")
        from .config import Config

        cfg = Config(**(cfg.to_dict() | planner_overrides))
        return cls(
            model, cfg, checkpoint["action_spec"], device, seed, checkpoint.get("action_prior")
        )

    def reset(self):
        self.embeddings = []
        self.actions = []
        self.cem.reset()

    @torch.inference_mode()
    def act(self, pixels):
        pixels = np.asarray(pixels)
        expected = (self.cfg.image_size, self.cfg.image_size, 3)
        if pixels.shape != expected or pixels.dtype != np.uint8:
            raise ValueError(f"Expected uint8 RGB {expected}, got {pixels.shape}/{pixels.dtype}")
        encoded = self.model.encode(torch.as_tensor(pixels.copy(), device=self.device)[None, None])
        self.embeddings.append(encoded)
        self.embeddings = self.embeddings[-self.cfg.history :]
        z = torch.cat(self.embeddings, dim=1)
        count = len(self.embeddings) - 1
        past = (
            torch.stack(self.actions[-count:])[None]
            if count
            else torch.empty(1, 0, self.codec.dim, device=self.device)
        )

        def score(candidates):
            return torch.cat(
                [
                    self.model.imagine(z, past, chunk, self.cfg.discount)
                    for chunk in candidates.split(self.cfg.candidate_chunk)
                ]
            )

        encoded_action = self.cem.plan(score)
        self.actions.append(encoded_action)
        self.actions = self.actions[-self.cfg.history :]
        return self.codec.decode(encoded_action.cpu().numpy())
