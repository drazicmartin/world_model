import torch
from world_model.config import Config
from world_model.envs import ActionCodec
from world_model.model import WorldModel
from world_model.planner import CEM


def test_continuous_cem_finds_optimum():
    cfg = Config(horizon=4, population=256, elites=32, iterations=8)
    codec = ActionCodec({"kind": "box", "low": [-2], "high": [2], "shape": [1], "dtype": "float32"})
    cem = CEM(cfg, codec, seed=1)
    action = cem.plan(lambda candidates: -(candidates - 0.65).square().sum((1, 2)))
    assert abs(action.item() - 0.65) < 0.1
    assert -1 <= action.item() <= 1


def test_discrete_cem_uses_categories():
    cfg = Config(horizon=3, population=128, elites=16, iterations=5)
    codec = ActionCodec({"kind": "discrete", "n": 3, "start": 0})
    action = CEM(cfg, codec, seed=2).plan(lambda x: x[..., 2].sum(1))
    assert torch.equal(action, torch.tensor([0.0, 0.0, 1.0]))


def test_continuous_cem_starts_from_action_prior():
    cfg = Config(horizon=4, population=512, elites=32, iterations=1)
    codec = ActionCodec(
        {"kind": "box", "low": [-1, -1], "high": [1, 1], "shape": [2], "dtype": "f"}
    )
    cem = CEM(cfg, codec, seed=3, prior={"mean": [-0.6, 0.2], "std": [0.1, 0.0]})
    assert torch.allclose(cem.mean, torch.tensor([[-0.6, 0.2]] * 4))
    seen = []
    cem.plan(lambda candidates: seen.append(candidates.clone()) or torch.zeros(len(candidates)))
    sampled = seen[0][1:]  # candidate 0 is the mean itself
    assert torch.allclose(sampled.mean((0, 1)), torch.tensor([-0.6, 0.2]), atol=0.01)
    # std is floored at min_std, so a constant training action still gets explored
    assert torch.allclose(sampled.std((0, 1)), torch.tensor([0.1, cfg.min_std]), atol=0.01)
    assert torch.equal(cem.mean[-1], torch.tensor([-0.6, 0.2]))  # warm start pads with the prior


def test_discrete_cem_starts_from_category_frequencies():
    cfg = Config(horizon=2, population=1000, elites=10, iterations=1)
    codec = ActionCodec({"kind": "discrete", "n": 3, "start": 0})
    cem = CEM(cfg, codec, seed=4, prior={"mean": [0.8, 0.2, 0.0], "std": [0.4, 0.4, 0.0]})
    seen = []
    cem.plan(lambda candidates: seen.append(candidates.clone()) or torch.zeros(len(candidates)))
    frequencies = seen[0].mean((0, 1))
    assert abs(frequencies[0] - 0.79) < 0.05 and 0 < frequencies[2] < 0.05  # unseen: clamped at 1%
    assert torch.allclose(cem.probs[-1], torch.tensor([0.8, 0.2, 0.01]) / 1.01)


def test_rollout_action_alignment_and_survival():
    class FakeModel:
        history = 3

        def predict(self, z, actions):
            assert z.shape[1] == actions.shape[1]
            return z + actions

        def outcomes(self, z, actions, next_z):
            # Reward on transition before termination; then 50% survival.
            return next_z[..., 0], torch.zeros_like(next_z[..., 0])

    # Already executed actions 1,2; current state 13. Candidate actions 3,4,5.
    score = WorldModel.imagine(
        FakeModel(),
        torch.tensor([[[10.0], [11.0], [13.0]]]),
        torch.tensor([[[1.0], [2.0]]]),
        torch.tensor([[[3.0], [4.0], [5.0]]]),
        discount=1.0,
    )
    assert torch.allclose(score, torch.tensor([16 + 0.5 * 20 + 0.25 * 25]))
