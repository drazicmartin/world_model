"""Small, explicit JSON configs; unknown fields are errors."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class Config:
    env_id: str = "Pendulum-v1"
    factory: str | None = None
    env_kwargs: dict = field(default_factory=dict)
    image_size: int = 64
    history: int = 3
    dim: int = 192
    encoder_depth: int = 12
    encoder_heads: int = 3
    patch_size: int = 8
    predictor_depth: int = 6
    predictor_heads: int = 8
    predictor_head_dim: int = 32
    mlp_dim: int = 768
    sigreg_weight: float = 0.09
    sigreg_projections: int = 256
    batch_size: int = 64
    epochs: int = 50
    lr: float = 0.00005
    weight_decay: float = 0.001
    workers: int = 0
    preload_max_gb: float = 4.0
    seed: int = 42
    episodes: int = 200
    max_steps: int = 500
    action_repeat: int = 1
    validation_fraction: float = 0.1
    horizon: int = 15
    population: int = 256
    elites: int = 32
    iterations: int = 5
    discount: float = 0.99
    smoothing: float = 0.1
    min_std: float = 0.05
    candidate_chunk: int = 64

    def __post_init__(self):
        for name in (
            "image_size",
            "history",
            "dim",
            "encoder_depth",
            "encoder_heads",
            "patch_size",
            "predictor_depth",
            "predictor_heads",
            "predictor_head_dim",
            "mlp_dim",
            "sigreg_projections",
            "batch_size",
            "epochs",
            "episodes",
            "max_steps",
            "action_repeat",
            "horizon",
            "population",
            "elites",
            "iterations",
            "candidate_chunk",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.image_size % self.patch_size or self.dim % self.encoder_heads:
            raise ValueError("image_size / patch_size and dim / encoder_heads must be integral")
        if not 0 < self.validation_fraction < 1 or self.elites > self.population:
            raise ValueError("invalid validation_fraction or elites > population")
        if not 0 < self.discount <= 1 or not 0 <= self.smoothing < 1 or self.min_std <= 0:
            raise ValueError("invalid CEM settings")
        if self.workers < 0 or self.preload_max_gb < 0 or self.lr <= 0 or self.sigreg_weight < 0:
            raise ValueError("invalid optimizer / loader settings")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)
