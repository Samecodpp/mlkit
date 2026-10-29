from dataclasses import dataclass, field

import torch
from torch import nn

from .activations import Activation, SiLU


class SinusoidalTimeEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")

        self.dim = dim
        two_i = torch.arange(0, dim, 2, dtype=torch.float32)
        self.register_buffer("inv_freq", 1.0 / (10000 ** (two_i / dim)), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t.float()[:, None] * self.inv_freq[None, :]
        embedding = torch.zeros(t.size(0), self.dim, device=t.device)
        embedding[:, 0::2] = torch.sin(args)
        embedding[:, 1::2] = torch.cos(args)
        return embedding


@dataclass(frozen=True)
class TimeEmbeddingConfig:
    embedding_dim: int = 64
    time_dim: int = 128
    activation: Activation = field(default_factory=SiLU)

    def __post_init__(self) -> None:
        if self.embedding_dim < 2 or self.embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be a positive even number, got {self.embedding_dim}")
        if self.time_dim < 1:
            raise ValueError(f"time_dim must be positive, got {self.time_dim}")


class TimeEmbedding(nn.Sequential):
    def __init__(self, config: TimeEmbeddingConfig) -> None:
        super().__init__(
            SinusoidalTimeEmbedding(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.time_dim),
            config.activation.create(),
            nn.Linear(config.time_dim, config.time_dim),
        )
        self.config = config
