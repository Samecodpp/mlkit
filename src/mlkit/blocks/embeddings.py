from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import nn

from .activations import Activation, SiLU


def inverse_frequencies(two_i: torch.Tensor, dim: int) -> torch.Tensor:
    return 1.0 / (10000 ** (two_i / dim))


class SinusoidalEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")

        self.dim = dim
        two_i = torch.arange(0, dim, 2, dtype=torch.float32)
        self.register_buffer("inv_freq", inverse_frequencies(two_i, dim), persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        args = positions.float()[:, None] * self.inv_freq[None, :]
        embedding = torch.zeros(positions.size(0), self.dim, device=positions.device)
        embedding[:, 0::2] = torch.sin(args)
        embedding[:, 1::2] = torch.cos(args)
        return embedding


class SinusoidalPositionalEmbedding(nn.Module):
    table: torch.Tensor

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.register_buffer("table", SinusoidalEmbedding(d_model)(torch.arange(max_len)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.table[:x.size(1)]


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()

        self.table = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        position = torch.arange(x.size(1), device=x.device)
        return x + self.table(position)


@dataclass(frozen=True)
class PositionalEmbedding(ABC):
    @abstractmethod
    def create(self, max_len: int, d_model: int) -> nn.Module: ...


@dataclass(frozen=True)
class SinusoidalPositions(PositionalEmbedding):
    def create(self, max_len: int, d_model: int) -> nn.Module:
        return SinusoidalPositionalEmbedding(max_len, d_model)


@dataclass(frozen=True)
class LearnedPositions(PositionalEmbedding):
    def create(self, max_len: int, d_model: int) -> nn.Module:
        return LearnedPositionalEmbedding(max_len, d_model)


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
            SinusoidalEmbedding(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.time_dim),
            config.activation.create(),
            nn.Linear(config.time_dim, config.time_dim),
        )
        self.config = config


@dataclass(frozen=True)
class TokenEmbeddingConfig:
    d_model: int
    max_len: int
    positional: PositionalEmbedding | None = field(default_factory=LearnedPositions)
    dropout: float = 0.1
    pad_id: int = 0

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.max_len < 1:
            raise ValueError(f"max_len must be positive, got {self.max_len}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.pad_id < 0:
            raise ValueError(f"pad_id must be non-negative, got {self.pad_id}")


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, config: TokenEmbeddingConfig) -> None:
        if config.pad_id >= vocab_size:
            raise ValueError(f"pad_id must be less than vocab_size, got {config.pad_id} and {vocab_size}")

        super().__init__()
        self.tokens = nn.Embedding(vocab_size, config.d_model, padding_idx=config.pad_id)

        self.positions: nn.Module | None = None
        if config.positional is not None:
            self.positions = config.positional.create(config.max_len, config.d_model)

        self.dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.tokens(ids)

        if self.positions is not None:
            x = self.positions(x)

        out: torch.Tensor = self.dropout(x)
        return out

