from dataclasses import dataclass, field

import torch
from torch import nn

from .activations import GELU, Activation
from .attention import SelfAttention, SelfAttentionConfig
from .linear import LinearBlock, LinearBlockConfig
from .norms import LayerNorm


@dataclass(frozen=True)
class TransformerEncoderBlockConfig:
    ff_dim: int
    attention: SelfAttentionConfig
    activation: Activation = field(default_factory=GELU)
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.ff_dim < 1:
            raise ValueError(f"ff_dim must be positive, got {self.ff_dim}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, config: TransformerEncoderBlockConfig) -> None:
        super().__init__()

        self.norm1 = LayerNorm().create(num_features=d_model, dim=1)
        self.attention = SelfAttention(d_model, config.attention)
        self.norm2 = LayerNorm().create(num_features=d_model, dim=1)
        self.ff = nn.Sequential(
            LinearBlock(
                d_model,
                LinearBlockConfig(
                    out_features=config.ff_dim,
                    activation=config.activation,
                    dropout=config.dropout,
                ),
            ),
            LinearBlock(
                config.ff_dim,
                LinearBlockConfig(
                    out_features=d_model,
                    activation=None,
                ),
            ),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        h: torch.Tensor
        weights: torch.Tensor
        h, weights = self.attention(self.norm1(x), mask)
        x = x + self.dropout(h)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, weights
