from dataclasses import dataclass, field, replace

import torch
from torch import nn

from .activations import Activation, SiLU
from .conv import ConvBlock, ConvBlockConfig
from .norms import GroupNorm, Norm


@dataclass(frozen=True)
class UNetBlockConfig:
    out_channels: int
    activation: Activation = field(default_factory=SiLU)
    norm: Norm | None = field(default_factory=GroupNorm)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.out_channels < 1:
            raise ValueError(f"out_channels must be positive, got {self.out_channels}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


class UNetBlock(nn.Module):
    def __init__(self, in_channels: int, time_dim: int | None, config: UNetBlockConfig) -> None:
        super().__init__()
        conv_config = ConvBlockConfig(
            config.out_channels,
            norm=config.norm,
            activation=config.activation,
            padding=1,
            pool_size=1,
            pre_activation=True,
        )
        self.conv1 = ConvBlock(in_channels, conv_config)
        self.time: nn.Sequential | None = None
        if time_dim is not None:
            self.time = nn.Sequential(config.activation.create(), nn.Linear(time_dim, config.out_channels))
        self.conv2 = ConvBlock(config.out_channels, replace(conv_config, dropout=config.dropout))
        self.shortcut: nn.Module = (
            nn.Identity()
            if in_channels == config.out_channels
            else nn.Conv2d(in_channels, config.out_channels, kernel_size=1)
        )
        self.config = config

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        h: torch.Tensor = self.conv1(x)
        if self.time is not None:
            if t_emb is None:
                raise ValueError("t_emb is required: the block was built with time_dim")
            h = h + self.time(t_emb)[:, :, None, None]
        elif t_emb is not None:
            raise ValueError("t_emb was given, but the block was built without time_dim")
        h = self.conv2(h)
        shortcut: torch.Tensor = self.shortcut(x)
        return h + shortcut
