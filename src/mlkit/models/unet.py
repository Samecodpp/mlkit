from dataclasses import dataclass, field

import torch
from torch import nn

from ..blocks.conv import ConvBlock, ConvBlockConfig, ConvTransposeBlock, ConvTransposeBlockConfig
from ..blocks.embeddings import TimeEmbedding, TimeEmbeddingConfig
from ..blocks.unet import UNetBlock, UNetBlockConfig


@dataclass(frozen=True)
class UNetConfig:
    in_channels: int
    blocks: tuple[UNetBlockConfig, ...]
    input_size: tuple[int, int] | None = None
    time: TimeEmbeddingConfig | None = field(default_factory=TimeEmbeddingConfig)

    def __post_init__(self) -> None:
        if self.in_channels < 1:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")
        if len(self.blocks) < 2:
            raise ValueError(f"blocks must have at least 2 levels, got {len(self.blocks)}")

        factor = 2 ** (len(self.blocks) - 1)
        if self.input_size is not None and any(size % factor != 0 for size in self.input_size):
            raise ValueError(
                f"input_size {self.input_size} must be divisible by {factor} "
                f"for {len(self.blocks)} levels ({len(self.blocks) - 1} downsamplings)"
            )


class UNet(nn.Module):
    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        time_dim = None if config.time is None else config.time.time_dim
        self.time = None if config.time is None else TimeEmbedding(config.time)

        top = config.blocks[0]
        self.stem = ConvBlock(
            in_channels=config.in_channels,
            config=ConvBlockConfig(
                out_channels=top.out_channels,
                padding=1,
                pool_size=1,
                activation=None,
            ),
        )

        self.encoder = nn.ModuleList()
        self.downs = nn.ModuleList()
        in_channels = top.out_channels
        for block_config in config.blocks[:-1]:
            self.encoder.append(
                UNetBlock(
                    in_channels=in_channels,
                    time_dim=time_dim,
                    config=block_config
                )
            )
            self.downs.append(
                ConvBlock(
                    in_channels=block_config.out_channels,
                    config=ConvBlockConfig(
                        out_channels=block_config.out_channels,
                        stride=2,
                        padding=1,
                        pool_size=1,
                        activation=None,
                    ),
                )
            )
            in_channels = block_config.out_channels

        self.bottleneck = UNetBlock(in_channels, time_dim, config.blocks[-1])

        self.ups = nn.ModuleList()
        self.decoder = nn.ModuleList()
        in_channels = config.blocks[-1].out_channels
        for block_config in reversed(config.blocks[:-1]):
            self.ups.append(
                ConvTransposeBlock(
                    in_channels=in_channels,
                    config=ConvTransposeBlockConfig(
                        out_channels=block_config.out_channels,
                        activation=None,
                    ),
                )
            )
            self.decoder.append(
                UNetBlock(
                    in_channels=2 * block_config.out_channels,
                    time_dim=time_dim,
                    config=block_config,
                )
            )
            in_channels = block_config.out_channels

        self.head = ConvBlock(
            in_channels=top.out_channels,
            config=ConvBlockConfig(
                config.in_channels,
                kernel_size=1,
                pool_size=1,
                norm=top.norm,
                activation=top.activation,
                pre_activation=True,
            ),
        )
        self.config = config

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        t_emb = self._embed_time(t)
        h: torch.Tensor = self.stem(x)

        skips: list[torch.Tensor] = []
        for block, down in zip(self.encoder, self.downs):
            h = block(h, t_emb)
            skips.append(h)
            h = down(h)

        h = self.bottleneck(h, t_emb)

        for up, block in zip(self.ups, self.decoder):
            h = torch.cat([up(h), skips.pop()], dim=1)
            h = block(h, t_emb)

        out: torch.Tensor = self.head(h)
        return out

    def _embed_time(self, t: torch.Tensor | None) -> torch.Tensor | None:
        if self.time is None:
            if t is not None:
                raise ValueError("t was given, but the model was built without time embedding")
            return None
        if t is None:
            raise ValueError("t is required: the model was built with time embedding")
        t_emb: torch.Tensor = self.time(t)
        return t_emb
