from dataclasses import dataclass

import torch
from torch import nn

from ..blocks.conv import ConvBlock, ConvBlockConfig
from ..blocks.linear import LinearBlock, LinearBlockConfig


@dataclass(frozen=True)
class CNNConfig:
    in_channels: int
    input_size: tuple[int, int]
    conv_configs: tuple[ConvBlockConfig, ...]
    linear_configs: tuple[LinearBlockConfig, ...]


class CNNClassifier(nn.Sequential):
    def __init__(self, config: CNNConfig) -> None:
        layers: list[nn.Module] = []
        in_channels = config.in_channels
        for conf in config.conv_configs:
            layers.append(ConvBlock(in_channels=in_channels, config=conf))
            in_channels = conf.out_channels

        conv = nn.Sequential(*layers)
        in_features = self._infer_flat_dim(
            conv=conv,
            in_channels=config.in_channels,
            input_size=config.input_size,
        )
        layers.append(nn.Flatten())
        for conf in config.linear_configs:
            layers.append(LinearBlock(in_features=in_features, config=conf))
            in_features = conf.out_features

        super().__init__(*layers)
        self.config = config

    @staticmethod
    def _infer_flat_dim(
        conv: nn.Sequential,
        in_channels: int,
        input_size: tuple[int, int],
    ) -> int:
        dummy = torch.zeros(1, in_channels, *input_size)
        was_training = conv.training
        conv.eval()

        with torch.no_grad():
            out: torch.Tensor = conv(dummy)

        conv.train(was_training)
        return out.numel()

