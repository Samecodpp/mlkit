from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


@dataclass(frozen=True)
class ConvBlockConfig:
    out_channels: int
    kernel_size: int = 3
    pool_size: int = 2
    padding: int | Literal["same", "valid"] = 0
    stride: int = 1
    batch_norm: bool = False

    def __post_init__(self) -> None:
        if self.padding == "same" and self.stride != 1:
            raise ValueError("padding='same' cannot be combined with stride > 1")


@dataclass(frozen=True)
class CNNConfig:
    in_channels: int
    input_size: tuple[int, int]
    num_classes: int
    conv_blocks: tuple[ConvBlockConfig, ...]
    hidden_sizes: tuple[int, ...] = ()
    dropout: float = 0.0


class CNNClassifier(nn.Module):
    def __init__(self, config: CNNConfig) -> None:
        super().__init__()
        self.config = config
        self.conv_layers = self._build_conv_layers(config)
        flat_dim = self._infer_flat_dim()
        self.classifier = self._build_classifier(flat_dim=flat_dim)

    def _build_conv_layers(self, config: CNNConfig) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_channels = config.in_channels

        for block in config.conv_blocks:
            layers.append(self._conv_block(in_channels=in_channels, config=block))
            in_channels = block.out_channels

        return nn.Sequential(*layers)

    def _build_classifier(self, flat_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Flatten()]
        in_features = flat_dim

        for hidden in self.config.hidden_sizes:
            layers.append(nn.Linear(in_features=in_features, out_features=hidden))
            layers.append(nn.ReLU())
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            in_features = hidden

        layers.append(nn.Linear(in_features=in_features, out_features=self.config.num_classes))
        return nn.Sequential(*layers)

    def _infer_flat_dim(self) -> int:
        dummy = torch.zeros(1, self.config.in_channels, *self.config.input_size)
        was_training = self.conv_layers.training
        self.conv_layers.eval()

        with torch.no_grad():
            out: torch.Tensor = self.conv_layers(dummy)

        self.conv_layers.train(was_training)
        return out.numel()

    @staticmethod
    def _conv_block(in_channels: int, config: ConvBlockConfig) -> nn.Sequential:
        block: list[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=config.out_channels,
                kernel_size=config.kernel_size,
                stride=config.stride,
                padding=config.padding,
                bias=not config.batch_norm,
            )
        ]

        if config.batch_norm:
            block.append(nn.BatchNorm2d(num_features=config.out_channels))

        block.append(nn.ReLU())

        if config.pool_size > 1:
            block.append(nn.MaxPool2d(kernel_size=config.pool_size))

        return nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.classifier(self.conv_layers(x))
        return logits
