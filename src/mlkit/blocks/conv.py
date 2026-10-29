from dataclasses import dataclass, field
from typing import Literal

from torch import nn

from .activations import Activation, ReLU
from .norms import Norm


@dataclass(frozen=True)
class _BaseConvBlockConfig:
    out_channels: int
    norm: Norm | None = None
    activation: Activation | None = field(default_factory=ReLU)


@dataclass(frozen=True)
class ConvBlockConfig(_BaseConvBlockConfig):
    kernel_size: int = 3
    stride: int = 1
    padding: int | Literal["same", "valid"] = 0
    pool_size: int = 2
    dropout: float = 0.0
    pre_activation: bool = False

    def __post_init__(self) -> None:
        if min(self.out_channels, self.kernel_size, self.stride, self.pool_size) < 1:
            raise ValueError(
                "out_channels, kernel_size, stride and pool_size must be positive, got "
                f"{self.out_channels}, {self.kernel_size}, {self.stride}, {self.pool_size}"
            )
        if self.padding == "same" and self.stride != 1:
            raise ValueError("padding='same' cannot be combined with stride > 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass(frozen=True)
class ConvTransposeBlockConfig(_BaseConvBlockConfig):
    kernel_size: int = 4
    stride: int = 2
    padding: int = 1
    output_padding: int = 0

    def __post_init__(self) -> None:
        if min(self.out_channels, self.kernel_size, self.stride) < 1:
            raise ValueError(
                "out_channels, kernel_size and stride must be positive, got "
                f"{self.out_channels}, {self.kernel_size}, {self.stride}"
            )
        if not 0 <= self.output_padding < self.stride:
            raise ValueError(
                f"output_padding must be in [0, stride), got {self.output_padding} with stride {self.stride}"
            )


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, config: ConvBlockConfig) -> None:
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config.out_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            bias=config.norm is None or config.pre_activation,
        )
        norm_channels = in_channels if config.pre_activation else config.out_channels

        activation_layers: list[nn.Module] = []
        if config.norm is not None:
            activation_layers.append(config.norm.create(norm_channels))

        if config.activation is not None:
            activation_layers.append(config.activation.create())

        if config.dropout > 0:
            activation_layers.append(nn.Dropout(config.dropout))

        layers: list[nn.Module] = (
            [*activation_layers, conv] if config.pre_activation else [conv, *activation_layers]
        )

        if config.pool_size > 1:
            layers.append(nn.MaxPool2d(kernel_size=config.pool_size))

        super().__init__(*layers)
        self.config = config


class ConvTransposeBlock(nn.Sequential):
    def __init__(self, in_channels: int, config: ConvTransposeBlockConfig) -> None:
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=config.out_channels,
                kernel_size=config.kernel_size,
                stride=config.stride,
                padding=config.padding,
                output_padding=config.output_padding,
                bias=config.norm is None,
            )
        ]

        if config.norm is not None:
            layers.append(config.norm.create(config.out_channels))

        if config.activation is not None:
            layers.append(config.activation.create())

        super().__init__(*layers)
        self.config = config
