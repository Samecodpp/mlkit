from dataclasses import dataclass, field

from torch import nn

from .activations import Activation, ReLU
from .norms import Norm


@dataclass(frozen=True)
class LinearBlockConfig:
    out_features: int
    norm: Norm | None = None
    activation: Activation | None = field(default_factory=ReLU)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.out_features < 1:
            raise ValueError(f"out_features must be positive, got {self.out_features}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


class LinearBlock(nn.Sequential):
    def __init__(self, in_features: int, config: LinearBlockConfig) -> None:
        layers: list[nn.Module] = [
            nn.Linear(
                in_features=in_features,
                out_features=config.out_features,
                bias=config.norm is None,
            )
        ]

        if config.norm is not None:
            layers.append(config.norm.create(config.out_features, dim=1))

        if config.activation is not None:
            layers.append(config.activation.create())

        if config.dropout > 0:
            layers.append(nn.Dropout(config.dropout))

        super().__init__(*layers)
        self.config = config
