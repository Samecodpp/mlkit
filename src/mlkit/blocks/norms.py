from dataclasses import dataclass
from typing import Literal

from torch import nn

type NormDim = Literal[1, 2]


@dataclass(frozen=True)
class BatchNorm:
    def create(self, num_features: int, dim: NormDim = 2) -> nn.Module:
        return nn.BatchNorm1d(num_features) if dim == 1 else nn.BatchNorm2d(num_features)


@dataclass(frozen=True)
class GroupNorm:
    num_groups: int = 8

    def __post_init__(self) -> None:
        if self.num_groups < 1:
            raise ValueError(f"num_groups must be positive, got {self.num_groups}")

    def create(self, num_features: int, dim: NormDim = 2) -> nn.Module:
        if num_features % self.num_groups != 0:
            raise ValueError(
                f"num_features must be divisible by num_groups, got {num_features} and {self.num_groups}"
            )
        return nn.GroupNorm(self.num_groups, num_features)


@dataclass(frozen=True)
class LayerNorm:
    def create(self, num_features: int, dim: NormDim = 2) -> nn.Module:
        return nn.LayerNorm(num_features) if dim == 1 else nn.GroupNorm(1, num_features)


type Norm = BatchNorm | GroupNorm | LayerNorm
