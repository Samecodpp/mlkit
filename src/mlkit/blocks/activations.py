from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ReLU:
    def create(self) -> nn.Module:
        return nn.ReLU()


@dataclass(frozen=True)
class LeakyReLU:
    negative_slope: float = 0.2

    def create(self) -> nn.Module:
        return nn.LeakyReLU(self.negative_slope)


@dataclass(frozen=True)
class Sigmoid:
    def create(self) -> nn.Module:
        return nn.Sigmoid()


@dataclass(frozen=True)
class Tanh:
    def create(self) -> nn.Module:
        return nn.Tanh()


@dataclass(frozen=True)
class SiLU:
    def create(self) -> nn.Module:
        return nn.SiLU()


type Activation = ReLU | LeakyReLU | Sigmoid | Tanh | SiLU
