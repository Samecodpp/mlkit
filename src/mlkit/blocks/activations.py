from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class Activation(ABC):
    @abstractmethod
    def create(self) -> nn.Module: ...


@dataclass(frozen=True)
class ReLU(Activation):
    def create(self) -> nn.Module:
        return nn.ReLU()


@dataclass(frozen=True)
class LeakyReLU(Activation):
    negative_slope: float = 0.2

    def create(self) -> nn.Module:
        return nn.LeakyReLU(self.negative_slope)


@dataclass(frozen=True)
class Sigmoid(Activation):
    def create(self) -> nn.Module:
        return nn.Sigmoid()


@dataclass(frozen=True)
class Tanh(Activation):
    def create(self) -> nn.Module:
        return nn.Tanh()


@dataclass(frozen=True)
class SiLU(Activation):
    def create(self) -> nn.Module:
        return nn.SiLU()


@dataclass(frozen=True)
class GELU(Activation):
    def create(self) -> nn.Module:
        return nn.GELU()
