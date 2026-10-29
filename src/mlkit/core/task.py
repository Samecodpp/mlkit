from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Literal

import torch
from torch import nn

from .batch import Batch, first_tensor

type OptimizerFactory = Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer]
type Split = Literal["train", "val", "test"]
type StepOutput = dict[str, torch.Tensor | float]


class Task[BatchT: Batch](nn.Module, ABC):
    grad_in_eval = False

    @abstractmethod
    def training_step(self, batch: BatchT) -> StepOutput: ...

    def validation_step(self, batch: BatchT) -> StepOutput:
        raise NotImplementedError(f"{type(self).__name__} does not implement validation_step")

    def reset_metrics(self, split: Split) -> None: ...

    def compute_metrics(self, split: Split) -> dict[str, float]:
        return {}

    def batch_size(self, batch: BatchT) -> int:
        tensor = first_tensor(batch)
        if tensor is None or tensor.dim() == 0:
            raise TypeError(
                f"Cannot infer batch size from {type(batch).__name__}: "
                f"override {type(self).__name__}.batch_size"
            )
        return tensor.size(0)

    @staticmethod
    def step(optimizer: torch.optim.Optimizer, loss: torch.Tensor) -> None:
        optimizer.zero_grad()
        torch.autograd.backward(loss)
        optimizer.step()
