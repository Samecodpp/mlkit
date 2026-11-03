import math
from dataclasses import dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


@dataclass(frozen=True)
class WarmupCosineLR:
    warmup_steps: int
    total_steps: int
    min_factor: float = 0.0

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.total_steps <= self.warmup_steps:
            raise ValueError(
                f"total_steps must exceed warmup_steps, got {self.total_steps} and {self.warmup_steps}"
            )
        if not 0.0 <= self.min_factor <= 1.0:
            raise ValueError(f"min_factor must be in [0, 1], got {self.min_factor}")

    def factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps

        progress = min(1.0, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_factor + (1 - self.min_factor) * cosine

    def __call__(self, optimizer: Optimizer) -> LambdaLR:
        return LambdaLR(optimizer, self.factor)
