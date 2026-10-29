from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import nn
from torchmetrics import MetricCollection

from ..core.batch import Unlabeled
from ..core.task import OptimizerFactory, Split, StepOutput, Task


class Schedule(ABC):
    @abstractmethod
    def betas(self, steps: int) -> torch.Tensor: ...


@dataclass(frozen=True)
class LinearSchedule(Schedule):
    beta_start: float
    beta_end: float

    def __post_init__(self) -> None:
        if not 0.0 < self.beta_start < 1.0:
            raise ValueError(f"beta_start must be in (0, 1), got {self.beta_start}")
        if not 0.0 < self.beta_end < 1.0:
            raise ValueError(f"beta_end must be in (0, 1), got {self.beta_end}")
        if self.beta_start > self.beta_end:
            raise ValueError(
                f"beta_start must not exceed beta_end, got {self.beta_start} and {self.beta_end}"
            )

    def betas(self, steps: int) -> torch.Tensor:
        return torch.linspace(
            start=self.beta_start,
            end=self.beta_end,
            steps=steps,
        )


@dataclass(frozen=True)
class CosineSchedule(Schedule):
    s: float = 0.008
    max_beta: float = 0.999

    def __post_init__(self) -> None:
        if self.s < 0:
            raise ValueError(f"s must be non-negative, got {self.s}")
        if not 0.0 < self.max_beta < 1.0:
            raise ValueError(f"max_beta must be in (0, 1), got {self.max_beta}")

    def betas(self, steps: int) -> torch.Tensor:
        t = torch.arange(steps + 1, dtype=torch.float64) / steps
        f = torch.cos((t + self.s) / (1 + self.s) * torch.pi / 2) ** 2
        alpha_bars = f / f[0]
        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
        return betas.clamp(max=self.max_beta).float()


class NoiseSchedule(nn.Module):
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    alpha_bars_prev: torch.Tensor
    posterior_variance: torch.Tensor

    def __init__(self, steps: int, schedule: Schedule) -> None:
        super().__init__()

        if steps < 1:
            raise ValueError(f"steps must be positive, got {steps}")

        self.schedule = schedule
        self.steps = steps

        betas = schedule.betas(steps)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])

        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alpha_bars", alpha_bars, persistent=False)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev, persistent=False)
        self.register_buffer(
            "posterior_variance",
            (1 - alpha_bars_prev) / (1 - alpha_bars) * betas,
            persistent=False,
        )

    def sample_steps(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.randint(high=self.steps, size=(batch_size,), device=device)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a_bars = self.alpha_bars[t, None, None, None]
        return torch.sqrt(a_bars) * x0 + torch.sqrt(1 - a_bars) * noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a_bars = self.alpha_bars[t, None, None, None]
        return (x_t - torch.sqrt(1 - a_bars) * noise) / torch.sqrt(a_bars)


class DiffusionTask(Task[Unlabeled]):
    def __init__(
        self,
        model: nn.Module,
        optimizer: OptimizerFactory,
        schedule: NoiseSchedule,
        image_shape: tuple[int, int, int],
        variance: Literal["beta", "posterior"] = "posterior",
        metrics: MetricCollection | None = None,
        loss_fn: nn.Module | None = None
    ) -> None:
        super().__init__()
        if variance not in ("beta", "posterior"):
            raise ValueError(f"variance must be 'beta' or 'posterior', got {variance!r}")

        self.model = model
        self.opt = optimizer(model.parameters())
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        self.schedule = schedule
        self.image_shape = image_shape
        self.variance = variance
        self.val_metrics = None if metrics is None else metrics.clone()
        self.test_metrics = None if metrics is None else metrics.clone()
        self._split: Split = "train"

    def _metrics(self, split: Split) -> MetricCollection | None:
        match split:
            case "train":
                return None
            case "val":
                return self.val_metrics
            case "test":
                return self.test_metrics

    def reset_metrics(self, split: Split) -> None:
        self._split = split
        metrics = self._metrics(split)
        if metrics is not None:
            metrics.reset()

    def compute_metrics(self, split: Split) -> dict[str, float]:
        metrics = self._metrics(split)
        if metrics is None:
            return {}
        return {name: float(value) for name, value in metrics.compute().items()}

    def _make_example(self, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.schedule.sample_steps(batch_size=x0.size(0), device=x0.device)
        noise = torch.randn_like(x0)

        x_t = self.schedule.add_noise(x0=x0, t=t, noise=noise)
        return x_t, t, noise

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    @staticmethod
    def _randn(
        shape: tuple[int, ...],
        generator: torch.Generator | None,
        device: torch.device,
    ) -> torch.Tensor:
        noise_device = device if generator is None else generator.device
        return torch.randn(shape, generator=generator, device=noise_device).to(device)

    @staticmethod
    def _to_unit(x: torch.Tensor) -> torch.Tensor:
        return ((x + 1) / 2).clamp(0, 1)

    @classmethod
    def _to_image(cls, x: torch.Tensor) -> torch.Tensor:
        return cls._to_unit(x).cpu()

    def _sigma(self, t: int) -> torch.Tensor:
        variance = self.schedule.betas[t] if self.variance == "beta" else self.schedule.posterior_variance[t]
        return torch.sqrt(variance)

    def _denoise_step(self, x_t: torch.Tensor, t: int, generator: torch.Generator | None) -> torch.Tensor:
        beta_t = self.schedule.betas[t]
        alpha_t = self.schedule.alphas[t]
        alpha_bar_t = self.schedule.alpha_bars[t]
        alpha_bar_prev = self.schedule.alpha_bars_prev[t]

        t_batch = torch.full((x_t.size(0),), t, dtype=torch.long, device=x_t.device)
        eps: torch.Tensor = self.model(x_t, t_batch)
        x0 = ((x_t - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)).clamp(-1, 1)
        mean = (
            torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t) * x0
            + torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * x_t
        )

        if t == 0:
            return mean
        return mean + self._sigma(t) * self._randn(tuple(x_t.shape), generator, x_t.device)

    def _denoise(
        self,
        x: torch.Tensor,
        start: int,
        generator: torch.Generator | None,
        remaining: int = 0,
    ) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()

        for t in range(start, remaining - 1, -1):
            x = self._denoise_step(x, t, generator)

        self.model.train(was_training)
        return x

    def training_step(self, batch: Unlabeled) -> StepOutput:
        x_t, t, noise = self._make_example(batch)

        pred: torch.Tensor = self.model(x_t, t)

        loss: torch.Tensor = self.loss_fn(pred, noise)
        self.step(self.opt, loss)
        return {"loss": loss}

    def validation_step(self, batch: Unlabeled) -> StepOutput:
        x_t, t, noise = self._make_example(batch)

        pred: torch.Tensor = self.model(x_t, t)

        loss: torch.Tensor = self.loss_fn(pred, noise)

        metrics = self._metrics(self._split)
        if metrics is not None:
            x0_pred = self.schedule.predict_x0(x_t, t, pred)
            metrics.update(self._to_unit(x0_pred), self._to_unit(batch))
        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        num: int | None = None,
        z: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        remaining: int = 0,
    ) -> torch.Tensor:
        if (num is None) == (z is None):
            raise ValueError("generate expects exactly one of num or z")
        if not 0 <= remaining < self.schedule.steps:
            raise ValueError(f"remaining must be in [0, {self.schedule.steps}), got {remaining}")

        device = self._device()
        x = self._randn((num or 0, *self.image_shape), generator, device) if z is None else z.to(device)
        return self._to_image(self._denoise(x, self.schedule.steps - 1, generator, remaining))

    @torch.no_grad()
    def restore(
        self,
        x0: torch.Tensor,
        steps: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 1 <= steps <= self.schedule.steps:
            raise ValueError(f"steps must be in [1, {self.schedule.steps}], got {steps}")

        device = self._device()
        x0 = x0.to(device)
        t = torch.full((x0.size(0),), steps - 1, dtype=torch.long, device=device)
        x_t = self.schedule.add_noise(x0, t, self._randn(tuple(x0.shape), generator, device))
        restored = self._denoise(x_t, steps - 1, generator)
        return self._to_image(x_t), self._to_image(restored)
