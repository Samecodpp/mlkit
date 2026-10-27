from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import mlflow
import torch
from torch import nn

from .history import Mode

if TYPE_CHECKING:
    from .trainer import Trainer

logger = logging.getLogger(__name__)

type MonitorStatus = Literal["improved", "not_improved", "invalid"]


class Callback:

    def on_fit_start(self, trainer: Trainer) -> None: ...

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: Mapping[str, float]) -> None: ...

    def on_fit_end(self, trainer: Trainer) -> None: ...


class ProgressLogger(Callback):
    def __init__(
        self,
        every: int = 1,
        precision: int = 4,
        metrics: Sequence[str] | None = None,
    ) -> None:
        if every < 1:
            raise ValueError(f"every must be at least 1, got {every}")

        self.every = every
        self.precision = precision
        self.metrics = metrics
        self._fit_start = 0.0
        self._last_report = 0.0

    def on_fit_start(self, trainer: Trainer) -> None:
        self._fit_start = self._last_report = time.perf_counter()

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: Mapping[str, float]) -> None:
        if epoch % self.every != 0:
            return

        now = time.perf_counter()
        values = " ".join(
            f"{name} {value:.{self.precision}g}"
            for name, value in logs.items()
            if self.metrics is None or name.partition("/")[2] in self.metrics
        )
        logger.info("Epoch %d | %s | %.1fs", epoch, values, now - self._last_report)
        self._last_report = now

    def on_fit_end(self, trainer: Trainer) -> None:
        logger.info("Fit ended at epoch %d after %.1fs", trainer.epoch, time.perf_counter() - self._fit_start)


class _MetricMonitor(Callback):
    def __init__(
        self,
        monitor: str,
        mode: Mode,
        min_delta: float = 0.0,
    ) -> None:
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self._reset_best()

    def _reset_best(self) -> None:
        self.best = math.inf if self.mode == "min" else -math.inf

    def _check(self, logs: Mapping[str, float]) -> MonitorStatus:
        if self.monitor not in logs:
            raise KeyError(f"{type(self).__name__} monitor '{self.monitor}' not in logs. Available: {sorted(logs)}")

        value = logs[self.monitor]
        if not math.isfinite(value):
            return "invalid"

        diff = self.best - value
        improvement = diff if self.mode == "min" else -diff

        if improvement > 0 and improvement >= self.min_delta:
            return "improved"

        return "not_improved"


class EarlyStopping(_MetricMonitor):
    def __init__(
        self,
        monitor: str,
        mode: Mode,
        patience: int = 5,
        min_delta: float = 0.0,
    ) -> None:
        if patience <= 0:
            raise ValueError(f"patience must be at least 1, got {patience}")
        super().__init__(
            monitor=monitor,
            mode=mode,
            min_delta=min_delta,
        )
        self.patience = patience
        self._reset()

    def _reset(self) -> None:
        self._reset_best()
        self.wait = 0

    def on_fit_start(self, trainer: Trainer) -> None:
        self._reset()

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: Mapping[str, float]) -> None:
        result = self._check(logs)

        if result == "invalid":
            logger.warning("Early stopping at epoch %d: %s is %s", epoch, self.monitor, logs[self.monitor])
            trainer.should_stop = True
        elif result == "improved":
            self.best = logs[self.monitor]
            self.wait = 0
        else:
            self.wait += 1

            if self.wait >= self.patience:
                logger.info(
                    "Early stopping at epoch %d: %s did not improve for %d epochs (best %.4g)",
                    epoch, self.monitor, self.wait, self.best,
                )
                trainer.should_stop = True


class ModelCheckpoint(_MetricMonitor):
    def __init__(
        self,
        task: nn.Module,
        monitor: str,
        mode: Mode,
        min_delta: float = 0.0,
        restore: bool = True,
    ) -> None:
        super().__init__(
            monitor=monitor,
            mode=mode,
            min_delta=min_delta,
        )
        self.task = task
        self.restore = restore
        self.best_epoch = 0
        self.best_state: dict[str, torch.Tensor] = {}

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: Mapping[str, float]) -> None:
        if self._check(logs) == "improved":
            self.best = logs[self.monitor]
            self.best_epoch = epoch
            self.best_state = {name: tensor.detach().clone() for name, tensor in self.task.state_dict().items()}

    def on_fit_end(self, trainer: Trainer) -> None:
        if self.restore and self.best_state:
            self.task.load_state_dict(self.best_state)
            logger.info("Restored weights from epoch %d (%s %.4g)", self.best_epoch, self.monitor, self.best)


class MLflowLogger(Callback):
    def on_fit_start(self, trainer: Trainer) -> None:
        if mlflow.active_run() is None:
            raise RuntimeError("MLflowLogger needs an active run: wrap training in `with mlflow.start_run():`")

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: Mapping[str, float]) -> None:
        mlflow.log_metrics(dict(logs), step=epoch)
