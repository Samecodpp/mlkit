from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING

from .history import Mode

if TYPE_CHECKING:
    from .trainer import Trainer

logger = logging.getLogger(__name__)


class Callback:

    def on_fit_start(self, trainer: Trainer) -> None: ...

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: Mapping[str, float]) -> None: ...

    def on_fit_end(self, trainer: Trainer) -> None: ...


class ProgressLogger(Callback):
    def __init__(self, every: int = 1, precision: int = 4) -> None:
        if every < 1:
            raise ValueError(f"every must be at least 1, got {every}")

        self.every = every
        self.precision = precision
        self._fit_start = 0.0
        self._last_report = 0.0

    def on_fit_start(self, trainer: Trainer) -> None:
        self._fit_start = self._last_report = time.perf_counter()

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: Mapping[str, float]) -> None:
        if epoch % self.every != 0:
            return

        now = time.perf_counter()
        values = " ".join(f"{name} {value:.{self.precision}g}" for name, value in logs.items())
        logger.info("Epoch %d | %s | %.1fs", epoch, values, now - self._last_report)
        self._last_report = now

    def on_fit_end(self, trainer: Trainer) -> None:
        logger.info("Fit ended at epoch %d after %.1fs", trainer.epoch, time.perf_counter() - self._fit_start)


class EarlyStopping(Callback):
    def __init__(
        self,
        monitor: str,
        mode: Mode,
        patience: int = 5,
        min_delta: float = 0.0,
    ) -> None:
        if patience <= 0:
            raise ValueError(f"patience must be at least 1, got {patience}")

        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self._reset()

    def on_fit_start(self, trainer: Trainer) -> None:
        self._reset()

    def _reset(self) -> None:
        self.best = math.inf if self.mode == "min" else -math.inf
        self.wait = 0

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: Mapping[str, float]) -> None:
        if self.monitor not in logs:
            raise KeyError(f"EarlyStopping monitor '{self.monitor}' not in logs. Available: {sorted(logs)}")

        value = logs[self.monitor]
        if not math.isfinite(value):
            logger.warning("Early stopping at epoch %d: %s is %s", epoch, self.monitor, value)
            trainer.should_stop = True
            return

        diff = self.best - value
        improvement = diff if self.mode == "min" else -diff

        if improvement > 0 and improvement >= self.min_delta:
            self.best = value
            self.wait = 0
            return

        self.wait += 1

        if self.wait >= self.patience:
            logger.info(
                "Early stopping at epoch %d: %s did not improve for %d epochs (best %.4g)",
                epoch, self.monitor, self.wait, self.best,
            )
            trainer.should_stop = True

