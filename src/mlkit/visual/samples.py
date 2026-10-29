from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import torch

from ..core.callbacks import Callback
from .images import plot_image_grid
from .saver import FigureSaver

if TYPE_CHECKING:
    from ..core.trainer import Trainer


class SampleGrid(Callback):
    def __init__(
        self,
        sample: Callable[[], torch.Tensor],
        saver: FigureSaver,
        every: int = 1,
        ncols: int = 8,
    ) -> None:
        if every < 1:
            raise ValueError(f"every must be at least 1, got {every}")

        self.sample = sample
        self.saver = saver
        self.every = every
        self.ncols = ncols

    def on_fit_start(self, trainer: Trainer) -> None:
        if trainer.epoch == 0:
            self._save(0)

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: Mapping[str, float]) -> None:
        if epoch % self.every == 0:
            self._save(epoch)

    def _save(self, epoch: int) -> None:
        fig = plot_image_grid(self.sample(), ncols=self.ncols)
        fig.suptitle(f"Epoch {epoch}")
        self.saver.save(fig, f"samples_epoch_{epoch:03d}")
        plt.close(fig)
