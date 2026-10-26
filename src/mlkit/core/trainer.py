from collections.abc import Iterable, Mapping, Sequence

import torch

from ..utils.device import get_device
from .batch import Batch, move_to
from .callbacks import Callback
from .history import History
from .task import Split, Task


def _prefixed(split: Split, results: Mapping[str, float]) -> dict[str, float]:
    return {f"{split}/{name}": value for name, value in results.items()}


class Trainer:
    def __init__(
        self,
        device: torch.device | None = None,
        callbacks: Sequence[Callback] = (),
    ) -> None:
        self.device = device or get_device()
        self.callbacks = list(callbacks)
        self.history = History()
        self.should_stop = False
        self.epoch = 0

    def _run_epoch[BatchT: Batch](
        self,
        task: Task[BatchT],
        data: Iterable[BatchT],
        split: Split,
    ) -> dict[str, float]:
        is_train = split == "train"

        task.train(is_train)
        task.reset_metrics(split)

        step = task.training_step if is_train else task.validation_step
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}

        num_batches = 0

        with torch.set_grad_enabled(is_train or task.grad_in_eval):
            for batch in data:
                batch = move_to(batch, self.device)
                n = task.batch_size(batch)
                output = step(batch)

                for name, value in output.items():
                    if isinstance(value, torch.Tensor):
                        value = value.detach().item()
                    totals[name] = totals.get(name, 0.0) + value * n
                    counts[name] = counts.get(name, 0) + n

                num_batches += 1

        if num_batches == 0:
            raise ValueError(f"{split} data yielded no batches")

        results = {name: totals[name] / counts[name] for name in totals}
        metrics = task.compute_metrics(split)

        if duplicates := results.keys() & metrics.keys():
            raise ValueError(
                f"{type(task).__name__} reports {sorted(duplicates)} "
                "both from its step and from compute_metrics"
            )

        return results | metrics

    def fit[BatchT: Batch](
        self,
        task: Task[BatchT],
        train_data: Iterable[BatchT],
        val_data: Iterable[BatchT] | None = None,
        *,
        epochs: int,
    ) -> History:
        if epochs <= 0:
            raise ValueError(f"epochs must be at least 1, got {epochs}")

        task.to(self.device)
        self.should_stop = False

        for callback in self.callbacks:
            callback.on_fit_start(self)

        try:
            for _ in range(epochs):
                self.epoch += 1
                result = _prefixed("train", self._run_epoch(task=task, data=train_data, split="train"))

                if val_data is not None:
                    result |= _prefixed("val", self._run_epoch(task=task, data=val_data, split="val"))

                self.history.log(self.epoch, result)

                for callback in self.callbacks:
                    callback.on_epoch_end(self, self.epoch, result)

                if self.should_stop:
                    break
        finally:
            for callback in self.callbacks:
                callback.on_fit_end(self)

        return self.history

    def evaluate[BatchT: Batch](
        self,
        task: Task[BatchT],
        data: Iterable[BatchT],
        split: Split = "test",
    ) -> dict[str, float]:
        if split == "train":
            raise ValueError("evaluate cannot run on the 'train' split: it would update the weights")

        task.to(self.device)
        return self._run_epoch(task=task, data=data, split=split)
