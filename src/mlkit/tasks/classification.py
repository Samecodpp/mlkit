from collections.abc import Iterable

import torch
from torch import nn
from torchmetrics import MetricCollection

from ..core.batch import Labeled, move_to
from ..core.task import OptimizerFactory, Split, StepOutput, Task


class ClassificationTask(Task[Labeled]):
    def __init__(
        self,
        model: nn.Module,
        optimizer: OptimizerFactory,
        metrics: MetricCollection,
        binary: bool = False,
        loss_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.opt = optimizer(model.parameters())
        self.binary = binary
        self.loss_fn = (
            loss_fn if loss_fn is not None
            else nn.BCEWithLogitsLoss() if binary
            else nn.CrossEntropyLoss()
        )
        self.train_metrics = metrics.clone()
        self.val_metrics = metrics.clone()
        self.test_metrics = metrics.clone()
        self._split: Split = "train"

    def _metrics(self, split: Split) -> MetricCollection:
        match split:
            case "train":
                return self.train_metrics
            case "val":
                return self.val_metrics
            case "test":
                return self.test_metrics

    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor
        if self.binary:
            loss = self.loss_fn(logits.squeeze(1), y.to(logits.dtype))
        else:
            loss = self.loss_fn(logits, y)
        return loss

    def _probs(self, logits: torch.Tensor) -> torch.Tensor:
        if self.binary:
            return torch.sigmoid(logits.squeeze(1))
        return torch.softmax(logits, dim=1)

    def reset_metrics(self, split: Split) -> None:
        self._split = split
        self._metrics(split).reset()

    def compute_metrics(self, split: Split) -> dict[str, float]:
        return {name: float(value) for name, value in self._metrics(split).compute().items()}

    def training_step(self, batch: Labeled) -> StepOutput:
        x, y = batch
        logits: torch.Tensor = self.model(x)
        loss = self._loss(logits, y)
        self.step(self.opt, loss)
        self.train_metrics.update(self._probs(logits.detach()), y)
        return {"loss": loss}

    def validation_step(self, batch: Labeled) -> StepOutput:
        x, y = batch
        logits: torch.Tensor = self.model(x)
        loss = self._loss(logits, y)
        self._metrics(self._split).update(self._probs(logits), y)
        return {"loss": loss}

    @torch.no_grad()
    def predict(self, data: Iterable[Labeled]) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        all_probs: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for batch in data:
            x, y = move_to(batch, device)
            logits: torch.Tensor = self.model(x)
            all_probs.append(self._probs(logits).cpu())
            all_targets.append(y.cpu())

        self.train(was_training)

        if not all_probs:
            raise ValueError("predict data yielded no batches")

        return torch.cat(all_probs), torch.cat(all_targets)
