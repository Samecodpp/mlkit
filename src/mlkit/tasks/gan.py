import torch
from torch import nn

from ..core.batch import Unlabeled
from ..core.task import OptimizerFactory, StepOutput, Task
from ..models.generator import Generator


class GANTask(Task[Unlabeled]):
    def __init__(
        self,
        generator: Generator,
        discriminator: nn.Module,
        g_optimizer: OptimizerFactory,
        d_optimizer: OptimizerFactory,
        loss_fn: nn.Module | None = None,
        real_label: float = 0.9,
    ) -> None:
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.g_opt = g_optimizer(generator.parameters())
        self.d_opt = d_optimizer(discriminator.parameters())
        self.loss_fn = loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss()
        self.real_label = real_label

    def _d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        return (
            self.loss_fn(real_logits, torch.full_like(real_logits, self.real_label))
            + self.loss_fn(fake_logits, torch.zeros_like(fake_logits))
        )

    def _g_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, torch.ones_like(logits))

    def training_step(self, batch: Unlabeled) -> StepOutput:
        z = torch.randn(batch.size(0), self.generator.config.latent_dim, device=batch.device)
        fake_batch: torch.Tensor = self.generator(z)

        real_logits: torch.Tensor = self.discriminator(batch)
        fake_logits: torch.Tensor = self.discriminator(fake_batch.detach())

        d_loss = self._d_loss(real_logits=real_logits, fake_logits=fake_logits)
        self.step(optimizer=self.d_opt, loss=d_loss)

        g_logits: torch.Tensor = self.discriminator(fake_batch)

        g_loss = self._g_loss(g_logits)
        self.step(optimizer=self.g_opt, loss=g_loss)

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "d_real": torch.sigmoid(real_logits.detach()).mean(),
            "d_fake": torch.sigmoid(fake_logits.detach()).mean(),
        }

    @torch.no_grad()
    def generate(self, num: int | None = None, z: torch.Tensor | None = None) -> torch.Tensor:
        if (num is None) == (z is None):
            raise ValueError("generate expects exactly one of num or z")

        device = next(self.generator.parameters()).device
        if z is None:
            z = torch.randn(num or 0, self.generator.config.latent_dim, device=device)

        was_training = self.generator.training
        self.generator.eval()
        images: torch.Tensor = self.generator(z.to(device))
        self.generator.train(was_training)

        return ((images + 1) / 2).clamp(0, 1).cpu()
