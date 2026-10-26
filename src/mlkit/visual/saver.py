from dataclasses import dataclass
from pathlib import Path

from matplotlib.figure import Figure
from torch.utils.tensorboard import SummaryWriter


@dataclass
class FigureSaver:
    directory: Path
    dpi: int = 150
    writer: SummaryWriter | None = None

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, fig: Figure, name: str, step: int = 0) -> Path:
        path = self.directory / f"{name}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        if self.writer is not None:
            self.writer.add_figure(name, fig, global_step=step, close=False)
        return path
