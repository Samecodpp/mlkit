from dataclasses import dataclass
from pathlib import Path

import mlflow
from matplotlib.figure import Figure


@dataclass
class FigureSaver:
    directory: Path
    dpi: int = 150
    log_to_mlflow: bool = True

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, fig: Figure, name: str) -> Path:
        path = self.directory / f"{name}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        if self.log_to_mlflow and mlflow.active_run() is not None:
            mlflow.log_artifact(str(path))
        return path
