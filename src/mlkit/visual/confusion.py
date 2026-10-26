from collections.abc import Sequence

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    if preds.shape != targets.shape:
        raise ValueError(f"preds {tuple(preds.shape)} and targets {tuple(targets.shape)} must have the same shape")

    counts = torch.bincount(targets.long() * num_classes + preds.long(), minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def plot_confusion_matrix(
    matrix: torch.Tensor,
    class_names: Sequence[str],
    normalize: bool = False,
    title: str | None = None,
) -> Figure:
    num_classes = len(class_names)
    if matrix.shape != (num_classes, num_classes):
        raise ValueError(f"matrix {tuple(matrix.shape)} does not match {num_classes} class names")

    values = matrix.float() / matrix.sum(dim=1, keepdim=True).clamp(min=1) if normalize else matrix
    value_format = ".2f" if normalize else "d"
    size = max(4.0, 0.6 * num_classes + 2)

    fig, ax = plt.subplots(figsize=(size + 1, size))
    image = ax.imshow(values.cpu().numpy(), cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(image, ax=ax)
    ax.grid(False)

    ticks = range(num_classes)
    ax.set_xticks(ticks, labels=class_names, rotation=45 if num_classes > 10 else 0)
    ax.set_yticks(ticks, labels=class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title or ("Confusion matrix (row-normalized)" if normalize else "Confusion matrix"))

    threshold = values.max().item() / 2
    for i in ticks:
        for j in ticks:
            value = values[i, j].item()
            color = "white" if value > threshold else "black"
            ax.text(j, i, format(value, value_format), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    return fig
