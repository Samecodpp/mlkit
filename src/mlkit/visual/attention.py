import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure


def plot_attention(
    weights: torch.Tensor,
    tokens: Sequence[str] | None = None,
    titles: Sequence[str] | None = None,
    ncols: int = 4,
    annotate: bool = False,
) -> Figure:
    if weights.dim() == 2:
        weights = weights.unsqueeze(0)
    if weights.dim() != 3 or weights.size(-1) != weights.size(-2):
        raise ValueError(f"weights must have shape (L, L) or (H, L, L), got {tuple(weights.shape)}")

    count, length = weights.size(0), weights.size(-1)
    if tokens is not None and len(tokens) != length:
        raise ValueError(f"got {len(tokens)} tokens for attention over {length} positions")
    if titles is not None and len(titles) != count:
        raise ValueError(f"got {len(titles)} titles for {count} attention maps")

    values = weights.detach().float().cpu()
    vmax = max(values.max().item(), 1e-6)
    ncols = min(ncols, count)
    nrows = math.ceil(count / ncols)
    size = max(3.0, 0.22 * length)

    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols + 1, size * nrows), squeeze=False, layout="constrained")
    images = []

    for index, ax in enumerate(axes.flat):
        if index >= count:
            ax.axis("off")
            continue

        images.append(ax.imshow(values[index].numpy(), cmap="Blues", vmin=0, vmax=vmax))
        ax.grid(False)
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")

        if tokens is not None:
            ticks = range(length)
            ax.set_xticks(ticks, labels=tokens, rotation=90, fontsize=8)
            ax.set_yticks(ticks, labels=tokens, fontsize=8)
        if titles is not None:
            ax.set_title(titles[index])

        if annotate:
            for i in range(length):
                for j in range(length):
                    value = values[index, i, j].item()
                    color = "white" if value > vmax / 2 else "black"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(images[0], ax=axes.ravel().tolist(), shrink=0.8, label="Attention weight")
    return fig
