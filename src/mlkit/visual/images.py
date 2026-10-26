import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure


def plot_image_grid(
    images: torch.Tensor,
    titles: Sequence[str] | None = None,
    ncols: int = 8,
) -> Figure:
    if images.dim() != 4:
        raise ValueError(f"images must have shape (N, C, H, W), got {tuple(images.shape)}")
    if titles is not None and len(titles) != len(images):
        raise ValueError(f"got {len(titles)} titles for {len(images)} images")

    count = images.size(0)
    ncols = min(ncols, count)
    nrows = math.ceil(count / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.8 * ncols, 2.0 * nrows), squeeze=False)

    for index, ax in enumerate(axes.flat):
        ax.axis("off")
        if index >= count:
            continue

        image = images[index].detach().cpu()
        if image.size(0) == 1:
            ax.imshow(image[0].numpy(), cmap="gray")
        else:
            ax.imshow(image.permute(1, 2, 0).clamp(0, 1).numpy())
        if titles is not None:
            ax.set_title(titles[index], fontsize=8)

    fig.tight_layout()
    return fig
