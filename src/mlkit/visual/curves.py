from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from ..core.history import History

LINESTYLES = ("-", "--", ":", "-.")


def plot_history(
    history: History,
    metrics: Sequence[str],
    splits: Sequence[str] = ("train", "val"),
) -> Figure:
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 3.8), squeeze=False)
    available = history.keys()

    for ax, metric in zip(axes[0], metrics, strict=True):
        keys = [f"{split}/{metric}" for split in splits if f"{split}/{metric}" in available]
        if not keys:
            raise KeyError(f"No series for metric '{metric}'. Available: {sorted(available)}")

        for key in keys:
            ax.plot(*history.series(key), label=key.partition("/")[0])
        _format_axis(ax, metric)

    fig.tight_layout()
    return fig


def plot_comparison(
    histories: Mapping[str, History],
    metrics: Sequence[str],
    splits: Sequence[str] = ("val",),
) -> Figure:
    if not histories:
        raise ValueError("histories must not be empty")
    if len(splits) > len(LINESTYLES):
        raise ValueError(f"at most {len(LINESTYLES)} splits can be distinguished, got {len(splits)}")

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 3.8), squeeze=False)

    for ax, metric in zip(axes[0], metrics, strict=True):
        for color_index, (name, history) in enumerate(histories.items()):
            available = history.keys()
            keys = [(split, f"{split}/{metric}") for split in splits if f"{split}/{metric}" in available]
            if not keys:
                raise KeyError(f"No series for metric '{metric}' in run '{name}'. Available: {sorted(available)}")

            for split, key in keys:
                ax.plot(
                    *history.series(key),
                    color=f"C{color_index}",
                    linestyle=LINESTYLES[splits.index(split)],
                    label=name if len(splits) == 1 else f"{name} ({split})",
                )
        _format_axis(ax, metric)

    fig.tight_layout()
    return fig


def _format_axis(ax: Axes, metric: str) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.legend(fontsize="small" if len(ax.get_lines()) > 4 else None)
