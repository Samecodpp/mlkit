import matplotlib.pyplot as plt


def use_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 100,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })
