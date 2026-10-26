import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from torchmetrics.functional.classification import binary_auroc, binary_roc


def plot_roc(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Figure:
    fpr, tpr, _ = binary_roc(probs, targets.long())
    auroc = binary_auroc(probs, targets.long()).item()

    preds = probs > threshold
    positives = targets == 1
    point_tpr = (preds & positives).sum() / positives.sum().clamp(min=1)
    point_fpr = (preds & ~positives).sum() / (~positives).sum().clamp(min=1)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr.cpu().numpy(), tpr.cpu().numpy(), label=f"ROC (AUC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random")
    ax.scatter(
        [point_fpr.item()], [point_tpr.item()], color="tab:red", zorder=3,
        label=f"threshold {threshold} (TPR {point_tpr.item():.3f}, FPR {point_fpr.item():.3f})",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")

    fig.tight_layout()
    return fig
