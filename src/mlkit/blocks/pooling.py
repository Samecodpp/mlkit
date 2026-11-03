import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask[..., None].to(x.dtype)
    summed = (x * weights).sum(dim=1)
    count = weights.sum(dim=1).clamp(min=1.0)
    return summed / count
