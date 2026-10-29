from collections.abc import Sequence, Mapping
from typing import cast

import torch


type Batch = torch.Tensor | Sequence[Batch] | Mapping[str, Batch]
type Labeled = tuple[torch.Tensor, torch.Tensor]
type Unlabeled = torch.Tensor


def move_to[B: Batch](batch: B, device: torch.device) -> B:
    return cast(B, _move(batch, device))


def _move(batch: Batch, device: torch.device) -> Batch:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, Mapping):
        return {key: _move(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_move(item, device) for item in batch)
    if isinstance(batch, list):
        return [_move(item, device) for item in batch]
    raise TypeError(f"Cannot move {type(batch).__name__} to device")


def first_tensor(batch: Batch) -> torch.Tensor | None:
    if isinstance(batch, torch.Tensor):
        return batch

    items = batch.values() if isinstance(batch, Mapping) else batch
    for item in items:
        tensor = first_tensor(item)
        if tensor is not None:
            return tensor
    return None
