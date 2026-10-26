from dataclasses import dataclass
from collections import defaultdict
from typing import Protocol, Sequence
import math

import torch


class Splitter(Protocol):
    def split(self, targets: Sequence[int]) -> list[list[int]]: ...


@dataclass(frozen=True)
class _FractionSplitter:
    fractions: tuple[float, ...]
    seed: int

    def __post_init__(self) -> None:
        if not self.fractions or any(fraction <= 0 for fraction in self.fractions):
            raise ValueError(f"fractions must be non-empty and positive, got {self.fractions}")
        if not math.isclose(sum(self.fractions), 1.0):
            raise ValueError(f"fractions must sum to 1, got {sum(self.fractions)}")

    def _cut(self, items: Sequence[int]) -> list[list[int]]:
        parts: list[list[int]] = []
        start = 0
        for fraction in self.fractions[:-1]:
            end = start + int(fraction * len(items))
            parts.append(list(items[start:end]))
            start = end
        parts.append(list(items[start:]))
        return parts


@dataclass(frozen=True)
class RandomSplitter(_FractionSplitter, Splitter):

    def split(self, targets: Sequence[int]) -> list[list[int]]:
        generator = torch.Generator().manual_seed(self.seed)
        order: list[int] = torch.randperm(len(targets), generator=generator).tolist()
        return self._cut(order)


@dataclass(frozen=True)
class StratifiedSplitter(_FractionSplitter, Splitter):

    def split(self, targets: Sequence[int]) -> list[list[int]]:
        groups: defaultdict[int, list[int]] = defaultdict(list)
        for index, target in enumerate(targets):
            groups[target].append(index)

        generator = torch.Generator().manual_seed(self.seed)
        parts: list[list[int]] = [[] for _ in self.fractions]
        for target in sorted(groups):
            group = groups[target]
            positions: list[int] = torch.randperm(len(group), generator=generator).tolist()
            shuffled = [group[position] for position in positions]
            
            for part, chunk in zip(parts, self._cut(shuffled), strict=True):
                part.extend(chunk)

        return parts
