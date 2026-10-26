from collections.abc import Callable, Sequence

from torch.utils.data import Dataset


class TransformSubset[X, Z, Y](Dataset[tuple[Z, Y]]):
    def __init__(
        self,
        base: Dataset[tuple[X, Y]],
        indices: Sequence[int],
        transform: Callable[[X], Z],
    ) -> None:
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Z, Y]:
        x, y = self.base[self.indices[index]]
        return self.transform(x), y
