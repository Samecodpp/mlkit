from collections.abc import Mapping
from typing import Literal

type Mode = Literal["min", "max"]
type Records = dict[str, list[tuple[int, float]]]


class History:
    def __init__(self) -> None:
        self.records: Records = {}

    def log(self, epoch: int, values: Mapping[str, float]) -> None:
        for key, value in values.items():
            self.records.setdefault(key, []).append((epoch, value))

    def series(self, key: str) -> tuple[list[int], list[float]]:
        record = self._record(key)
        epochs = [epoch for epoch, _ in record]
        values = [value for _, value in record]
        return epochs, values

    def last(self, key: str) -> float:
        return self._record(key)[-1][1]

    def best(self, key: str, mode: Mode) -> tuple[int, float]:
        select = min if mode == "min" else max
        return select(self._record(key), key=lambda item: item[1])

    def keys(self) -> list[str]:
        return list(self.records.keys())

    def to_dict(self) -> dict[str, dict[int, float]]:
        return {key: dict(record) for key, record in self.records.items()}

    def _record(self, key: str) -> list[tuple[int, float]]:
        if key not in self.records:
            raise KeyError(f"'{key}' not found. Available: {sorted(self.records)}")
        return self.records[key]
