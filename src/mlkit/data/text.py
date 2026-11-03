import re
from collections.abc import Callable, Sequence, Iterable
from collections import Counter
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ..core.batch import Labeled


type Tokenizer = Callable[[str], list[str]]


@dataclass(frozen=True)
class WordTokenizer:
    pattern: str = r"\w+|[^\w\s]"

    def __call__(self, text: str) -> list[str]:
        tokens: list[str] = re.findall(self.pattern, text.lower())
        return tokens


class Vocabulary:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, min_freq: int = 2, max_size: int = 10_000) -> None:
        if min_freq < 1:
            raise ValueError(f"min_freq must be at least 1, got {min_freq}")

        if max_size <= 1:
            raise ValueError(f"max_size must be at least 2 to fit {self.PAD} and {self.UNK}, got {max_size}")

        self.min_freq = min_freq
        self.max_size = max_size
        self.decode_list: list[str] = [self.PAD, self.UNK]
        self.encode_dict: dict[str, int] = {self.PAD: 0, self.UNK: 1}

    def __len__(self) -> int:
        return len(self.decode_list)

    def _reset(self) -> None:
        self.decode_list = [self.PAD, self.UNK]
        self.encode_dict = {self.PAD: 0, self.UNK: 1}

    def fill(self, corpus: Iterable[Sequence[str]]) -> None:
        self._reset()

        counter = Counter(token for seq in corpus for token in seq)
        counts = {token: count for token, count in counter.items() if count >= self.min_freq}

        self.decode_list.extend(sorted(counts, key=lambda t: (-counts[t], t))[:self.max_size-2])
        self.encode_dict.update({key: idx for idx, key in enumerate(self.decode_list[2:], start=2)})

    def encode(self, tokens: Sequence[str]) -> list[int]:
        return [self.encode_dict.get(token, self.unk_id) for token in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.decode_list[idx] for idx in ids]

    @property
    def pad_id(self) -> int:
        return self.encode_dict[self.PAD]

    @property
    def unk_id(self) -> int:
        return self.encode_dict[self.UNK]


@dataclass(frozen=True)
class TextEncoder:
    tokenizer: Tokenizer
    vocabulary: Vocabulary
    max_len: int = 64

    def __post_init__(self) -> None:
        if self.max_len < 1:
            raise ValueError(f"max_len must be at least 1, got {self.max_len}")

    def __call__(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text)[:self.max_len]
        encoded_tokens = self.vocabulary.encode(tokens)

        if len(encoded_tokens) <= 0:
            encoded_tokens = [self.vocabulary.unk_id]

        return torch.tensor(encoded_tokens, dtype=torch.long)


class TextDataset(Dataset[tuple[str, int]]):
    def __init__(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        if len(texts) != len(labels):
            raise ValueError(f"texts and labels must have the same length, got {len(texts)} and {len(labels)}")

        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.texts[index], self.labels[index]


@dataclass(frozen=True)
class PadCollate:
    pad_id: int

    def __call__(self, batch: Sequence[tuple[torch.Tensor, int]]) -> Labeled:
        sequences = [ids for ids, _ in batch]
        labels = [label for _, label in batch]

        padded = pad_sequence(sequences, batch_first=True, padding_value=self.pad_id)
        return padded, torch.tensor(labels, dtype=torch.long)



