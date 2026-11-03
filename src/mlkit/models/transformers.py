from dataclasses import dataclass

import torch
from torch import nn

from ..blocks.embeddings import TokenEmbedding, TokenEmbeddingConfig
from ..blocks.linear import LinearBlock, LinearBlockConfig
from ..blocks.norms import LayerNorm
from ..blocks.pooling import masked_mean
from ..blocks.transformer import TransformerEncoderBlock, TransformerEncoderBlockConfig


@dataclass(frozen=True)
class TransformerClassifierConfig:
    vocab_size: int
    embedding: TokenEmbeddingConfig
    blocks: tuple[TransformerEncoderBlockConfig, ...]
    head: tuple[LinearBlockConfig, ...]

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError(f"vocab_size must be at least 2, got {self.vocab_size}")
        if not self.head:
            raise ValueError(f"head must contain at least one block, got {len(self.head)}")


class TransformerClassifier(nn.Module):
    def __init__(self, config: TransformerClassifierConfig) -> None:
        super().__init__()
        d_model = config.embedding.d_model

        self.embedding = TokenEmbedding(vocab_size=config.vocab_size, config=config.embedding)
        self.blocks = nn.ModuleList([TransformerEncoderBlock(d_model, conf) for conf in config.blocks])
        self.norm = LayerNorm().create(num_features=d_model, dim=1)

        layers: list[nn.Module] = []
        in_features = d_model
        for conf in config.head:
            layers.append(LinearBlock(in_features=in_features, config=conf))
            in_features = conf.out_features

        self.head = nn.Sequential(*layers)
        self.config = config

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_attention(ids)
        return logits

    def forward_with_attention(self, ids: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        mask = ids != self.config.embedding.pad_id
        x: torch.Tensor = self.embedding(ids)

        attentions: list[torch.Tensor] = []
        for block in self.blocks:
            x, weights = block(x, mask)
            attentions.append(weights)

        x = self.norm(x)
        logits: torch.Tensor = self.head(masked_mean(x, mask))
        return logits, tuple(attentions)
