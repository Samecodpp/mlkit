from dataclasses import dataclass

import math

import torch
import torch.nn.functional as F
from torch import nn


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    out = F.dropout(input=weights, p=dropout) @ v

    return out, weights


@dataclass(frozen=True)
class SelfAttentionConfig:
    num_heads: int
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, config: SelfAttentionConfig) -> None:
        if d_model % config.num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, got {d_model} and {config.num_heads}")

        super().__init__()

        self.head_dim = d_model // config.num_heads
        self.w_qkv = nn.Linear(
            in_features=d_model,
            out_features=d_model*3,
            bias=config.bias,
        )
        self.w_out = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=config.bias,
        )
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        qkv: torch.Tensor = self.w_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.config.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.config.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.config.num_heads, self.head_dim).transpose(1, 2)

        if mask is not None:
            mask = mask[:, None, None, :]

        dropout = self.config.dropout if self.training else 0.0
        out, weights = scaled_dot_product_attention(q, k, v, mask, dropout)
        out = out.transpose(1, 2).reshape(B, L, -1)
        result: torch.Tensor = self.w_out(out)
        return result, weights
