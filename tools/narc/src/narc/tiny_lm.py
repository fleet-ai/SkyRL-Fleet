from __future__ import annotations

from typing import Any

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class TinyDecoderBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        hidden_dim = d_model * mlp_ratio

        self.attn_norm = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.mlp_norm = RMSNorm(d_model)
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, d_model = x.shape
        normed = self.attn_norm(x)
        qkv = self.qkv(normed).view(
            batch,
            sequence,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        mask = torch.ones(
            sequence,
            sequence,
            dtype=torch.bool,
            device=x.device,
        ).triu(1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        attended = torch.matmul(weights, value)
        attended = (
            attended.transpose(1, 2).contiguous().view(batch, sequence, d_model)
        )
        x = x + self.out_proj(attended)

        normed = self.mlp_norm(x)
        gate = self.gate_proj(normed)
        up = self.up_proj(normed)
        x = x + self.down_proj((gate * torch.sigmoid(gate)) * up)
        return x


class TinyDecoderLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        sequence_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(sequence_length, d_model)
        self.layers = nn.ModuleList(
            [
                TinyDecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.sequence_length = sequence_length
        self.reset_parameters(seed=0)

    def reset_parameters(self, *, seed: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if "norm.weight" in name:
                    parameter.fill_(1.0)
                elif parameter.dim() == 1:
                    parameter.zero_()
                else:
                    values = torch.randn(
                        parameter.shape,
                        generator=generator,
                        dtype=torch.float32,
                    )
                    parameter.copy_(values * 0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, sequence = input_ids.shape
        if sequence > self.sequence_length:
            raise ValueError(
                f"sequence length {sequence} exceeds maximum {self.sequence_length}"
            )
        positions = torch.arange(sequence, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.lm_head(x)


def build_model(config: Any, *, seed: int) -> TinyDecoderLM:
    model = TinyDecoderLM(
        vocab_size=config.vocab_size,
        sequence_length=config.sequence_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
    )
    model.reset_parameters(seed=seed)
    return model
