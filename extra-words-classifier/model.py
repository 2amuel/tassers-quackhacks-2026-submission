from __future__ import annotations

import math

import torch
import torch.nn as nn


LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
WORDS = (
    "HELLO",
    "MY",
    "NAME",
    "I",
    "YOU",
    "YOUR",
    "HOW",
    "GOOD",
    "BAD",
    "HAPPY",
    "SAD",
    "NOT",
)
OUTPUT_LABELS = tuple(dict.fromkeys((*LETTERS, *WORDS)))


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer("pe", self._build(max_len))

    def _build(self, max_len: int) -> torch.Tensor:
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.embed_dim)
        )
        pe = torch.zeros(max_len, self.embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            self.pe = self._build(x.size(1)).to(device=x.device, dtype=x.dtype)
        return x + self.pe[:, : x.size(1)]


class LandmarkWindowTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        window_frames: int = 48,
        num_classes: int = len(OUTPUT_LABELS),
        embed_dim: int = 192,
        num_heads: int = 6,
        num_layers: int = 3,
        feedforward_dim: int = 384,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.window_frames = window_frames

        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.position = PositionalEncoding(embed_dim, window_frames)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pool = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3:
            raise ValueError("Expected windows shape (batch, frames, features).")
        if windows.size(-1) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {windows.size(-1)}.")

        x = self.input_projection(windows)
        x = self.position(x)
        x = self.encoder(x)
        attention = torch.softmax(self.attention_pool(x).squeeze(-1), dim=1)
        pooled = torch.sum(x * attention.unsqueeze(-1), dim=1)
        return self.classifier(pooled)


def create_model(input_dim: int, window_frames: int) -> LandmarkWindowTransformerClassifier:
    return LandmarkWindowTransformerClassifier(input_dim=input_dim, window_frames=window_frames)
