import math

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int, hidden: int = 48):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, horizon))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int, d_model: int = 48, nhead: int = 4):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=96,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))

    def forward(self, x):
        z = self.pos(self.proj(x))
        z = self.encoder(z)
        return self.head(z.mean(dim=1))


class CNNTransformerForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int, d_model: int = 48, nhead: int = 4):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.local = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))

    def forward(self, x):
        z = self.proj(x)
        local = self.local(z.transpose(1, 2)).transpose(1, 2)
        z = self.pos(z + local)
        enc = self.encoder(z)
        pooled = enc.mean(dim=1)
        return self.head(pooled * self.gate(pooled))


def make_model(name: str, input_dim: int, horizon: int) -> nn.Module:
    if name == "lstm":
        return LSTMForecaster(input_dim, horizon)
    if name == "transformer":
        return TransformerForecaster(input_dim, horizon)
    if name == "cnn_transformer":
        return CNNTransformerForecaster(input_dim, horizon)
    raise ValueError(name)

