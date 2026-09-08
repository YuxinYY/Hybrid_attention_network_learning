"""Text baselines over the same frozen FinBERT embeddings used by HAN."""
import torch
from torch import nn


def daily_mean(x_text, x_mask):
    """Average real posts only; absent days are exactly zero."""
    clean = x_text.masked_fill(~x_mask.unsqueeze(-1), 0)
    counts = x_mask.sum(dim=2, keepdim=True).clamp_min(1)
    return clean.sum(dim=2) / counts, x_mask.any(dim=2)


def window_mean(x_text, x_mask):
    """Equal weight per observed day, excluding empty days."""
    days, valid = daily_mean(x_text, x_mask)
    return days.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)


class TemporalCNN(nn.Module):
    """Daily mean -> Conv1d kernels 3/5 (64 channels each) -> max pool -> logits."""
    def __init__(self, embedding_dim=768, dropout=0.4):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, 64, k, padding=k // 2) for k in (3, 5)
        ])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, 32),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 2))

    def forward(self, x_text, x_mask):
        days, _ = daily_mean(x_text, x_mask)
        # Keep zero-valued missing days at their original temporal positions.
        features = [conv(days.transpose(1, 2)).relu().amax(dim=2) for conv in self.convs]
        return self.head(torch.cat(features, dim=1)), None, None


class TemporalTransformer(nn.Module):
    """Daily means + positions + CLS -> two-layer Transformer Encoder.

    PyTorch reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
    This temporal encoder is trained from scratch; only FinBERT features are pretrained.
    """
    def __init__(self, embedding_dim=768, dropout=0.4, max_days=20):
        super().__init__()
        self.max_days = max_days
        self.project = nn.Linear(embedding_dim, 128)
        self.positions = nn.Parameter(torch.randn(1, max_days + 1, 128) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, 128) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=dropout,
            activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        # Encoder clones start with identical weights unless independently initialized.
        for block in self.encoder.layers:
            for param in block.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, 32),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 2))

    def forward(self, x_text, x_mask):
        days, valid = daily_mean(x_text, x_mask)
        batch, window, _ = days.shape
        if window > self.max_days:
            raise ValueError(f'Window {window} exceeds max_days={self.max_days}')
        tokens = torch.cat([self.cls.expand(batch, -1, -1), self.project(days)], dim=1)
        tokens = tokens + self.positions[:, :window + 1]
        # CLS always remains unmasked, including windows with no text at all.
        padding = torch.cat([torch.zeros(batch, 1, dtype=torch.bool, device=days.device),
                             ~valid], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=padding)
        return self.head(encoded[:, 0]), None, None
