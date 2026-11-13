from __future__ import annotations

import torch
import torch.nn as nn


class LightweightLogTransformer(nn.Module):
    """Compact Transformer Encoder classifier designed for fast CPU/ONNX inference."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 48,
        embed_dim: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        # Float mask avoids nested-tensor ONNX export paths while preserving attention behavior.
        key_padding_mask = attention_mask == 0
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)

        mask_expanded = attention_mask.unsqueeze(-1).to(encoded.dtype)
        summed = torch.sum(encoded * mask_expanded, dim=1)
        counts = torch.clamp(mask_expanded.sum(dim=1), min=1.0)
        pooled = self.norm(summed / counts)
        return self.classifier(pooled)
