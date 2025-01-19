import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, x.size(1)]


class TransformerModel(nn.Module):
    """Transformer model for time series prediction"""

    def __init__(self, features_dim, d_model=128, n_head=8, num_layers=3, output_size=96):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(features_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, output_size)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        # src shape: [batch_size, seq_len, input_size]

        # Create mask for transformer
        src_mask = None  # You can implement masking if needed

        # Embed input
        src = self.embedding(src)  # [batch_size, seq_len, d_model]

        # Add positional encoding
        src = self.pos_encoder(src)

        # Transform
        output = self.transformer_encoder(src, src_mask)  # [batch_size, seq_len, d_model]

        # Decode only the last sequence element
        output = output[:, -1, :]  # [batch_size, d_model]

        # Project to output size
        output = self.decoder(output)  # [batch_size, output_size]

        return output

