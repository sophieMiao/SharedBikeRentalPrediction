import torch
from torch import nn

from trans.model import PositionalEncoding
from lstm.model import LSTMModel


class MixedModel(nn.Module):
    """Transformer model for time series prediction"""

    def __init__(
        self, features_dim, d_model=128, n_head=8, num_layers=3, output_size=96
    ):
        super(MixedModel, self).__init__()

        self.embedding = nn.Linear(features_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.lstm = LSTMModel(
            input_size=d_model,
            hidden_size1=128,
            hidden_size2=64,
            output_size=output_size,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for name, p in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p.data)
                elif "bias" in name:
                    nn.init.constant_(p, 0.0)
            elif "fc" in name:
                if "weight" in name:
                    nn.init.xavier_uniform_(p)
                elif "bias" in name:
                    nn.init.constant_(p, 0.0)
            elif p.dim() > 1:
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
        output = self.transformer_encoder(
            src, src_mask
        )  # [batch_size, seq_len, d_model]

        # # Decode only the last sequence element
        # output = output[:, -1, :]  # [batch_size, d_model]

        # Project to output size
        output = self.lstm(output)  # [batch_size, output_size]

        return output
