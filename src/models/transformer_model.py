import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, : x.size(1), :]
        return x


class EEGTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]  (or [batch_size, input_dim] if 1 window = 1 token)
        returns logits: [batch_size]
        """
        if x.dim() == 2:  # [batch_size, input_dim], add seq dim
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]

        x = self.input_fc(x)  # [batch_size, seq_len, d_model]
        x = self.encoder(x)  # [batch_size, seq_len, d_model]

        # Pool sequence dimension (take first token or mean)
        x = x.mean(dim=1)  # [batch_size, d_model]

        logits = self.fc(x)  # [batch_size, 1]
        return logits.squeeze(-1)  # [batch_size]
