import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(
        self, n_channels, cnn_channels=64, lstm_hidden=128, lstm_layers=2, dropout=0.3
    ):
        super().__init__()
        # Temporal conv across channels-as-input (treat each channel as input channel)
        self.conv1 = nn.Conv1d(n_channels, cnn_channels, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(cnn_channels)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x):
        # x: (B, C, T)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        # (B, F, T') -> (B, T', F)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)  # temporal average pooling
        x = self.dropout(x)
        logits = self.head(x).squeeze(-1)
        return logits  # raw logits
