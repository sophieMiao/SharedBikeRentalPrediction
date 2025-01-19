import torch
from torch import nn


class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""

    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, output_size=96):
        super(LSTMModel, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        self.lstm1 = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size1, batch_first=True
        )

        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(0.1)

        self.lstm2 = nn.LSTM(
            input_size=hidden_size1, hidden_size=hidden_size2, batch_first=True
        )

        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(0.1)

        self.fc1 = nn.Linear(hidden_size2, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
            elif "fc" in name:
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        h0_1 = torch.zeros(1, batch_size, self.hidden_size1).to(x.device)
        c0_1 = torch.zeros(1, batch_size, self.hidden_size1).to(x.device)

        h0_2 = torch.zeros(1, batch_size, self.hidden_size2).to(x.device)
        c0_2 = torch.zeros(1, batch_size, self.hidden_size2).to(x.device)

        out, _ = self.lstm1(x, (h0_1, c0_1))

        out = out.contiguous().view(-1, self.hidden_size1)
        out = self.bn1(out)
        out = out.view(batch_size, seq_len, self.hidden_size1)
        out = self.dropout1(out)

        out, _ = self.lstm2(out, (h0_2, c0_2))

        out = out[:, -1, :]
        out = self.bn2(out)
        out = self.dropout2(out)

        out = self.fc1(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out
