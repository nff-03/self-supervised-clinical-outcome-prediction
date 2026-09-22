import torch
import torch.nn as nn


class GRUMortalityModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.0):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, 48, input_dim)
        _, h_n = self.gru(x)          # h_n: (num_layers, batch, hidden_dim)
        last_hidden = h_n[-1]         # (batch, hidden_dim)
        logits = self.classifier(last_hidden).squeeze(-1)  # (batch,)
        return logits