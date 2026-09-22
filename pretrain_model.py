import torch
import torch.nn as nn


class GRUPretrainModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (batch, 48, features)
        out, _ = self.gru(x)  # (batch, 48, hidden)
        out = self.output_layer(out)  # (batch, 48, features)
        return out