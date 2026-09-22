import torch
import torch.nn as nn


class GRUFinetuneModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        _, h_n = self.gru(x)
        last_hidden = h_n[-1]
        logits = self.classifier(last_hidden).squeeze(-1)
        return logits