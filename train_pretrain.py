import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data_utils import BASE_DIR, build_dataset, impute, normalize
from pretrain_model import GRUPretrainModel


class ICUData(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def mask_input(x, mask_ratio=0.20):
    mask = torch.rand_like(x) < mask_ratio
    x_masked = x.clone()
    x_masked[mask] = 0
    return x_masked, mask


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_dir = BASE_DIR / "data" / "set-a"
    outcome_file = BASE_DIR / "data" / "Outcomes-a.txt"

    X, _ = build_dataset(data_dir, outcome_file)
    X = impute(X)
    X = normalize(X)

    dataset = ICUData(X)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = GRUPretrainModel(input_dim=X.shape[-1]).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for xb in loader:
            xb = xb.to(device)

            xb_masked, mask = mask_input(xb)

            optimizer.zero_grad()
            preds = model(xb_masked)

            loss = ((preds - xb) ** 2)[mask].mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:02d} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "pretrained_gru.pt")
    print("Saved pretrained model.")


if __name__ == "__main__":
    main()