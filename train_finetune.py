import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from data_utils import build_dataset, impute, normalize, BASE_DIR
from finetune_model import GRUFinetuneModel


class ICUData(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            probs = torch.sigmoid(logits)

            running_loss += loss.item() * xb.size(0)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs > 0.5).cpu().numpy().astype(int))
            all_targets.extend(yb.cpu().numpy().astype(int))

    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)

    return avg_loss, acc, auc


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).cpu().numpy().astype(int)

        running_loss += loss.item() * xb.size(0)
        all_preds.extend(preds)
        all_targets.extend(yb.cpu().numpy().astype(int))

    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)

    return avg_loss, acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_dir = BASE_DIR / "data" / "set-a"
    outcome_file = BASE_DIR / "data" / "Outcomes-a.txt"

    X, y = build_dataset(data_dir, outcome_file)
    X = impute(X)
    X = normalize(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_ds = ICUData(X_train, y_train)
    val_ds = ICUData(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = GRUFinetuneModel(input_dim=X.shape[-1], hidden_dim=64).to(device)

    # load pretrained GRU weights only
    pretrained_dict = torch.load("pretrained_gru.pt", map_location=device)
    model_dict = model.state_dict()

    filtered_dict = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict and "gru" in k
    }
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    print("Loaded pretrained GRU weights.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50
    best_val_auc = 0.0
    best_epoch = 0
    patience = 5
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_auc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), "best_finetuned_gru.pt")
            print("Saved new best fine-tuned model.")
        else:
            patience_counter += 1
            print(f"No improvement in Val AUC ({patience_counter}/{patience}).")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"\nBest Validation AUC: {best_val_auc:.4f}")
    print(f"Best Epoch: {best_epoch}")


if __name__ == "__main__":
    main()