import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------
# BUILD NETWORK
# ---------------------------
class BibleNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BibleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 80),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(40, num_classes)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # ---------------------------
    # LOAD DATA
    # ---------------------------
    print("Loading data...")

    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")

    num_classes = len(np.unique(y_train))

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {num_classes}")
    print(f"Class dist: {np.bincount(y_train)}")

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create PyTorch datasets & dataloaders
    batch_size = 128
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = BibleNet(X_train.shape[1], num_classes).to(device)


    # ---------------------------
    # TRAIN NETWORK
    # ---------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-2)  # weight_decay = L2

    epochs = 12
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            # accumulate total loss weighted by batch size
            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_val, y_val in test_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)
                loss = criterion(val_outputs, y_val)
                val_loss += loss.item() * X_val.size(0)
                _, val_predicted = torch.max(val_outputs, 1)
                val_correct += (val_predicted == y_val).sum().item()
                val_total += y_val.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


    # ---------------------------
    # SAVE TRAINED MODEL
    # ---------------------------
    torch.save(model.state_dict(), "pytorch_nn_model.pth")
    print("Neural network saved to pytorch_nn_model.pth")
