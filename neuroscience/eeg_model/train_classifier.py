import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from eeg_model.model import EEGNetAdvanced
import numpy as np
import os

DATA_DIR = "data/motor_imagery/preprocessed"
MODEL_PATH = "neuroscience/eeg_model/eeg_classifier.pth"

def train():
    # Load preprocessed data
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)

    model = EEGNetAdvanced(num_classes=6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(50):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

if __name__ == "__main__":
    train()
