import torch
from torch.utils.data import DataLoader, TensorDataset
from model import EEGNet
import numpy as np

# Load preprocessed data
X = np.load("data/motor_imagery/preprocessed/X.npy")  # Shape: (N, 1, 16, 64)
y = np.load("data/motor_imagery/preprocessed/y.npy")  # Shape: (N,)

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Train/test split
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model and optimizer
model = EEGNet(num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(20):
    model.train()
    running_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "eeg_classifier.pth")
