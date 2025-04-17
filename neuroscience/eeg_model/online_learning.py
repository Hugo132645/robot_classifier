import torch
from torch.utils.data import DataLoader, TensorDataset
from eeg_model.model import EEGNetAdvanced
import numpy as np
import os

MODEL_PATH = "neuroscience/eeg_model/eeg_classifier.pth"

def update_model_with_feedback(X_new, y_new):
    model = EEGNetAdvanced(num_classes=6)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.CrossEntropyLoss()

    dataset = TensorDataset(torch.tensor(X_new, dtype=torch.float32), torch.tensor(y_new, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    for epoch in range(3):  # Light update
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), MODEL_PATH)
