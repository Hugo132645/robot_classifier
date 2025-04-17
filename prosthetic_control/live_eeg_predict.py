import torch
from eeg.model import EEGNet
import numpy as np
from utils.signal_processing import get_live_eeg_window
from prosthetic_control.actuator_interface import send_command

model = EEGNet(num_classes=2)
model.load_state_dict(torch.load("eeg_classifier.pth"))
model.eval()

while True:
    eeg_data = get_live_eeg_window()  # Shape: (1, 1, 16, 64)
    with torch.no_grad():
        output = model(torch.tensor(eeg_data, dtype=torch.float32))
        pred = torch.argmax(output, dim=1).item()
        if pred == 0:
            send_command("GRIP")
        elif pred == 1:
            send_command("RELEASE")
