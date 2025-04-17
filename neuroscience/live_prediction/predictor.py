import torch
import numpy as np
from eeg_model.model import EEGNetAdvanced
from utils.signal_processing import get_live_eeg_window
from feedback.correction_logger import log_feedback

model = EEGNetAdvanced(num_classes=6)
model.load_state_dict(torch.load("neuroscience/eeg_model/eeg_classifier.pth"))
model.eval()

LABELS = ["rest", "open_hand", "close_hand", "pinch_grip", "thumbs_up", "wave"]

def predict_and_act():
    while True:
        eeg_data = get_live_eeg_window()  # (1, 1, 16, 64)
        with torch.no_grad():
            output = model(torch.tensor(eeg_data, dtype=torch.float32))
            pred_idx = torch.argmax(output, dim=1).item()
            prediction = LABELS[pred_idx]
            print("Prediction:", prediction)

            feedback = input("Is this correct? (y/n): ").strip().lower()
            if feedback == 'n':
                correct_label = input(f"Enter correct label {LABELS}: ")
                if correct_label in LABELS:
                    log_feedback(eeg_data, LABELS.index(correct_label))

if __name__ == "__main__":
    predict_and_act()
