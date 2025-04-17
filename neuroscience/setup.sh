#!/bin/bash

echo "[INFO] Setting up EEG prosthetic environment..."

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install torch numpy scipy matplotlib scikit-learn ipykernel

echo "[INFO] Creating data folders..."
mkdir -p data/motor_imagery/preprocessed
mkdir -p neuroscience/feedback

echo "[INFO] Setup complete. You can now train the model using:"
echo "python neuroscience/eeg_model/train_classifier.py"
