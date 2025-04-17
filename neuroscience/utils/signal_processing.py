import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut=8, highcut=30, fs=256):
    b, a = butter(4, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
    return filtfilt(b, a, data, axis=-1)

def get_live_eeg_window():
    # Simulated EEG data (shape: 1, 1, 16 channels, 64 timepoints)
    return np.random.randn(1, 1, 16, 64).astype(np.float32)
