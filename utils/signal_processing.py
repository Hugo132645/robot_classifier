import numpy as np

def bandpass_filter(data, lowcut=8, highcut=30, fs=256):
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [lowcut/fs*2, highcut/fs*2], btype='band')
    return filtfilt(b, a, data, axis=-1)

def get_live_eeg_window():
    # Dummy EEG window for testing
    return np.random.randn(1, 1, 16, 64).astype(np.float32)
