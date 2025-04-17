import numpy as np
import os

CORRECTION_FILE = "neuroscience/feedback/corrections.npz"

def log_feedback(X, y):
    if os.path.exists(CORRECTION_FILE):
        data = np.load(CORRECTION_FILE)
        X_old = data["X"]
        y_old = data["y"]
        X_new = np.concatenate([X_old, X], axis=0)
        y_new = np.concatenate([y_old, [y]], axis=0)
    else:
        X_new = X
        y_new = np.array([y])

    np.savez_compressed(CORRECTION_FILE, X=X_new, y=y_new)
    print("Logged correction.")
