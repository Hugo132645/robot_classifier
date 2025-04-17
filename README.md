# InMoov Prosthetic Control – EEG-Driven Adaptive Learning System

This project integrates EEG-based movement classification, real-time prosthetic control, continuous learning through feedback, and vision-based grasp validation using ROS2.

---

## Table of Contents

1. [Overview](#-inmoov-prosthetic-control--eeg-driven-adaptive-learning-system)
2. [Project Structure](#-project-structure)
3. [Getting Started](#-getting-started)
    - [Install Dependencies](#1-install-dependencies)
    - [Train the Classifier](#2-train-the-classifier)
    - [Real-Time EEG Prediction](#3-predict-movements-in-real-time)
    - [Update Classifier with Feedback](#4-improve-the-classifier-from-feedback)
4. [EEG Model Details](#-model-highlights)
5. [Integrated Movements](#-integrated-movements)
6. [Vision System Modules](#-vision-modules-explained)
7. [ROS2 Communication](#-ros2-integration)
8. [Contributing](#-contributing)
9. [License](#-license)

---

## Project Structure

```
robot_classifier/
│
├── data/
│   └── motor_imagery/
│       └── raw/
│       └── preprocessed/
│
├── neuroscience/
│   ├── eeg_model/
│   │   ├── model.py
│   │   ├── train_classifier.py
│   │   ├── online_learning.py
│   │
│   ├── live_prediction/
│   │   └── predictor.py
│   │
│   ├── feedback/
│   │   └── correction_logger.py
│   │
│   ├── utils/
│   │   └── signal_processing.py
│   │
│   └── setup.sh
│
├── computer_vision/
│   ├── main.py
│   ├── grasp_validation.py
│   ├── grasp_identifier.py
│   ├── object_detection.py
│   ├── hand_landmarks.py
│   └── CONST.py
│
├── robotics/
│   ├── ROS2_node.py
│   └── arduino_servo_control.ino
```

---

## Getting Started

### 1. Install Dependencies

```bash
cd neuroscience
bash setup.sh
```

### 2. Train the Classifier

Place your EEG training data in `data/motor_imagery/preprocessed/` as `X.npy` and `y.npy`, then run:

```bash
python eeg_model/train_classifier.py
```

### 3. Predict Movements in Real-Time

```bash
python live_prediction/predictor.py
```

When a prediction is wrong, the system will ask for a correction, which is saved for later.

### 4. Improve the Classifier from Feedback

```bash
python -c "
import numpy as np
from feedback.correction_logger import CORRECTION_FILE
from eeg_model.online_learning import update_model_with_feedback
data = np.load(CORRECTION_FILE)
update_model_with_feedback(data['X'], data['y'])
"
```

---

## Model Highlights

- **EEGNetAdvanced**
  - Depthwise separable convolutions
  - Spatial attention mechanism
  - Dropout, batch normalization
- **Online Learning**
  - Learns from feedback over time
  - Adapts to new user-specific EEG signals

---

## Integrated Movements

- `rest`
- `open_hand`
- `close_hand`
- `pinch_grip`
- `thumbs_up`
- `wave`

These correspond to movements executed by ROS2 and the prosthetic arm.

---

## Vision Modules Explained

The `computer_vision/` folder includes:
- Real-time object and hand detection with MediaPipe and YOLO
- Grasp classification based on geometry + ML
- Stability and landmark-based grasp checking

---

## ROS2 Integration

The `robotics/ROS2_node.py` reads EEG predictions, translates them into motion commands and sends them to:
- Arduino via serial (servo control)
- Other ROS2 components

---

## Contributing

Feel free to fork and make improvements – especially if you’re improving classifier accuracy or adding new gesture support.

---

## License

MIT License. See `LICENSE` file for details.
