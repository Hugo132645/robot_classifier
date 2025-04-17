# InMoov Prosthetic Control – EEG-Driven Adaptive Learning System

This project integrates EEG-based movement classification, real-time prosthetic control, continuous learning through feedback, and vision-based grasp validation using ROS2.

## Project Structure

``bash
inmoov_prosthetic_eeg/
│
├── data/
│   └── motor_imagery/
│       └── raw/                     # Raw EEG data (optional)
│       └── preprocessed/            # Numpy files: X.npy and y.npy
│
├── neuroscience/
│   ├── eeg_model/
│   │   ├── model.py                 # EEGNetAdvanced with attention + depthwise conv
│   │   ├── train_classifier.py      # Full training pipeline
│   │   ├── online_learning.py       # Incremental model update using feedback
│   │
│   ├── live_prediction/
│   │   └── predictor.py             # Real-time EEG classifier with correction interface
│   │
│   ├── feedback/
│   │   └── correction_logger.py     # Logs user corrections in NPZ file
│   │
│   ├── utils/
│   │   └── signal_processing.py     # Simulated EEG + filters
│   │
│   └── setup.sh                     # Setup script for dependencies
│
├── computer_vision/                # Your grasp and detection modules (CV_PATH)
│   ├── main.py
│   ├── grasp_validation.py
│   ├── grasp_identifier.py
│   ├── object_detection.py
│   ├── hand_landmarks.py
│   └── CONST.py
│
├── robotics/                       # For ROS2 and Arduino communication (ROBOTICS_PATH)
│   ├── ROS2_node.py                # EEG movement ROS2 publisher
│   └── arduino_servo_control.ino   # Servo command listener on Arduino
``
