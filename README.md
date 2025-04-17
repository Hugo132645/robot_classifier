# robot_classifier
Machine learning classifier for motor imagery.

## Structure

inmoov_prosthetic_eeg/
│
├── README.md
├── requirements.txt
├── setup.sh
├── config/
│   └── eeg_config.yaml
│
├── data/
│   └── motor_imagery/
│       └── raw/
│       └── preprocessed/
│
├── eeg/
│   ├── collect_data.py
│   ├── preprocess.py
│   ├── train_classifier.py
│   └── model.py
│
├── firmware/
│   └── arduino_inmoov_controller.ino
│
├── prosthetic_control/
│   ├── live_eeg_predict.py
│   └── actuator_interface.py
│
├── ros2_integration/
│   ├── launch/
│   └── nodes/
│       ├── eeg_node.py
│       └── motor_control_node.py
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
└── utils/
    └── signal_processing.py
