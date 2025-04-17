# Translated code from MRL

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
import numpy as np
import random
import time

# Limits of the speed variable
MAX_SPEED = 1.0
MIN_SPEED = 0.2
current_speed = 0.5

# ROS2 node for EEG processing
class EEGMovementNode(Node):
    def __init__(self):
        super().__init__('eeg_movement_node')

        # Publisher for movement commands
        self.mov_pub = self.create_publisher(String, 'movement_command', 10)

        # Publisher for speed adjustment
        self.speed_pub = self.create_publisher(Float32, 'movement_speed', 10)
        
        # Timer to simulate EEG data per second
        self.timer = self.create_timer(1.0, self.execute_eeg_command)

        self.get_logger().info("EEG Movement Node Initialized")
    
    def classify_signal(self, eeg_data):
        # Classes that transform EEG signal into movement command
        movements1 = ["rest", "open_hand", "close_hand", "pinch_grip", "point_finger", "thumbs_up", "wave", "rotate_wrist", "stop"]
        return random.choice(movements1) # To be changed later to integrate with the Neurosciences Department
    
    def adjust_speed(self, intensity):
        global current_speed
        current_speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * intensity

        # Publish speed to Arduino
        speed_msg = Float32()
        speed_msg.data = current_speed
        self.speed_pub.publish(speed_msg)

        self.get_logger().info(f"New speed: {current_speed:.2f}")

    def execute_eeg_command(self):
        # Application of the EEG data (so classify movement, adjust speed, publish commands, etc)
        eeg_data = np.random.randn(8) # To be changed with the Neurosciences Department
        movement = self.classify_signal(eeg_data)
        intensity = abs(np.random.uniform(0.2, 1.0))

        self.get_logger().info(f"EEG detected: {movement} at intensity {intensity:.2f}")

        # Adjust speed
        self.adjust_speed(intensity)

        # Publish movement command
        command_msg = String()
        command_msg.data = movement
        self.mov_pub.publish(command_msg)

        self.get_logger().info(f"Executing movement: {movement}")

def main(args=None):
    rclpy.init(args=args)
    node = EEGMovementNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
