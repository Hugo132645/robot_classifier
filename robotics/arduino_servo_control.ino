#include <Servo.h>
#include <ros.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32.h>

// First define servos
Servo fingers[6]; // including wrist
int finger_pins[6] = {3, 5, 6, 9, 10, 11}; // The PWM pins on which each servo is connected

// Default speed
float current_speed = 0.5;

// ROS Node
ros::NodeHandle nh;

void moveHand(int thumb, int index, int middle, int ring, int pinky, int wrist) {
  fingers[0].write(thumb);
  fingers[1].write(index);
  fingers[2].write(middle);
  fingers[3].write(ring);
  fingers[4].write(pinky);
  fingers[5].write(wrist);
}

void movementCallback(const std_msgs::String& msg) {
  String movement = msg.data;

  if (movement == "rest") moveHand(90,90,90,90,90,90);
  else if (movement == "open_hand") moveHand(0,0,0,0,0,90);
  else if (movement == "close_hand") moveHand(180,180,180,180,180,90);
  else if (movement == "pinch_grip") moveHand(180,180,0,0,0,90);
  else if (movement == "point_finger") moveHand(0,180,0,0,0,90);
  else if (movement == "thumbs_up") moveHand(180,0,0,0,0,90);
  else if (movement == "wave") {
    for (int i = 0; i < 3; i++) {
      moveHand(0,180,0,180,0,180);
      delay(300);
      moveHand(180,0,180,0,180,0);
      delay(300);
    }
  }
  else if (movement == "rotate_wrist") fingers[5].write(180);
  else if (movement == "stop") moveHand(90,90,90,90,90,90);
}

// Speed command callback
void speedCallback(const std_msgs::Float32& msg) {
  current_speed = msg.data;
}

// ROS subscribers
ros::Subscriber<std_msgs::String> sub_movement("movement_command", movementCallback);
ros::Subscriber<std_msgs::Float32> sub_speed("movement_speed", speedCallback);

void setup() {
  
  nh.initNode();
  nh.subscribe(sub_movement);
  nh.subscribe(sub_speed);

  // Attach servos
  for (int i = 0; i < 6; i++) {
    fingers[i].attach(finger_pins[i]);
  }

  moveHand(90,90,90,90,90,90);

}

void loop() {

  nh.spinOnce();
  delay(10);

}
