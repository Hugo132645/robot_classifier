#include <Servo.h>

Servo gripServo;

void setup() {
  Serial.begin(9600);
  gripServo.attach(9);
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'G') {
      gripServo.write(45); // Grip
    } else if (cmd == 'R') {
      gripServo.write(90); // Release
    }
  }
}
