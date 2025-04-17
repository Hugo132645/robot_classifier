import serial

arduino = serial.Serial('/dev/ttyACM0', 9600)

def send_command(command):
    if command == "GRIP":
        arduino.write(b'G')
    elif command == "RELEASE":
        arduino.write(b'R')
