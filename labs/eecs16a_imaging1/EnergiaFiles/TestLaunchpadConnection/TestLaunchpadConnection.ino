int handshake = 0;

void setup()
{
  // start serial port at 115200 bps:
  Serial.begin(115200);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for Leonardo only
  }
}

void loop() {
  // if we get a valid byte, the serial port is connected to the launchpad
  if (Serial.available() > 0) {
    handshake = Serial.read();
    if (handshake == 57)  // Ascii code for numerical 9
      Serial.flush();
    else if (handshake == 54) { // Ascii code for 6 so serial monitor works
      Serial.println("Your launchpad has successfully uploaded this script, and connected to the serial monitor!");
    }
  }
}
