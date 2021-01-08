/*
  LightSensor.ino
  Reads an analog input on pin P6.0, prints the result to the serial monitor.
  This example code is in the public domain.
 */

// the setup routine runs once when you press reset:
void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(115200);
}

// the loop routine runs over and over again forever:
void loop() {
  // read the input on analog pin P6.0 (A0):
  int sensorValue = analogRead(A0);
  // print out the value you read:
  float volt = 3.3 * sensorValue / 4096.0;
  Serial.print(0);
  Serial.print(" ");
  Serial.print(3.3);
  Serial.print(" ");
  Serial.print(volt, 2);
  Serial.println(" volts");
  delay(20);  // delay set to minimum for real time plotting 
}
