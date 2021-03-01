unsigned long firstSensor = 0;    // first analog sensor
unsigned int numAvgs = 50;
int handshake = 0;
float vcc = 3.3;
float bits = 4096.0;
int count = 0;
void setup()
{
  // start serial port at 115200 bps:
  Serial.begin(115200);
  count = 0;
}

void loop() {
  if (Serial.available() > 0) {
    handshake = Serial.read();
    if (handshake == 57) 
      Serial.flush();
    else if (handshake == 54) {
      firstSensor = 0;
      for (int count = 0; count < numAvgs; count++) {  
        firstSensor += analogRead(A0);
        delay(1);
      }
      Serial.print("[");
      Serial.print(count);
      Serial.print("]: ");
      Serial.println((firstSensor / numAvgs) / bits * vcc, 3);
      count++;
    }
  }
}
