// Continuous IMU → Serial @ 100 Hz with LED indicators
// LED meanings:
//   Fast blink = IMU init failed
//   Solid ON   = IMU OK, streaming
//   Flicker    = every data output

#include <Arduino.h>
#include <Wire.h>
#include "LSM6DS3.h"

const float FS_HZ = 100.0;
const unsigned long PERIOD_US = (unsigned long)(1e6 / FS_HZ);
unsigned long last_us = 0;

LSM6DS3 imuA(I2C_MODE, 0x6A);
LSM6DS3 imuB(I2C_MODE, 0x6B);
LSM6DS3* IMU = &imuA;

void failBlink() {
  pinMode(LED_BUILTIN, OUTPUT);
  while (true) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(250);
    digitalWrite(LED_BUILTIN, LOW);
    delay(250);
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  Serial.begin(115200);
  Wire.begin();
  delay(100);

  // Try 0x6A first, then 0x6B
  if (IMU->begin() != 0) {
    IMU = &imuB;
    if (IMU->begin() != 0) {
      Serial.println("ERROR: IMU init failed (0x6A and 0x6B)");
      failBlink();  // blink fast forever
    }
  }

  Serial.println("timestamp_ms,ax,ay,az,gx,gy,gz,tempC");
  last_us = micros();

  // Solid LED = ready/streaming
  digitalWrite(LED_BUILTIN, HIGH);
}

void loop() {
  unsigned long now = micros();
  if (now - last_us >= PERIOD_US) {
    last_us += PERIOD_US;

    float ax = IMU->readFloatAccelX();
    float ay = IMU->readFloatAccelY();
    float az = IMU->readFloatAccelZ();
    float gx = IMU->readFloatGyroX();
    float gy = IMU->readFloatGyroY();
    float gz = IMU->readFloatGyroZ();
    float tc = IMU->readTempC();

    // Flicker LED briefly on each sample
    digitalWrite(LED_BUILTIN, LOW);
    delayMicroseconds(100);
    digitalWrite(LED_BUILTIN, HIGH);

    // Print CSV line
    Serial.print(millis()); Serial.print(",");
    Serial.print(ax,6); Serial.print(","); Serial.print(ay,6); Serial.print(","); Serial.print(az,6); Serial.print(",");
    Serial.print(gx,6); Serial.print(","); Serial.print(gy,6); Serial.print(","); Serial.print(gz,6); Serial.print(",");
    Serial.println(tc,2);
  }
}
