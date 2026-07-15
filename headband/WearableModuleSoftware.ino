#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

// ── Calibration offsets (set after calibration runs) ─────────────
float offset_ax = 0, offset_ay = 0, offset_az = 0;
float offset_gx = 0, offset_gy = 0, offset_gz = 0;

// ── Calibration function ─────────────────────────────────────────
void calibrateMPU() {
  Serial.println("Calibrating MPU6050...");
  Serial.println("Keep sensor flat and still for 5 seconds!");
  delay(3000);

  float ax=0, ay=0, az=0, gx=0, gy=0, gz=0;
  int samples = 500;

  for (int i = 0; i < samples; i++) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    ax += a.acceleration.x;
    ay += a.acceleration.y;
    az += a.acceleration.z;
    gx += g.gyro.x;
    gy += g.gyro.y;
    gz += g.gyro.z;
    delay(5);
  }

  // Average the samples
  offset_ax = ax / samples;
  offset_ay = ay / samples;
  offset_az = (az / samples) - 9.81; // subtract gravity from Z
  offset_gx = gx / samples;
  offset_gy = gy / samples;
  offset_gz = gz / samples;

  Serial.println("--- Calibration complete ---");
  Serial.print("Accel offsets X: "); Serial.print(offset_ax);
  Serial.print(" Y: ");              Serial.print(offset_ay);
  Serial.print(" Z: ");              Serial.println(offset_az);
  Serial.print("Gyro offsets  X: "); Serial.print(offset_gx);
  Serial.print(" Y: ");              Serial.print(offset_gy);
  Serial.print(" Z: ");              Serial.println(offset_gz);
  Serial.println("----------------------------");
}

void setup(void) {
  Serial.begin(115200);
  Wire.begin(8, 9); // SDA=GPIO8, SCL=GPIO9
  while (!Serial)
    delay(10);

  Serial.println("Adafruit MPU6050 test!");

  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) { delay(10); }
  }
  Serial.println("MPU6050 Found!");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  Serial.print("Accelerometer range set to: ");
  switch (mpu.getAccelerometerRange()) {
    case MPU6050_RANGE_2_G:  Serial.println("+-2G");  break;
    case MPU6050_RANGE_4_G:  Serial.println("+-4G");  break;
    case MPU6050_RANGE_8_G:  Serial.println("+-8G");  break;
    case MPU6050_RANGE_16_G: Serial.println("+-16G"); break;
  }

  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  Serial.print("Gyro range set to: ");
  switch (mpu.getGyroRange()) {
    case MPU6050_RANGE_250_DEG:  Serial.println("+- 250 deg/s");  break;
    case MPU6050_RANGE_500_DEG:  Serial.println("+- 500 deg/s");  break;
    case MPU6050_RANGE_1000_DEG: Serial.println("+- 1000 deg/s"); break;
    case MPU6050_RANGE_2000_DEG: Serial.println("+- 2000 deg/s"); break;
  }

  mpu.setFilterBandwidth(MPU6050_BAND_10_HZ); // 10Hz for road vibration filtering
  Serial.print("Filter bandwidth set to: ");
  switch (mpu.getFilterBandwidth()) {
    case MPU6050_BAND_260_HZ: Serial.println("260 Hz"); break;
    case MPU6050_BAND_184_HZ: Serial.println("184 Hz"); break;
    case MPU6050_BAND_94_HZ:  Serial.println("94 Hz");  break;
    case MPU6050_BAND_44_HZ:  Serial.println("44 Hz");  break;
    case MPU6050_BAND_21_HZ:  Serial.println("21 Hz");  break;
    case MPU6050_BAND_10_HZ:  Serial.println("10 Hz");  break;
    case MPU6050_BAND_5_HZ:   Serial.println("5 Hz");   break;
  }

  // Run calibration on startup
  calibrateMPU();

  Serial.println("");
  delay(100);
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // Apply calibration offsets
  float cal_ax = a.acceleration.x - offset_ax;
  float cal_ay = a.acceleration.y - offset_ay;
  float cal_az = a.acceleration.z - offset_az;
  float cal_gx = g.gyro.x - offset_gx;
  float cal_gy = g.gyro.y - offset_gy;
  float cal_gz = g.gyro.z - offset_gz;

  // Calculate pitch angle
  float pitch = atan2(cal_ax,
                sqrt(cal_ay * cal_ay + cal_az * cal_az))
                * 180.0 / PI;

  // Print calibrated accelerometer values
  Serial.print("Acceleration X: "); Serial.print(cal_ax);
  Serial.print(", Y: ");            Serial.print(cal_ay);
  Serial.print(", Z: ");            Serial.print(cal_az);
  Serial.println(" m/s^2");

  // Print calibrated gyro values
  Serial.print("Rotation X: "); Serial.print(cal_gx);
  Serial.print(", Y: ");        Serial.print(cal_gy);
  Serial.print(", Z: ");        Serial.print(cal_gz);
  Serial.println(" rad/s");

  // Print pitch angle
  Serial.print("Pitch: ");
  Serial.print(pitch);
  Serial.println(" degrees");

  Serial.println("");
  delay(500);
}
