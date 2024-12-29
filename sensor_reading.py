import time
import board
import busio
import adafruit_mpu6050
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# Initialize I2C bus for MPU6050
i2c = busio.I2C(board.SCL, board.SDA)
mpu = adafruit_mpu6050.MPU6050(i2c)

# Initialize I2C bus for ADS1115
ads = ADS.ADS1115(i2c)

# Define analog input channels on ADS1115
chan0 = AnalogIn(ads, ADS.P0)  # Pressure sensor 1
chan1 = AnalogIn(ads, ADS.P1)  # Pressure sensor 2
chan2 = AnalogIn(ads, ADS.P2)  # Pressure sensor 3
chan3 = AnalogIn(ads, ADS.P3)  # Optional: Additional sensor

# Calibration parameters (replace with actual values)
ACCEL_OFFSET_X = 0.123
ACCEL_OFFSET_Y = -0.456
ACCEL_OFFSET_Z = 9.81
GYRO_OFFSET_X = 0.01
GYRO_OFFSET_Y = -0.02
GYRO_OFFSET_Z = 0.03
PRESSURE_SCALE_0 = 100.0  # Scale factor for pressure sensor 1
PRESSURE_SCALE_1 = 100.0  # Scale factor for pressure sensor 2
PRESSURE_SCALE_2 = 100.0  # Scale factor for pressure sensor 3

# Filtering parameters
ALPHA = 0.2  # Complementary filter constant

# Data storage
accel_data = [0.0, 0.0, 0.0]
gyro_data = [0.0, 0.0, 0.0]
pressure_data = [0.0, 0.0, 0.0]
filtered_angle = 0.0

def calibrate_sensors():
    """
    Calibrates the sensors by averaging readings over a period of time.

    This function should be run with the skateboard stationary and level.
    """
    print("Calibrating sensors...")
    accel_sum = [0.0, 0.0, 0.0]
    gyro_sum = [0.0, 0.0, 0.0]
    num_readings = 100

    for _ in range(num_readings):
        accel_x, accel_y, accel_z = mpu.acceleration
        gyro_x, gyro_y, gyro_z = mpu.gyro
        accel_sum[0] += accel_x
        accel_sum[1] += accel_y
        accel_sum[2] += accel_z
        gyro_sum[0] += gyro_x
        gyro_sum[1] += gyro_y
        gyro_sum[2] += gyro_z
        time.sleep(0.01)

    global ACCEL_OFFSET_X, ACCEL_OFFSET_Y, ACCEL_OFFSET_Z
    global GYRO_OFFSET_X, GYRO_OFFSET_Y, GYRO_OFFSET_Z
    ACCEL_OFFSET_X = accel_sum[0] / num_readings
    ACCEL_OFFSET_Y = accel_sum[1] / num_readings
    ACCEL_OFFSET_Z = accel_sum[2] / num_readings - 9.81  # Subtract gravity
    GYRO_OFFSET_X = gyro_sum[0] / num_readings
    GYRO_OFFSET_Y = gyro_sum[1] / num_readings
    GYRO_OFFSET_Z = gyro_sum[2] / num_readings
    print("Calibration complete.")

def read_accelerometer():
    """
    Reads and calibrates accelerometer data.
    """
    global accel_data
    accel_x, accel_y, accel_z = mpu.acceleration
    accel_data[0] = accel_x - ACCEL_OFFSET_X
    accel_data[1] = accel_y - ACCEL_OFFSET_Y
    accel_data[2] = accel_z - ACCEL_OFFSET_Z
    return accel_data

def read_gyroscope():
    """
    Reads and calibrates gyroscope data.
    """
    global gyro_data
    gyro_x, gyro_y, gyro_z = mpu.gyro
    gyro_data[0] = gyro_x - GYRO_OFFSET_X
    gyro_data[1] = gyro_y - GYRO_OFFSET_Y
    gyro_data[2] = gyro_z - GYRO_OFFSET_Z
    return gyro_data

def read_pressure_sensors():
    """
    Reads and scales pressure sensor data.
    """
    global pressure_data
    pressure_data[0] = chan0.voltage * PRESSURE_SCALE_0
    pressure_data[1] = chan1.voltage * PRESSURE_SCALE_1
    pressure_data[2] = chan2.voltage * PRESSURE_SCALE_2
    return pressure_data

def complementary_filter(accel_angle, gyro_rate, dt):
    """
    Applies a complementary filter to combine accelerometer and gyroscope data.
    """
    global filtered_angle
    filtered_angle = ALPHA * (filtered_angle + gyro_rate * dt) + (1 - ALPHA) * accel_angle
    return filtered_angle

def get_sensor_data():
    """
    Reads and processes all sensor data.
    """
    # Read raw sensor data
    accel_data = read_accelerometer()
    gyro_data = read_gyroscope()
    pressure_data = read_pressure_sensors()

    # Calculate angle from accelerometer
    accel_angle = calculate_angle_from_accel(accel_data)  # Implement this function

    # Calculate angle from gyroscope (integrate gyro rate)
    gyro_rate = gyro_data[1]  # Assuming Y-axis is the rotation axis
    dt = time.monotonic() - previous_time  # Calculate time difference
    gyro_angle = integrate_gyro_rate(gyro_rate, dt)  # Implement this function

    # Apply complementary filter
    filtered_angle = complementary_filter(accel_angle, gyro_rate, dt)

    # Package sensor data
    sensor_data = {
        "acceleration": accel_data,
        "gyroscope": gyro_data,
        "pressure": pressure_data,
        "angle": filtered_angle,
    }

    return sensor_data

# Example usage
calibrate_sensors()
previous_time = time.monotonic()

while True:
    sensor_data = get_sensor_data()
    print(sensor_data)
    time.sleep(0.01)
