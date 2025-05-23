import time
import board
import busio
import adafruit_mpu6050
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import math
from main import config # Import config from main.py

# Initialize I2C bus for MPU6050
i2c = busio.I2C(board.SCL, board.SDA) # Hardware specific, not in config
mpu = adafruit_mpu6050.MPU6050(i2c)   # Hardware specific, not in config

# Initialize I2C bus for ADS1115
ads = ADS.ADS1115(i2c) # Hardware specific, not in config

# Define analog input channels on ADS1115
chan0 = AnalogIn(ads, ADS.P0)  # Hardware specific
chan1 = AnalogIn(ads, ADS.P1)  # Hardware specific
chan2 = AnalogIn(ads, ADS.P2)  # Hardware specific
chan3 = AnalogIn(ads, ADS.P3)  # Hardware specific

# Calibration parameters - initialized from config, can be updated by calibrate_sensors
ACCEL_OFFSET_X = config.ACCEL_OFFSET_X
ACCEL_OFFSET_Y = config.ACCEL_OFFSET_Y
ACCEL_OFFSET_Z = config.ACCEL_OFFSET_Z
GYRO_OFFSET_X = config.GYRO_OFFSET_X
GYRO_OFFSET_Y = config.GYRO_OFFSET_Y
GYRO_OFFSET_Z = config.GYRO_OFFSET_Z

# Standard deviations - initialized from config, updated by calibrate_sensors
ACCEL_STD_X = config.ACCEL_STD_X
ACCEL_STD_Y = config.ACCEL_STD_Y
ACCEL_STD_Z = config.ACCEL_STD_Z
GYRO_STD_X = config.GYRO_STD_X
GYRO_STD_Y = config.GYRO_STD_Y
GYRO_STD_Z = config.GYRO_STD_Z

PRESSURE_SCALE_0 = config.PRESSURE_SCALE_0
PRESSURE_SCALE_1 = config.PRESSURE_SCALE_1
PRESSURE_SCALE_2 = config.PRESSURE_SCALE_2
# ALPHA is used directly from config.ALPHA in complementary_filter

# Data storage (module-level state)
accel_data_global_storage = [0.0, 0.0, 0.0] # Renamed to avoid confusion in read_accelerometer
gyro_data_global_storage = [0.0, 0.0, 0.0]   # Renamed
pressure_data_global_storage = [0.0, 0.0, 0.0] # Renamed
filtered_pitch_angle = 0.0 # Renamed from filtered_angle for clarity
integrated_gyro_angle = 0.0
gyro_integration_call_count = 0 # Counter for integrate_gyro_rate damping

# For dt calculation in get_sensor_data
sensor_previous_time = time.monotonic() # Initialize at module load

def calibrate_sensors():
    """
    Calibrates the sensors by averaging readings over a period of time.

    This function should be run with the skateboard stationary and level.
    """
    print("Calibrating sensors...")
    accel_readings = [[], [], []] # For X, Y, Z axes
    gyro_readings = [[], [], []]  # For X, Y, Z axes
    num_readings = 100

    for _ in range(num_readings):
        ax, ay, az = mpu.acceleration
        gx, gy, gz = mpu.gyro
        accel_readings[0].append(ax)
        accel_readings[1].append(ay)
        accel_readings[2].append(az)
        gyro_readings[0].append(gx)
        gyro_readings[1].append(gy)
        gyro_readings[2].append(gz)
        time.sleep(0.01)

    global ACCEL_OFFSET_X, ACCEL_OFFSET_Y, ACCEL_OFFSET_Z
    global GYRO_OFFSET_X, GYRO_OFFSET_Y, GYRO_OFFSET_Z
    global ACCEL_STD_X, ACCEL_STD_Y, ACCEL_STD_Z
    global GYRO_STD_X, GYRO_STD_Y, GYRO_STD_Z

    ACCEL_OFFSET_X = np.mean(accel_readings[0])
    ACCEL_OFFSET_Y = np.mean(accel_readings[1])
    ACCEL_OFFSET_Z = np.mean(accel_readings[2]) - 9.81  # Subtract gravity for Z offset
    GYRO_OFFSET_X = np.mean(gyro_readings[0])
    GYRO_OFFSET_Y = np.mean(gyro_readings[1])
    GYRO_OFFSET_Z = np.mean(gyro_readings[2])

    ACCEL_STD_X = np.std(accel_readings[0])
    ACCEL_STD_Y = np.std(accel_readings[1])
    ACCEL_STD_Z = np.std(accel_readings[2])
    GYRO_STD_X = np.std(gyro_readings[0])
    GYRO_STD_Y = np.std(gyro_readings[1])
    GYRO_STD_Z = np.std(gyro_readings[2])

    print("Calibration complete.")
    print(f"Accelerometer Offsets (X,Y,Z): {ACCEL_OFFSET_X:.3f}, {ACCEL_OFFSET_Y:.3f}, {ACCEL_OFFSET_Z:.3f}")
    print(f"Gyroscope Offsets (X,Y,Z): {GYRO_OFFSET_X:.3f}, {GYRO_OFFSET_Y:.3f}, {GYRO_OFFSET_Z:.3f}")
    print(f"Accelerometer X axis noise (std dev): {ACCEL_STD_X:.3f}")
    print(f"Accelerometer Y axis noise (std dev): {ACCEL_STD_Y:.3f}")
    print(f"Accelerometer Z axis noise (std dev): {ACCEL_STD_Z:.3f}")
    print(f"Gyroscope X axis noise (std dev): {GYRO_STD_X:.3f}")
    print(f"Gyroscope Y axis noise (std dev): {GYRO_STD_Y:.3f}")
    print(f"Gyroscope Z axis noise (std dev): {GYRO_STD_Z:.3f}")


def read_accelerometer():
    """
    Reads and calibrates accelerometer data.
    """
    # This function returns the calibrated data but also updates a global for some reason.
    # Let's make it return only, and the caller can decide to store it globally if needed.
    # For now, maintaining original behavior of updating accel_data_global_storage.
    global accel_data_global_storage
    accel_x, accel_y, accel_z = mpu.acceleration
    accel_data_global_storage[0] = accel_x - ACCEL_OFFSET_X
    accel_data_global_storage[1] = accel_y - ACCEL_OFFSET_Y
    accel_data_global_storage[2] = accel_z - ACCEL_OFFSET_Z
    return list(accel_data_global_storage) # Return a copy

def read_gyroscope():
    """
    Reads and calibrates gyroscope data.
    """
    global gyro_data_global_storage
    gyro_x, gyro_y, gyro_z = mpu.gyro
    gyro_data_global_storage[0] = gyro_x - GYRO_OFFSET_X
    gyro_data_global_storage[1] = gyro_y - GYRO_OFFSET_Y
    gyro_data_global_storage[2] = gyro_z - GYRO_OFFSET_Z
    return list(gyro_data_global_storage) # Return a copy

def read_pressure_sensors():
    """
    Reads and scales pressure sensor data.
    """
    global pressure_data_global_storage
    pressure_data_global_storage[0] = chan0.voltage * PRESSURE_SCALE_0
    pressure_data_global_storage[1] = chan1.voltage * PRESSURE_SCALE_1
    pressure_data_global_storage[2] = chan2.voltage * PRESSURE_SCALE_2
    return list(pressure_data_global_storage) # Return a copy

def complementary_filter(current_pitch_accel, gyro_rate_pitch, dt):
    """
    Applies a complementary filter to combine accelerometer pitch and gyroscope pitch rate.
    Updates the global filtered_pitch_angle.
    Args:
        current_pitch_accel (float): Pitch angle calculated from accelerometer (radians).
        gyro_rate_pitch (float): Gyroscope rate around the pitch axis (radians/sec).
        dt (float): Time delta.
    Returns:
        float: Filtered pitch angle (radians).
    """
    global filtered_pitch_angle # This global is updated by the filter
    filtered_pitch_angle = config.ALPHA * (filtered_pitch_angle + gyro_rate_pitch * dt) + \
                           (1 - config.ALPHA) * current_pitch_accel
    return filtered_pitch_angle

def get_sensor_data():
    """
    Reads and processes all sensor data.
    Updates sensor_previous_time for dt calculation.
    """
    global sensor_previous_time # Manage previous_time for dt calculation globally within this module

    current_time = time.monotonic()
    dt = current_time - sensor_previous_time
    sensor_previous_time = current_time

    # Read raw sensor data
    current_accel_data = read_accelerometer() 
    current_gyro_data = read_gyroscope()   
    current_pressure_data = read_pressure_sensors() 

    # Calculate pitch and roll from accelerometer (pitch in radians, roll in degrees)
    # calculate_angle_from_accel now returns (pitch_rad, roll_deg)
    pitch_accel_rad, roll_accel_deg = calculate_angle_from_accel(current_accel_data)

    # Gyro data for pitch axis (assuming Y-axis for pitch)
    # Convert gyro rate from deg/s (MPU6050 default) to rad/s for filter and integration
    # MPU6050 returns values in degrees/sec.
    gyro_rate_pitch_dps = current_gyro_data[1] # Assuming Y-axis is pitch
    gyro_rate_pitch_rps = math.radians(gyro_rate_pitch_dps) # Convert to radians/second

    # Integrate gyro rate for a raw gyro-based angle (pitch)
    # This integrated_gyro_angle is also in radians
    integrated_pitch_gyro = integrate_gyro_rate(gyro_rate_pitch_rps, dt) # dt is now correctly calculated

    # Apply complementary filter for pitch
    # filtered_pitch_angle (global) is updated by this function
    # It uses accelerometer pitch (radians) and gyro pitch rate (radians/sec)
    final_filtered_pitch_rad = complementary_filter(pitch_accel_rad, gyro_rate_pitch_rps, dt)
    final_filtered_pitch_deg = math.degrees(final_filtered_pitch_rad) # Convert to degrees for output if desired

    # Package sensor data
    sensor_data_pkg = { 
        "acceleration": current_accel_data, 
        "gyroscope": current_gyro_data,     # In degrees/sec from MPU6050
        "pressure": current_pressure_data,  
        "angle_pitch": final_filtered_pitch_deg, # Filtered pitch in degrees
        "angle_roll": roll_accel_deg,          # Raw roll from accelerometer in degrees
        "raw_integrated_gyro_pitch": math.degrees(integrated_pitch_gyro) # Raw integrated gyro pitch in degrees
    }
    return sensor_data_pkg

def calculate_angle_from_accel(accel_data_local):
    """
    Calculates pitch (radians) and roll (degrees) angles from accelerometer data.
    Args:
        accel_data_local (list or tuple): Calibrated accelerometer data [ax, ay, az].
    Returns:
        tuple: (pitch_radians, roll_degrees)
    """
    ax, ay, az = accel_data_local
    # Calculate pitch in radians
    # Pitch is rotation around Y-axis (ax vs az)
    pitch_rad = math.atan2(-ax, math.sqrt(ay**2 + az**2)) # Common convention: -ax for pitch
    
    # Calculate roll in radians and then convert to degrees
    # Roll is rotation around X-axis (ay vs az)
    roll_rad = math.atan2(ay, az) # Simpler form: math.atan2(ay, az) if Z is dominant vertical component
    # A more robust roll considering ax: roll_rad = math.atan2(ay, math.sqrt(ax**2 + az**2))
    # Given typical MPU6050 orientation, ay vs az is common for roll if board is flat.
    # Let's use the one from the prompt: math.atan2(accel_data[1], math.sqrt(accel_data[0]**2 + accel_data[2]**2))
    # This means ay vs sqrt(ax^2 + az^2)
    roll_rad_complex = math.atan2(ay, math.sqrt(ax**2 + az**2))
    roll_deg = math.degrees(roll_rad_complex)
    
    return pitch_rad, roll_deg

def integrate_gyro_rate(gyro_rate_rps, dt):
    """
    Integrates gyroscope rate (radians/sec) to get an angle (radians).
    Includes a simple damping mechanism every 100 calls.
    Args:
        gyro_rate_rps (float): Gyroscope rate in radians per second.
        dt (float): Time delta since last update.
    Returns:
        float: Integrated gyro angle in radians.
    """
    global integrated_gyro_angle, gyro_integration_call_count
    
    # TODO: Implement sophisticated drift correction algorithm here (e.g., based on sensor fusion with accelerometer during static periods)
    
    integrated_gyro_angle += gyro_rate_rps * dt
    
    gyro_integration_call_count += 1
    if gyro_integration_call_count >= 100:
        integrated_gyro_angle *= 0.999  # Apply slight damping
        gyro_integration_call_count = 0 # Reset counter
        
    return integrated_gyro_angle

# Example usage
if __name__ == '__main__':
    calibrate_sensors()
    # sensor_previous_time is already initialized globally and managed by get_sensor_data()

    while True:
        sensor_data_values = get_sensor_data() # previous_time is now handled by get_sensor_data
        print(sensor_data_values)
        time.sleep(0.01)
