import time
import threading
import queue
import numpy as np
import RPi.GPIO as GPIO
from sensor_reading import get_sensor_data, calibrate_sensors
from data_preprocessing import preprocess_sensor_data
import speed_prediction # Import the module itself
import motor_control # Import the module itself

# --- Master Configuration ---
class MasterConfig:
    # From data_preprocessing.py
    WINDOW_SIZE = 10
    POLYORDER = 3
    CUTOFF = 5.0 # Low-pass cutoff
    FS = 50.0
    # New for band-pass filter
    LOW_CUTOFF_BP = 2.0
    HIGH_CUTOFF_BP = 10.0
    ORDER_BP = 3
    RMS_WINDOW_SIZE = 5 # For RMS feature

    # From speed_prediction.py
    PREDICTION_WINDOW = 5  # This is also used in model_training.py as timesteps
    # SMOOTHING_FACTOR = 0.2 # Removed, replaced by DES parameters
    ALPHA_DES = 0.4  # Level smoothing factor for Double Exponential Smoothing
    BETA_DES = 0.3   # Trend smoothing factor for Double Exponential Smoothing
    MODEL_FILE_PATH = 'best_model.h5' # Shared with model_training.py

    # From motor_control.py
    PWM_PIN = 18
    PWM_FREQUENCY = 50
    MAX_SPEED = 100
    MIN_SPEED = 0
    ACCELERATION_RATE = 10
    BRAKING_RATE = 20
    SPEED_DEADZONE = 5

    # From model_training.py
    DATA_FILE = 'sensor_data.csv'
    TEST_SIZE = 0.2
    VALIDATION_SPLIT = 0.2
    EPOCHS = 100
    BATCH_SIZE = 32
    LSTM_UNITS = 64
    DROPOUT_RATE = 0.2
    L1_REG = 0.001
    L2_REG = 0.001
    PATIENCE = 10
    RANDOM_STATE = 42 # For train_test_split

    # From sensor_reading.py
    ACCEL_OFFSET_X = 0.123
    ACCEL_OFFSET_Y = -0.456
    ACCEL_OFFSET_Z = 9.81
    GYRO_OFFSET_X = 0.01
    GYRO_OFFSET_Y = -0.02
    GYRO_OFFSET_Z = 0.03
    PRESSURE_SCALE_0 = 100.0
    PRESSURE_SCALE_1 = 100.0
    PRESSURE_SCALE_2 = 100.0
    ALPHA = 0.2  # Complementary filter constant
    # Standard deviations from calibration
    ACCEL_STD_X = 0.0
    ACCEL_STD_Y = 0.0
    ACCEL_STD_Z = 0.0
    GYRO_STD_X = 0.0
    GYRO_STD_Y = 0.0
    GYRO_STD_Z = 0.0

    # For motor_control.py jerk limitation
    MAX_JERK = 5.0

    # For main.py PID tuning placeholder
    PID_P = 1.0
    PID_I = 0.1
    PID_D = 0.05


# --- Global Variables ---
config = MasterConfig() # Create an instance of the config
sensor_data_queue = queue.Queue()
predicted_speed_queue = queue.Queue()
emergency_stop_flag = threading.Event()
current_speed = 0.0
previous_time = time.time()

# --- Threading Functions ---
def sensor_reading_thread():
    """
    Reads sensor data and puts it into the queue.
    """
    calibrate_sensors()
    while not emergency_stop_flag.is_set():
        try:
            sensor_data = get_sensor_data()
            sensor_data_queue.put(sensor_data)
        except Exception as e:
            print(f"Error in sensor reading thread: {e}")
            emergency_stop_flag.set()
        time.sleep(0.01)

def data_preprocessing_thread():
    """
    Processes sensor data and puts it into the prediction queue.
    """
    while not emergency_stop_flag.is_set():
        try:
            sensor_data = sensor_data_queue.get()
            processed_data = preprocess_sensor_data(sensor_data)
            predicted_speed_queue.put(processed_data)
        except Exception as e:
            print(f"Error in data preprocessing thread: {e}")
            emergency_stop_flag.set()

def speed_prediction_thread():
    """
    Predicts speed based on processed sensor data.
    """
    while not emergency_stop_flag.is_set():
        try:
            processed_data = predicted_speed_queue.get() # This is actually a single feature vector
            speed_prediction.update_sensor_data_window(processed_data, config.PREDICTION_WINDOW) 
            if len(speed_prediction.sensor_data_window) == config.PREDICTION_WINDOW: 
                raw_predicted_speed = speed_prediction.predict_speed(speed_prediction.sensor_data_window, config.PREDICTION_WINDOW)
                
                # Apply smoothing using the new Double Exponential Smoothing
                smoothed_predicted_speed = speed_prediction.smooth_speed(raw_predicted_speed, config) # Pass config
                
                # Put the smoothed speed onto the queue for motor control
                predicted_speed_queue.put(smoothed_predicted_speed)
        except Exception as e:
            print(f"Error in speed prediction thread: {e}")
            emergency_stop_flag.set()

def motor_control_thread():
    """
    Controls the motor based on predicted speed.
    """
    global current_speed, previous_time
    while not emergency_stop_flag.is_set():
        try:
            predicted_speed = predicted_speed_queue.get()

            # Calculate elapsed time
            current_time = time.time()
            dt = current_time - previous_time
            previous_time = current_time

            # --- Safety Check (Example) ---
            if is_emergency_situation():  # Replace with your emergency situation detection
                motor_control.emergency_stop() 
                emergency_stop_flag.set()
                break

            # --- Speed Control ---
            if predicted_speed > current_speed:
                # Accelerate towards target speed, passing config for MAX_JERK etc.
                current_speed = motor_control.accelerate(current_speed, predicted_speed, dt, config)
            elif predicted_speed < current_speed:
                # Decelerate (Brake)
                # Create dummy inputs for adaptive braking
                decel_profile = [0.95, 0.85, 0.75] # Example profile
                env_cond = {"road_condition": "dry", "incline_angle": -2.0} # Example conditions
                
                # Get adaptive braking rate
                adaptive_rate_val = motor_control.calculate_adaptive_braking(
                    current_speed, 
                    decel_profile, 
                    env_cond, 
                    config
                )
                
                # Use this rate with the brake function.
                # Brake towards the `predicted_speed` if it's lower, using the adaptive rate.
                current_speed = motor_control.brake(
                    current_speed, 
                    dt, 
                    config, 
                    target_speed=predicted_speed, # Brake towards the (lower) predicted speed
                    custom_braking_rate=adaptive_rate_val
                )
            # If predicted_speed == current_speed, do nothing or maintain speed via set_motor_speed

            # Set the motor speed
            motor_control.set_motor_speed(current_speed) 

        except Exception as e:
            print(f"Error in motor control thread: {e}")
            emergency_stop_flag.set()
        time.sleep(0.01)

# --- Emergency Stop Handling ---
def is_emergency_situation():
    """
    Checks for emergency situations (replace with your actual logic).

    Returns:
        bool: True if an emergency situation is detected, False otherwise.
    """
    # Example: Check for sudden tilting or impact
    # ... (Your complex emergency detection logic here)
    return False

# --- Dynamic PID Tuning Service (Placeholder) ---
def dynamic_pid_tuning_service(app_config): # Renamed parameter to avoid conflict
    """
    Simulates a dynamic PID tuning service.
    In a real system, this would analyze performance and adjust PID gains.
    """
    # Example: Simulate some performance metric analysis
    # For now, just prints current PID values from config.
    # In a real scenario, it might modify app_config.PID_P etc. if they were not read-only.
    # Or, it could communicate with a PID controller object.
    print(f"Dynamic PID Tuning Service: Analyzing performance metrics (simulated).")
    print(f"Current PID Gains - P: {app_config.PID_P}, I: {app_config.PID_I}, D: {app_config.PID_D}.")
    # Placeholder for logic that might adjust these values if they were part of a mutable object
    # e.g., pid_controller.Kp = new_P_value

# --- Main Execution ---
if __name__ == '__main__':
    last_pid_tune_time = time.time()
    pid_tune_interval = 10 # seconds

    try:
        # Initialize threads
        sensor_thread = threading.Thread(target=sensor_reading_thread)
        preprocessing_thread = threading.Thread(target=data_preprocessing_thread)
        prediction_thread = threading.Thread(target=speed_prediction_thread)
        motor_thread = threading.Thread(target=motor_control_thread)

        # Start threads
        sensor_thread.start()
        preprocessing_thread.start()
        prediction_thread.start()
        motor_thread.start()

        # Keep the main thread alive
        while not emergency_stop_flag.is_set():
            current_loop_time = time.time()
            if current_loop_time - last_pid_tune_time > pid_tune_interval:
                dynamic_pid_tuning_service(config) # Call the PID tuning service
                last_pid_tune_time = current_loop_time
            time.sleep(1)

    except KeyboardInterrupt:
        print("Exiting...")
        emergency_stop_flag.set()

    finally:
        # Wait for threads to finish
        sensor_thread.join()
        preprocessing_thread.join()
        prediction_thread.join()
        motor_thread.join()

        # Cleanup GPIO
        GPIO.cleanup()
