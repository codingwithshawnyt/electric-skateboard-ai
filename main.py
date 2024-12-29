import time
import threading
import queue
import numpy as np
import RPi.GPIO as GPIO
from sensor_reading import get_sensor_data, calibrate_sensors
from data_preprocessing import preprocess_sensor_data
from speed_prediction import predict_speed, update_sensor_data_window
from motor_control import set_motor_speed, emergency_stop

# --- Configuration Parameters ---
# ... (Include all the parameters from previous files)

# --- Global Variables ---
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
            processed_data = predicted_speed_queue.get()
            update_sensor_data_window(processed_data)
            if len(sensor_data_window) == PREDICTION_WINDOW:
                predicted_speed = predict_speed(sensor_data_window)
                # ... (Apply any additional smoothing or filtering to predicted_speed)
                predicted_speed_queue.put(predicted_speed)
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
                emergency_stop()
                emergency_stop_flag.set()
                break

            # --- Speed Control ---
            if predicted_speed > current_speed:
                # Accelerate towards target speed
                current_speed = accelerate(current_speed, predicted_speed, dt)
            else:
                # Brake towards target speed
                current_speed = accelerate(current_speed, predicted_speed, dt)
                # Or apply more aggressive braking:
                # current_speed = brake(current_speed, dt)

            # Set the motor speed
            set_motor_speed(current_speed)

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

# --- Main Execution ---
if __name__ == '__main__':
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
