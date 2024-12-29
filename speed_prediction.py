import numpy as np
from tensorflow.keras.models import load_model

# --- Configuration Parameters ---
MODEL_FILE = 'best_model.h5'  # Path to the trained model file
PREDICTION_WINDOW = 5         # Number of past time steps to consider for prediction
SMOOTHING_FACTOR = 0.2       # Smoothing factor for predicted speed (exponential moving average)

# --- Load Trained Model ---
model = load_model(MODEL_FILE)

# --- Prediction Function ---
def predict_speed(sensor_data_window):
    """
    Predicts the desired speed based on a window of sensor data.

    Args:
        sensor_data_window (list): List of sensor data readings for the past time steps.

    Returns:
        float: Predicted speed.
    """
    # Preprocess the sensor data window (replace with your preprocessing function)
    processed_data = preprocess_sensor_data_window(sensor_data_window)

    # Reshape data for the model (samples, timesteps, features)
    input_data = np.reshape(processed_data, (1, PREDICTION_WINDOW, processed_data.shape[1]))

    # Make the prediction
    predicted_speed = model.predict(input_data)[0][0]

    return predicted_speed

# --- Sensor Data Window Management ---
sensor_data_window = []

def update_sensor_data_window(new_sensor_data):
    """
    Updates the sensor data window with the latest reading.

    Args:
        new_sensor_data (numpy.ndarray): Latest sensor data reading.
    """
    global sensor_data_window
    sensor_data_window.append(new_sensor_data)
    if len(sensor_data_window) > PREDICTION_WINDOW:
        sensor_data_window.pop(0)  # Remove oldest data point

# --- Speed Smoothing ---
previous_speed = 0.0

def smooth_speed(predicted_speed):
    """
    Smooths the predicted speed using exponential moving average.

    Args:
        predicted_speed (float): Predicted speed from the model.

    Returns:
        float: Smoothed speed.
    """
    global previous_speed
    smoothed_speed = (
        SMOOTHING_FACTOR * predicted_speed + (1 - SMOOTHING_FACTOR) * previous_speed
    )
    previous_speed = smoothed_speed
    return smoothed_speed

