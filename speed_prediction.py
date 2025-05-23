import numpy as np
from tensorflow.keras.models import load_model
from data_preprocessing import preprocess_sensor_data # Import the function
from main import config # Import config from main.py

# --- Load Trained Model ---
# Ensure model is loaded after config is available if MODEL_FILE_PATH is used at module load time
# For now, assuming load_model is fine here as it's typically done once.
# If main.py modifies this path before loading, this could be an issue.
# However, MasterConfig has it as a static value.
model = load_model(config.MODEL_FILE_PATH)

# --- Prediction Function ---
def predict_speed(sensor_data_window, prediction_window_config):
    """
    Predicts the desired speed based on a window of sensor data.

    Args:
        sensor_data_window (list): List of sensor data readings for the past time steps.

    Returns:
        float: Predicted speed.
    """
    # Preprocess the sensor data window
    processed_data_list = preprocess_sensor_data_window(sensor_data_window)

    # Stack the list of processed data arrays into a single NumPy array
    # Ensure the shape is (timesteps, features) before reshaping for the model
    if not processed_data_list: # Handle empty list case
        return 0.0 # Or raise an error, or return a default value
    
    processed_data_array = np.array(processed_data_list)

    # Reshape data for the model (samples, timesteps, features)
    # The processed_data_array should already be (timesteps, features_per_timestep)
    # So, if prediction_window_config is the number of timesteps, and processed_data_array.shape[1] is features
    input_data = np.reshape(processed_data_array, (1, prediction_window_config, processed_data_array.shape[1]))

    # Make the prediction
    predicted_speed = model.predict(input_data)[0][0]

    return predicted_speed

def preprocess_sensor_data_window(current_sensor_data_window):
    """
    Preprocesses a window of sensor data.
    Args:
        current_sensor_data_window (list): List of sensor_data dictionaries.
    Returns:
        list: List of preprocessed sensor data arrays.
    """
    processed_window = []
    for sensor_data_snapshot in current_sensor_data_window:
        processed_snapshot = preprocess_sensor_data(sensor_data_snapshot)
        processed_window.append(processed_snapshot)
    return processed_window

# --- Sensor Data Window Management ---
sensor_data_window = [] # This remains a global list specific to this module's state

def update_sensor_data_window(new_sensor_data, prediction_window_config):
    """
    Updates the sensor data window with the latest reading.

    Args:
        new_sensor_data (numpy.ndarray): Latest sensor data reading.
        prediction_window_config (int): The prediction window size from config.
    """
    global sensor_data_window
    sensor_data_window.append(new_sensor_data)
    if len(sensor_data_window) > prediction_window_config:
        sensor_data_window.pop(0)  # Remove oldest data point

# --- Speed Smoothing ---
# Globals for Double Exponential Smoothing state
des_level = None
des_trend = None

def smooth_speed(predicted_speed, current_config): # Pass config
    """
    Smooths the predicted speed using Double Exponential Smoothing (Holt's method).

    Args:
        predicted_speed (float): Predicted speed from the model (current_value).
        current_config (MasterConfig): The application's configuration object.

    Returns:
        float: Smoothed speed (forecast).
    """
    global des_level, des_trend

    if des_level is None: # First time initialization
        des_level = predicted_speed
        des_trend = 0 # Initialize trend to 0, or difference between first two if available
                      # For simplicity, 0 is fine for the first call.
        forecast = predicted_speed
    else:
        # Store previous values before updating
        previous_level = des_level
        previous_trend = des_trend

        # Calculate new level
        new_level = current_config.ALPHA_DES * predicted_speed + \
                    (1 - current_config.ALPHA_DES) * (previous_level + previous_trend)
        
        # Calculate new trend
        new_trend = current_config.BETA_DES * (new_level - previous_level) + \
                    (1 - current_config.BETA_DES) * previous_trend
        
        # Update state for next call
        des_level = new_level
        des_trend = new_trend
        
        # Forecast for the current period (or next, depending on interpretation)
        forecast = new_level + new_trend
        
    return forecast

