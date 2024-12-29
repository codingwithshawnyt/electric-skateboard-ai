import numpy as np
from scipy.signal import butter, lfilter, savgol_filter

# --- Configuration Parameters ---
WINDOW_SIZE = 10  # Window size for moving average filter
POLYORDER = 3     # Polynomial order for Savitzky-Golay filter
CUTOFF = 5.0    # Cutoff frequency for low-pass filter (Hz)
FS = 50.0       # Sampling frequency (Hz)

# --- Feature Engineering Functions ---
def calculate_jerk(acceleration_data, dt=0.02):
    """
    Calculates jerk (rate of change of acceleration) from acceleration data.

    Args:
        acceleration_data (list): List of acceleration values.
        dt (float): Time interval between acceleration readings.

    Returns:
        list: Jerk values.
    """
    jerk = np.gradient(acceleration_data, dt)
    return jerk

def calculate_pressure_difference(pressure_data):
    """
    Calculates the difference in pressure between sensor pairs.

    Args:
        pressure_data (list): List of pressure values from multiple sensors.

    Returns:
        list: Pressure difference values.
    """
    pressure_diff = [
        pressure_data[0] - pressure_data[1],
        pressure_data[1] - pressure_data[2],
        pressure_data[0] - pressure_data[2],
    ]
    return pressure_diff

# --- Filtering Functions ---
def butter_lowpass(cutoff, fs, order=5):
    """
    Designs a Butterworth low-pass filter.

    Args:
        cutoff (float): Cutoff frequency (Hz).
        fs (float): Sampling frequency (Hz).
        order (int): Filter order.

    Returns:
        tuple: Filter coefficients (b, a).
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    """
    Applies a Butterworth low-pass filter to the data.

    Args:
        data (list): Data to be filtered.
        cutoff (float): Cutoff frequency (Hz).
        fs (float): Sampling frequency (Hz).
        order (int): Filter order.

    Returns:
        list: Filtered data.
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def moving_average_filter(data, window_size):
    """
    Applies a moving average filter to the data.

    Args:
        data (list): Data to be filtered.
        window_size (int): Size of the moving average window.

    Returns:
        list: Filtered data.
    """
    if len(data) < window_size:
        raise ValueError("Window size is larger than data length.")

    filtered_data = np.convolve(data, np.ones(window_size), 'valid') / window_size
    return filtered_data

def savitzky_golay_filter(data, window_size, polyorder):
    """
    Applies a Savitzky-Golay filter to the data.

    Args:
        data (list): Data to be filtered.
        window_size (int): Size of the filter window.
        polyorder (int): Order of the polynomial used to fit the samples.

    Returns:
        list: Filtered data.
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure window size is odd

    filtered_data = savgol_filter(data, window_size, polyorder)
    return filtered_data

# --- Data Preprocessing Pipeline ---
def preprocess_sensor_data(sensor_data):
    """
    Preprocesses the sensor data by applying filtering and feature engineering.

    Args:
        sensor_data (dict): Dictionary containing sensor readings.

    Returns:
        numpy.ndarray: Preprocessed sensor data as a NumPy array.
    """
    # Extract sensor readings
    acceleration = sensor_data["acceleration"]
    gyroscope = sensor_data["gyroscope"]
    pressure = sensor_data["pressure"]
    angle = sensor_data["angle"]

    # --- Apply Filtering ---
    # Low-pass filter acceleration and gyroscope data
    filtered_acceleration = lowpass_filter(acceleration, CUTOFF, FS)
    filtered_gyroscope = lowpass_filter(gyroscope, CUTOFF, FS)

    # Moving average filter on pressure data
    filtered_pressure = moving_average_filter(pressure, WINDOW_SIZE)

    # Savitzky-Golay filter on angle data
    filtered_angle = savitzky_golay_filter(angle, WINDOW_SIZE, POLYORDER)

    # --- Feature Engineering ---
    # Calculate jerk from filtered acceleration
    jerk = calculate_jerk(filtered_acceleration)

    # Calculate pressure difference
    pressure_diff = calculate_pressure_difference(filtered_pressure)

    # --- Combine Features ---
    # Create a feature vector by concatenating all features
    features = np.concatenate((
        filtered_acceleration,
        filtered_gyroscope,
        filtered_pressure,
        pressure_diff,
        jerk,
        [filtered_angle],  # Enclose angle in a list to concatenate correctly
    ))

    return features
