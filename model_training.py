import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from main import config # Import config from main.py

# --- Data Loading and Preprocessing ---
def load_data(data_file_path): # Changed parameter name for clarity
    """
    Loads sensor data from a CSV file.

    Args:
        data_file (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded data.
    """
    data = pd.read_csv(data_file)
    return data

def preprocess_data(data):
    """
    Preprocesses the data by separating features and labels, 
    splitting into training and testing sets, and scaling the features.

    Args:
        data (pandas.DataFrame): Data to be preprocessed.

    Returns:
        tuple: Training and testing data (X_train, X_test, y_train, y_test).
    """
    # Separate features (X) and labels (y)
    X = data.drop('target_speed', axis=1)  # Assuming 'target_speed' is the label column
    y = data['target_speed']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # Scale features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# --- Model Building and Training ---
def create_model(input_shape):
    """
    Creates a Bidirectional LSTM model with regularization and dropout.

    Args:
        input_shape (tuple): Shape of the input data.

    Returns:
        tensorflow.keras.models.Sequential: Compiled model.
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(config.LSTM_UNITS, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(config.DROPOUT_RATE))
    model.add(Bidirectional(LSTM(config.LSTM_UNITS // 2)))
    model.add(Dropout(config.DROPOUT_RATE))
    model.add(Dense(1, activation='linear', kernel_regularizer=l1_l2(l1=config.L1_REG, l2=config.L2_REG)))

    optimizer = Adam() # Consider making learning rate configurable too if Adam() default is not desired.
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def train_model(model, X_train, y_train):
    """
    Trains the model with early stopping and model checkpointing.

    Args:
        model (tensorflow.keras.models.Sequential): Model to be trained.
        X_train (numpy.ndarray): Training data features.
        y_train (numpy.ndarray): Training data labels.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=config.PATIENCE)
    model_checkpoint = ModelCheckpoint(
        config.MODEL_FILE_PATH, monitor='val_loss', save_best_only=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        callbacks=[early_stopping, model_checkpoint],
    )
    return history

# --- Hyperparameter Tuning ---
# This function might need adjustment if its parameters (lstm_units, dropout_rate, etc.)
# are intended to be sourced from a different range than MasterConfig defaults.
# For now, assuming it tests a grid around potential config values or specific ranges.
def tune_hyperparameters(X_train_orig, y_train_orig): # Renamed args to avoid conflict
    """
    Performs grid search to find the best hyperparameters for the model.
    Note: This function creates its own model based on the grid.
    Args:
        X_train_orig (numpy.ndarray): Training data features.
        y_train_orig (numpy.ndarray): Training data labels.

    Returns:
        sklearn.model_selection.GridSearchCV: Grid search object with best parameters.
    """
    # Assuming input_shape for tuning is derived similarly to the main training
    if X_train_orig.shape[1] % config.PREDICTION_WINDOW != 0:
        raise ValueError(f"X_train.shape[1] ({X_train_orig.shape[1]}) must be divisible by PREDICTION_WINDOW ({config.PREDICTION_WINDOW}) for reshaping.")
    num_features = X_train_orig.shape[1] // config.PREDICTION_WINDOW
    input_shape_for_tuning = (config.PREDICTION_WINDOW, num_features)

    # Placeholder for a KerasClassifier or similar wrapper if using GridSearchCV with Keras
    # This part needs a Keras wrapper (e.g., from scikeras.wrappers import KerasClassifier)
    # For simplicity, this example will skip the direct execution of GridSearchCV
    # and focus on the parameter grid.
    
    # Example: Define a function to create model for GridSearchCV
    def create_model_for_tuning(lstm_units, dropout_rate, l1_reg, l2_reg):
        model = Sequential()
        model.add(Bidirectional(LSTM(lstm_units, return_sequences=True), input_shape=input_shape_for_tuning))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(lstm_units // 2)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
        optimizer = Adam() # Or Adam(learning_rate=config.LEARNING_RATE) if defined
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    # This is a conceptual param_grid. Actual usage with Keras requires a wrapper.
    param_grid = {
        'lstm_units': [32, 64, 128],       # These would be passed to create_model_for_tuning
        'dropout_rate': [0.1, 0.2, 0.3],
        'l1_reg': [0.001, 0.01],
        'l2_reg': [0.001, 0.01],
        # 'batch_size': [config.BATCH_SIZE, 64], # Batch size for fit, not model creation
        # 'epochs': [config.EPOCHS, 150],       # Epochs for fit
    }
    
    print("Hyperparameter tuning setup (conceptual):")
    print("Param grid:", param_grid)
    print("Note: Actual GridSearchCV with Keras requires a wrapper like KerasClassifier.")
    # Example: model_for_grid = KerasClassifier(build_fn=create_model_for_tuning, verbose=0)
    # grid_search = GridSearchCV(estimator=model_for_grid, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=2)
    # grid_result = grid_search.fit(X_train_orig.reshape(-1, config.PREDICTION_WINDOW, num_features), y_train_orig)
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # return grid_result
    return None # Placeholder since full GridSearchCV is not implemented here

# --- Main Execution ---
if __name__ == '__main__':
    # Load and preprocess data
    data = load_data(config.DATA_FILE) # Use config
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Reshape input data for LSTM (samples, timesteps, features)
    # Task 2: Clarify LSTM Input Shape
    if X_train.shape[1] % config.PREDICTION_WINDOW != 0:
        raise ValueError(f"X_train.shape[1] ({X_train.shape[1]}) must be divisible by PREDICTION_WINDOW ({config.PREDICTION_WINDOW}) for reshaping.")
    num_features = X_train.shape[1] // config.PREDICTION_WINDOW
    
    input_shape = (config.PREDICTION_WINDOW, num_features)
    
    X_train_reshaped = X_train.reshape(X_train.shape[0], config.PREDICTION_WINDOW, num_features)
    X_test_reshaped = X_test.reshape(X_test.shape[0], config.PREDICTION_WINDOW, num_features)

    # Create and train the model
    model = create_model(input_shape)
    train_model(model, X_train_reshaped, y_train) # Use reshaped data

    # Optional: Hyperparameter tuning
    # print("Starting hyperparameter tuning (if implemented)...")
    # tuned_model_results = tune_hyperparameters(X_train, y_train) # Pass original X_train for tuning reshape
    # if tuned_model_results:
    #    print("Hyperparameter tuning complete. Best params:", tuned_model_results.best_params_)

