import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

# --- Configuration Parameters ---
DATA_FILE = 'sensor_data.csv'  # Replace with your data file
TEST_SIZE = 0.2               # Proportion of data for testing
VALIDATION_SPLIT = 0.2       # Proportion of training data for validation
EPOCHS = 100                  # Number of training epochs
BATCH_SIZE = 32               # Batch size for training
LSTM_UNITS = 64              # Number of units in LSTM layer
DROPOUT_RATE = 0.2           # Dropout rate for regularization
L1_REG = 0.001               # L1 regularization strength
L2_REG = 0.001               # L2 regularization strength
PATIENCE = 10                 # Patience for early stopping

# --- Data Loading and Preprocessing ---
def load_data(data_file):
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
        X, y, test_size=TEST_SIZE, random_state=42
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
    model.add(Bidirectional(LSTM(LSTM_UNITS, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Bidirectional(LSTM(LSTM_UNITS // 2)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(1, activation='linear', kernel_regularizer=l1_l2(l1=L1_REG, l2=L2_REG)))

    optimizer = Adam()
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
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE)
    model_checkpoint = ModelCheckpoint(
        'best_model.h5', monitor='val_loss', save_best_only=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping, model_checkpoint],
    )
    return history

# --- Hyperparameter Tuning ---
def tune_hyperparameters(model, X_train, y_train):
    """
    Performs grid search to find the best hyperparameters for the model.

    Args:
        model (tensorflow.keras.models.Sequential): Model to be tuned.
        X_train (numpy.ndarray): Training data features.
        y_train (numpy.ndarray): Training data labels.

    Returns:
        sklearn.model_selection.GridSearchCV: Grid search object with best parameters.
    """
    param_grid = {
        'lstm_units': [32, 64, 128],
        'dropout_rate': [0.1, 0.2, 0.3],
        'l1_reg': [0.001, 0.01, 0.1],
        'l2_reg': [0.001, 0.01, 0.1],
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=2,
    )

    grid_result = grid_search.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return grid_search

# --- Main Execution ---
if __name__ == '__main__':
    # Load and preprocess data
    data = load_data(DATA_FILE)
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Reshape input data for LSTM (samples, timesteps, features)
    # Assuming each sample has 1 timestep and multiple features
    input_shape = (X_train.shape[1], 1) 
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Create and train the model
    model = create_model(input_shape)
    train_model(model, X_train, y_train)

    # Optional: Hyperparameter tuning
    # tuned_model = tune_hyperparameters(model, X_train, y_train)
