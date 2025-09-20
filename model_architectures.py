# model_architectures.py
"""
This script contains functions to create different Keras models for time-series prediction.
This script contains functions to create, train, save, and load Keras models.
"""
import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Create sequences for LSTM
def create_sequences_multivariate(data, look_back, target_column_index):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :]) # Take all features for X
        y.append(data[i + look_back, target_column_index]) # Predict only the Close price (or another target)
    return np.array(X), np.array(y)


def create_lstm_model(look_back, num_features):
    """
    Builds and compiles a Bidirectional LSTM model.

    Args:
        look_back (int): The number of previous time steps to use as input.
        num_features (int): The number of features for each time step.

    Returns:
        A compiled Keras Sequential model.
    """
    model = Sequential()

    model.add(Bidirectional(LSTM(units=200, return_sequences=True), input_shape=(look_back, num_features)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
    model.add(Dropout(0.3))

    model.add(Bidirectional(LSTM(units=50, return_sequences=False)))
    model.add(Dropout(0.3))

    model.add(Dense(units=1))

    model.compile(optimizer='AdamW', loss='mean_squared_error')

    print("--- Bidirectional LSTM Model Created ---")
    model.summary()

    return model

def create_cnnlstm_model(look_back, num_features):
    """
    Builds and compiles a hybrid CNN-LSTM model.

    Args:
        look_back (int): The number of previous time steps to use as input.
        num_features (int): The number of features for each time step.

    Returns:
        A compiled Keras Sequential model.
    """
    model = Sequential()

    # --- CNN Feature Extraction Layers ---
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(look_back, num_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # --- LSTM Sequence Processing Layers ---
    model.add(Bidirectional(LSTM(units=200, return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
    model.add(Dropout(0.3))

    model.add(Bidirectional(LSTM(units=50, return_sequences=False)))
    model.add(Dropout(0.3))

    # --- Final Output Layer ---
    model.add(Dense(units=1))

    model.compile(optimizer='AdamW', loss='mean_squared_error')

    print("--- CNN-LSTM Hybrid Model Created ---")
    model.summary()

    return model


def train_model(model, X_train, y_train, ticker_symbol):
    """
    Trains the given Keras model.

    Args:
        model: The compiled Keras model to train.
        X_train: The training input data.
        y_train: The training target data.
        ticker_symbol (str): The ticker symbol being trained, for logging purposes.

    Returns:
        The history object from model.fit().
    """
    print(f"\nTraining the model for {ticker_symbol}....")
    if len(y_train)<2:
        print(F"\n {ticker_symbol} Training data contains 1 samples, which is not sufficient to split it into a validation and training set")
        return None
    else:
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.0000001)

        history = model.fit(
            X_train,
            y_train,
            epochs=200,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )

        return history

def save_model_to_disk(model, model_path):
    """
    Saves the trained model to a specified path.
    """
    try:
        print(f"\nSaving trained model to {model_path}...")
        model.save(model_path)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")


def load_model_from_disk(model_path):
    """
    Loads a Keras model from a specified path if it exists.
    """
    if os.path.exists(model_path):
        # print(f"Loading existing model from {model_path}...")
        try:
            model = load_model(model_path)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}. A new model will be trained.")
            return None
    else:
        print("No existing model found. A new model will be created and trained.")
        return None
