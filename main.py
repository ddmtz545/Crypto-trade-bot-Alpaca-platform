# main.py
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import datetime
import pandas as pd
# --- Your Custom Project Modules -----------
# These functions are called from within your pipeline and must be imported.
import config as cfg
from data_handler import get_ticker_reset_index_relabel # Or wherever this function is

from indicators_compute_DFlabels_adjust import (
    indicators_compute_DFlabels_adjust,
    is_market_bullish
    )
from model_architectures import (
    create_lstm_model,
    create_cnnlstm_model,
    train_model,save_model_to_disk,
    load_model_from_disk,
    create_sequences_multivariate
    )
####loading intersection slope functionalities
from intersection_slope_signal import (
    analyze_trends_and_generate_signals,
    # generate_dual_confirmation_signals
    )
import intersection_slope_signal as ints
#plot_training_lossimport plotter
from plotter import (
    plot_price_prediction_day_tf,
    plot_price_prediction_minute_tf,
    get_market_open_time,
    find_closest_timestamp,
    simple_series_plot
    )

##------------------------------------------------------------------------------


##this function runs the prediction pipline for one ticker
##it returns next_day_price_prediction and buy and sell signals analysis_df
##also plots real price and prediction graph
def run_prediction_pipeline_load_model(
    ticker_dataframe,
    ticker,
    feature_sets,
    SELECTED_feature_SET,
    look_back,
    train_size_ratio,
    time_interval,
    start_date,
    FOLDER_MODELS_SAVED,
    ENABLE_ADVANCED_ANALYSIS,
    ENABLE_MODEL_SAVE,
    FOLDER_PLOTS_SAVED,
    ENABLE_SAVE_PREDICTION_FIGURE,
    ENTRY_EXIT_SIMPLE_SERIES_PLOT,
    strong_sell_signals,
    tickers_strong_sell_signals,
    strong_buy_signals,
    tickers_strong_buy_signals,
    VOLUME_PERIOD_MVA_DAILY
):
    """
    Main pipeline to run the full process for a single ticker and feature set.
    """
    # print(f"--- Starting pipeline for {ticker} with feature set {SELECTED_feature_SET} ---")
    #function body
    ticker_symbol = ticker
    df = get_ticker_reset_index_relabel(ticker_dataframe,ticker_symbol)
    if df is None:
        return None  # This skips to the next ticker if the function failed.

#------------------------indicators computation and data frame labels rename

    # print('data from get_ticker_reset_index_relabel: \n',df)
    df = indicators_compute_DFlabels_adjust(df,ticker_symbol)
    # **THE FIX**: The loop now checks the function's output.
    if df is None:
        return None  # This skips to the next ticker if the function failed.

    ##################################################################################
    # 1. Your 'features' variable is now set automatically
    features = feature_sets[SELECTED_feature_SET]
    #################################################################################

    data = df[features].values # Convert selected features to numpy array

    # --- 2. Data Preprocessing (with multivariate input) ---
    # 2. Preprocess for LSTM
    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences for LSTM

    # Assuming 'close' is the 4th column (index 4) in the 'features' list above
    # Make sure this index is correct based on your 'features' list
    close_col_index = features.index('close')
    X, y = create_sequences_multivariate(scaled_data, look_back, close_col_index)

#------------------------------train and test data sets-------------------------

    train_size = int(len(X) * train_size_ratio)
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]

    # print(f"\nShape of X_train: {X_train.shape}")
    # print(f"Shape of y_train: {y_train.shape}")
    # print(f"Shape of X_test: {X_test.shape}")
    # print(f"Shape of y_test: {y_test.shape}")

    try:
        num_features = X_train.shape[2] # Number of features in your input sequences
        print(f"Number of features in input sequences: {num_features}")

    except Exception as e:
         print(f"Error in downloaded (index out of rsange) {ticker_symbol} data: {e}")
         print("Checking next ticker.")
         print("Skipping to the next ticker.")
         print("----------------------------------------------------------------------------------------\n\n\n\n")
         return None

    # 4. Build and Train Model
    # --- Define Model Filename --- to load saved model without training
    MODEL_FILENAME = f'{ticker_symbol}_Stock_Price_{SELECTED_feature_SET}_{time_interval}_Prediction_tsr{train_size_ratio}_{start_date}.keras'
    # Get the directory of the current script (main_script.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the model file
    # Joins: my_project_directory + models + my_keras_model.keras
    model_path_name = os.path.join(script_dir,FOLDER_MODELS_SAVED,MODEL_FILENAME)


    # # --- 3.Load the LSTM Model -------------
    model = load_model_from_disk(model_path_name)
    # Check if the model failed to load
    if model is None:
        if ENABLE_ADVANCED_ANALYSIS:
            if cfg.CNN_LSTM_CREATE_SWITCH :
                model = create_cnnlstm_model(look_back, num_features)
            else:
                model = create_lstm_model(look_back, num_features)

            # --- 4. Train the Model ---
            history = train_model(model, X_train, y_train, ticker_symbol)
            if history is None:
                return None  # This skips to the next ticker if the function failed.
            # --- Save the Trained Model ---
            if ENABLE_MODEL_SAVE:
                 save_model_to_disk(model, model_path_name)
        else:
            print(f"No model found for {ticker}. Skipping to next iteration.")
            return None # This command skips the rest of the loop

    # Optional: Plot training loss

    # --- 5. Make Predictions & Evaluate ---

    # Get predictions on the test set
    predictions_scaled = model.predict(X_test)

    # Inverse scale the predictions and actual values to get real prices
    # Need to create a dummy array for inverse_transform if only predicting one column
    # The scaler was fitted on 'data' which has 'num_features' columns.
    # To inverse transform a single column, we need to put it back into a 'num_features' shape.
    dummy_array_predictions = np.zeros((len(predictions_scaled), num_features))
    dummy_array_predictions[:, close_col_index] = predictions_scaled.flatten()
    predictions = scaler.inverse_transform(dummy_array_predictions)[:, close_col_index]

    dummy_array_y_test = np.zeros((len(y_test), num_features))
    dummy_array_y_test[:, close_col_index] = y_test.flatten()
    y_test_actual = scaler.inverse_transform(dummy_array_y_test)[:, close_col_index]
    ##note: y_test_actual = df['close'][-len(y_test):] can be used instead of y_test_actual dummy array
    # print('This is df that should be the as real price:',df['close'][-131:])
    # print('This is y_test that should be the as real price:',y_test)
    # print('This is y_test_actual that should be the as real price:',y_test_actual)

    # Calculate RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    print(f"\nRoot Mean Squared Error on Test Set: {rmse:.4f}")

    # 6. Plot Results
    # Visualize the predictions vs actual prices
    # Adjusting dates for chronological test set
    train_end_date_idx = len(df) - len(y_test_actual)
    test_dates = df.iloc[train_end_date_idx:]['Date'].reset_index(drop=True)

    ##
    plot_price_prediction_day_tf(
        test_dates,
        y_test_actual,
        predictions,
        ticker_symbol,
        SELECTED_feature_SET,
        time_interval,
        train_size_ratio,
        start_date,
        FOLDER_PLOTS_SAVED,
        ENABLE_SAVE_PREDICTION_FIGURE,
        df,
        VOLUME_PERIOD_MVA_DAILY
    )

    # 7. Analyze trends and find advanced signals
    ##slope _03intersection_slope_signal
    last_n_days_df = analyze_trends_and_generate_signals(test_dates,y_test_actual,predictions)
    # print('This is analyze_trends_and_generate_signals signal output:\n',last_n_days_df)
    ####Checking last values to find strong signal###############
    # print('This is selected slope signal output:\n',last_n_days_df.iloc[-1:,7])
    singnals_last_n_days_df = ints.generate_advanced_signals(last_n_days_df)
    print('This is generate_advanced_signals output:\n ',singnals_last_n_days_df.iloc[-100:,1:9])
    print('This is generate_advanced_signals output:\n ',singnals_last_n_days_df.iloc[-100:,9:])

    # singnals_last_n_days_df2 = ints.generate_dual_confirmation_signals(singnals_last_n_days_df)
    # print('This is generate_dual_confirmation_signals output:\n ',singnals_last_n_days_df2.iloc[-100:,1:11])
    # print('This is generate_dual_confirmation_signals:\n ',singnals_last_n_days_df2.iloc[-100:,11:])

    ints.find_best_trend_signal_advanced(singnals_last_n_days_df,ticker,strong_sell_signals,tickers_strong_sell_signals,strong_buy_signals,tickers_strong_buy_signals,df,VOLUME_PERIOD_MVA_DAILY,cfg.BEST_SIGNAL_LOOKBACK_DAYS)

    ##this plots signals
    if ENTRY_EXIT_SIMPLE_SERIES_PLOT :
        simple_series_plot(
            ticker_symbol,
            SELECTED_feature_SET,
            time_interval,
            train_size_ratio,
            start_date,
            FOLDER_PLOTS_SAVED,
            ENABLE_SAVE_PREDICTION_FIGURE,
            singnals_last_n_days_df,
            df,
            ENTRY_EXIT_SIMPLE_SERIES_PLOT,
            LOOKBACK_DAYS = cfg.BEST_SIGNAL_LOOKBACK_DAYS,
            VOLUME_PERIOD = cfg.VOLUME_PERIOD_MVA_DAILY
            )




    # 8. Predict Next day price
    # --- Predicting Future Prices (one step ahead for the next trading day) ---
    # Get the last 'look_back' days from the original scaled data
    last_sequence_scaled = scaled_data[-look_back:]
    last_sequence_scaled = last_sequence_scaled.reshape(1, look_back, num_features) # Reshape for prediction

    # Predict the next day's price (still scaled)
    next_day_scaled_prediction = model.predict(last_sequence_scaled)

    # Inverse scale the prediction
    dummy_array_next_day = np.zeros((1, num_features))
    dummy_array_next_day[:, close_col_index] = next_day_scaled_prediction.flatten()
    next_day_price_prediction = scaler.inverse_transform(dummy_array_next_day)[:, close_col_index]

    return (
        next_day_price_prediction,
        strong_sell_signals,
        tickers_strong_sell_signals,
        strong_buy_signals,
        tickers_strong_buy_signals
    )




####this is the same as above only creates model and do not load
def run_prediction_pipeline_create_model(
    ticker_dataframe,
    ticker,
    feature_sets,
    SELECTED_feature_SET,
    look_back,
    train_size_ratio,
    time_interval,
    start_date,
    FOLDER_MODELS_SAVED,
    ENABLE_ADVANCED_ANALYSIS,
    ENABLE_MODEL_SAVE,
    FOLDER_PLOTS_SAVED,
    ENABLE_SAVE_PREDICTION_FIGURE,
    ENTRY_EXIT_SIMPLE_SERIES_PLOT,
    strong_sell_signals,
    tickers_strong_sell_signals,
    strong_buy_signals,
    tickers_strong_buy_signals,
    VOLUME_PERIOD_MVA_DAILY
):
    """
    Main pipeline to run the full process for a single ticker and feature set.
    """
    # print(f"--- Starting pipeline for {ticker} with feature set {SELECTED_feature_SET} ---")
    #function body
    ticker_symbol = ticker
    df = get_ticker_reset_index_relabel(ticker_dataframe,ticker_symbol)
    if df is None:
        return None  # This skips to the next ticker if the function failed.

#------------------------indicators computation and data frame labels rename

    # print('data from get_ticker_reset_index_relabel: \n',df)
    df = indicators_compute_DFlabels_adjust(df,ticker_symbol)
    # **THE FIX**: The loop now checks the function's output.
    if df is None:
        return None  # This skips to the next ticker if the function failed.

    ##################################################################################
    # 1. Your 'features' variable is now set automatically
    features = feature_sets[SELECTED_feature_SET]
    #################################################################################

    data = df[features].values # Convert selected features to numpy array

    # --- 2. Data Preprocessing (with multivariate input) ---
    # 2. Preprocess for LSTM
    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences for LSTM

    # Assuming 'close' is the 4th column (index 4) in the 'features' list above
    # Make sure this index is correct based on your 'features' list
    close_col_index = features.index('close')
    X, y = create_sequences_multivariate(scaled_data, look_back, close_col_index)

#------------------------------train and test data sets-------------------------

    train_size = int(len(X) * train_size_ratio)
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]

    # print(f"\nShape of X_train: {X_train.shape}")
    # print(f"Shape of y_train: {y_train.shape}")
    # print(f"Shape of X_test: {X_test.shape}")
    # print(f"Shape of y_test: {y_test.shape}")

    try:
        num_features = X_train.shape[2] # Number of features in your input sequences
        print(f"Number of features in input sequences: {num_features}")

    except Exception as e:
         print(f"Error in downloaded (index out of rsange) {ticker_symbol} data: {e}")
         print("Checking next ticker.")
         print("Skipping to the next ticker.")
         print("----------------------------------------------------------------------------------------\n\n\n\n")
         return None

    # 4. Build and Train Model
    # --- Define Model Filename --- to load saved model without training
    MODEL_FILENAME = f'{ticker_symbol}_Stock_Price_{SELECTED_feature_SET}_{time_interval}_Prediction_tsr{train_size_ratio}_{start_date}.keras'


    # --- 3. Build the Model ---
    # Call the function from the other script to create the model
    # You can easily switch between models by changing which function you call.

    if cfg.CNN_LSTM_CREATE_SWITCH :
        model = create_cnnlstm_model(look_back, num_features)
    else:
        model = create_lstm_model(look_back, num_features)

    # --- 4. Train the Model ---
    history = train_model(model, X_train, y_train, ticker_symbol)
    if history is None:
        return None  # This skips to the next ticker if the function failed.

    # Get the directory of the current script (main_script.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the model file
    # Joins: my_project_directory + models + my_keras_model.keras
    model_path_name = os.path.join(script_dir,FOLDER_MODELS_SAVED,MODEL_FILENAME)


    # --- Save the Trained Model ---
    if ENABLE_MODEL_SAVE:
         save_model_to_disk(model, model_path_name)
    # Optional: Plot training loss

    # --- 5. Make Predictions & Evaluate ---

    # Get predictions on the test set
    predictions_scaled = model.predict(X_test)

    dummy_array_predictions = np.zeros((len(predictions_scaled), num_features))
    dummy_array_predictions[:, close_col_index] = predictions_scaled.flatten()
    predictions = scaler.inverse_transform(dummy_array_predictions)[:, close_col_index]

    dummy_array_y_test = np.zeros((len(y_test), num_features))
    dummy_array_y_test[:, close_col_index] = y_test.flatten()
    y_test_actual = scaler.inverse_transform(dummy_array_y_test)[:, close_col_index]


    # Calculate RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    print(f"\nRoot Mean Squared Error on Test Set: {rmse:.4f}")

    # 6. Plot Results
    # Visualize the predictions vs actual prices
    # Adjusting dates for chronological test set
    train_end_date_idx = len(df) - len(y_test_actual)
    test_dates = df.iloc[train_end_date_idx:]['Date'].reset_index(drop=True)

    ##
    plot_price_prediction_day_tf(
        test_dates,
        y_test_actual,
        predictions,
        ticker_symbol,
        SELECTED_feature_SET,
        time_interval,
        train_size_ratio,
        start_date,
        FOLDER_PLOTS_SAVED,
        ENABLE_SAVE_PREDICTION_FIGURE,
        df,
        VOLUME_PERIOD_MVA_DAILY
    )

    # 7. Analyze trends and find advanced signals
    ##slope _03intersection_slope_signal
    last_n_days_df = analyze_trends_and_generate_signals(test_dates,y_test_actual,predictions)

    singnals_last_n_days_df = ints.generate_advanced_signals(last_n_days_df)
 
    ints.find_best_trend_signal_advanced(singnals_last_n_days_df,ticker,strong_sell_signals,tickers_strong_sell_signals,strong_buy_signals,tickers_strong_buy_signals,df,VOLUME_PERIOD_MVA_DAILY,cfg.BEST_SIGNAL_LOOKBACK_DAYS)



    # 8. Predict Next day price
    # --- Predicting Future Prices (one step ahead for the next trading day) ---
    # Get the last 'look_back' days from the original scaled data
    last_sequence_scaled = scaled_data[-look_back:]
    last_sequence_scaled = last_sequence_scaled.reshape(1, look_back, num_features) # Reshape for prediction

    # Predict the next day's price (still scaled)
    next_day_scaled_prediction = model.predict(last_sequence_scaled)

    # Inverse scale the prediction
    dummy_array_next_day = np.zeros((1, num_features))
    dummy_array_next_day[:, close_col_index] = next_day_scaled_prediction.flatten()
    next_day_price_prediction = scaler.inverse_transform(dummy_array_next_day)[:, close_col_index]

    return (
        next_day_price_prediction,
        strong_sell_signals,
        tickers_strong_sell_signals,
        strong_buy_signals,
        tickers_strong_buy_signals
    )



####this is the same as below only creates model and do not load
def run_prediction_pipeline_create_minute_model(
    ticker_dataframe,
    ticker,
    feature_sets,
    SELECTED_feature_SET,
    look_back,
    train_size_ratio,
    time_interval,
    start_date,
    FOLDER_MODELS_SAVED,
    ENABLE_ADVANCED_ANALYSIS,
    ENABLE_MODEL_SAVE,
    FOLDER_PLOTS_SAVED,
    ENABLE_SAVE_PREDICTION_FIGURE,
    ENTRY_EXIT_SIMPLE_SERIES_PLOT,
    strong_sell_signals,
    tickers_strong_sell_signals,
    strong_buy_signals,
    tickers_strong_buy_signals,
    TIME_CONV_RATIO,
    VOLUME_PERIOD_MVA_MINUTE
):
    """
    Main pipeline to run the full process for a single ticker and feature set.
    """
    # print(f"--- Starting pipeline for {ticker} with feature set {SELECTED_feature_SET} ---")
    #function body
    ticker_symbol = ticker
    df = get_ticker_reset_index_relabel(ticker_dataframe,ticker_symbol)
    if df is None:
        return None  # This skips to the next ticker if the function failed.

#------------------------indicators computation and data frame labels rename

    # print('data from get_ticker_reset_index_relabel: \n',df)
    df = indicators_compute_DFlabels_adjust(df,ticker_symbol)
    # **THE FIX**: The loop now checks the function's output.
    if df is None:
        return None  # This skips to the next ticker if the function failed.

    ##################################################################################
    # 1. Your 'features' variable is now set automatically
    features = feature_sets[SELECTED_feature_SET]
    #################################################################################

    data = df[features].values # Convert selected features to numpy array

    # --- 2. Data Preprocessing (with multivariate input) ---
    # 2. Preprocess for LSTM
    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences for LSTM

    # Assuming 'close' is the 4th column (index 4) in the 'features' list above
    # Make sure this index is correct based on your 'features' list
    close_col_index = features.index('close')
    X, y = create_sequences_multivariate(scaled_data, look_back, close_col_index)

#------------------------------train and test data sets-------------------------

    train_size = int(len(X) * train_size_ratio)
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]

    # print(f"\nShape of X_train: {X_train.shape}")
    # print(f"Shape of y_train: {y_train.shape}")
    # print(f"Shape of X_test: {X_test.shape}")
    # print(f"Shape of y_test: {y_test.shape}")

    try:
        num_features = X_train.shape[2] # Number of features in your input sequences
        print(f"Number of features in input sequences: {num_features}")

    except Exception as e:
         print(f"Error in downloaded (index out of rsange) {ticker_symbol} data: {e}")
         print("Checking next ticker.")
         print("Skipping to the next ticker.")
         print("----------------------------------------------------------------------------------------\n\n\n\n")
         return None

    # 4. Build and Train Model
    # --- Define Model Filename --- to load saved model without training
    MODEL_FILENAME = f'{ticker_symbol}_Stock_Price_{SELECTED_feature_SET}_{time_interval}_Prediction_tsr{train_size_ratio}_.keras'


    # --- 3. Build the Model ---
    # Call the function from the other script to create the model
    # You can easily switch between models by changing which function you call.
    if cfg.CNN_LSTM_CREATE_SWITCH :
        model = create_cnnlstm_model(look_back, num_features)
    else:
        model = create_lstm_model(look_back, num_features)

    # --- 4. Train the Model ---
    history = train_model(model, X_train, y_train, ticker_symbol)
    if history is None:
        return None  # This skips to the next ticker if the function failed.

    # Get the directory of the current script (main_script.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the model file
    # Joins: my_project_directory + models + my_keras_model.keras
    model_path_name = os.path.join(script_dir,FOLDER_MODELS_SAVED,MODEL_FILENAME)


    # --- Save the Trained Model ---
    if ENABLE_MODEL_SAVE:
         save_model_to_disk(model, model_path_name)
    # Optional: Plot training loss

    # --- 5. Make Predictions & Evaluate ---

    # Get predictions on the test set
    predictions_scaled = model.predict(X_test)

    # Inverse scale the predictions and actual values to get real prices
    # Need to create a dummy array for inverse_transform if only predicting one column
    # The scaler was fitted on 'data' which has 'num_features' columns.
    # To inverse transform a single column, we need to put it back into a 'num_features' shape.
    dummy_array_predictions = np.zeros((len(predictions_scaled), num_features))
    dummy_array_predictions[:, close_col_index] = predictions_scaled.flatten()
    predictions = scaler.inverse_transform(dummy_array_predictions)[:, close_col_index]

    dummy_array_y_test = np.zeros((len(y_test), num_features))
    dummy_array_y_test[:, close_col_index] = y_test.flatten()
    y_test_actual = scaler.inverse_transform(dummy_array_y_test)[:, close_col_index]


    # Calculate RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    print(f"\nRoot Mean Squared Error on Test Set: {rmse:.4f}")

    # Visualize the predictions vs actual prices
    # Adjusting dates for chronological test set
    train_end_date_idx = len(df) - len(y_test_actual)
    test_dates = df.index[train_end_date_idx:]

    ##
    plot_price_prediction_minute_tf(
        test_dates,
        y_test_actual,
        predictions,
        ticker_symbol,
        SELECTED_feature_SET,
        time_interval,
        train_size_ratio,
        start_date,
        FOLDER_PLOTS_SAVED,
        ENABLE_SAVE_PREDICTION_FIGURE,
        TIME_CONV_RATIO,
        df,
        VOLUME_PERIOD_MVA_MINUTE
    )

    # 7. Analyze trends and find advanced signals
    ##slope _03intersection_slope_signal
    last_n_days_df = analyze_trends_and_generate_signals(test_dates,y_test_actual,predictions)
    ####Checking last values to find strong signal###############
    singnals_last_n_days_df = ints.generate_advanced_signals(last_n_days_df)

    ints.find_best_trend_signal_advanced(singnals_last_n_days_df,ticker,strong_sell_signals,tickers_strong_sell_signals,strong_buy_signals,tickers_strong_buy_signals,df,VOLUME_PERIOD_MVA_MINUTE,cfg.BEST_SIGNAL_LOOKBACK_DAYS)



    # 8. Predict Next day price
    # --- Predicting Future Prices (one step ahead for the next trading day) ---
    # Get the last 'look_back' days from the original scaled data
    last_sequence_scaled = scaled_data[-look_back:]
    last_sequence_scaled = last_sequence_scaled.reshape(1, look_back, num_features) # Reshape for prediction

    # Predict the next day's price (still scaled)
    next_day_scaled_prediction = model.predict(last_sequence_scaled)

    # Inverse scale the prediction
    dummy_array_next_day = np.zeros((1, num_features))
    dummy_array_next_day[:, close_col_index] = next_day_scaled_prediction.flatten()
    next_day_price_prediction = scaler.inverse_transform(dummy_array_next_day)[:, close_col_index]

    return (
        next_day_price_prediction,
        strong_sell_signals,
        tickers_strong_sell_signals,
        strong_buy_signals,
        tickers_strong_buy_signals
    )






##this function runs the prediction pipline for one ticker
##it returns next_day_price_prediction and buy and sell signals analysis_df
##also plots real price and prediction graph
def run_prediction_pipeline_load_minute_model(
    ticker_dataframe,
    ticker,
    feature_sets,
    SELECTED_feature_SET,
    look_back,
    train_size_ratio,
    time_interval,
    start_date,
    FOLDER_MODELS_SAVED,
    ENABLE_ADVANCED_ANALYSIS,
    ENABLE_MODEL_SAVE,
    FOLDER_PLOTS_SAVED,
    ENABLE_SAVE_PREDICTION_FIGURE,
    ENTRY_EXIT_SIMPLE_SERIES_PLOT,
    strong_sell_signals,
    tickers_strong_sell_signals,
    strong_buy_signals,
    tickers_strong_buy_signals,
    TIME_CONV_RATIO,
    VOLUME_PERIOD_MVA_MINUTE
):
    """
    Main pipeline to run the full process for a single ticker and feature set.
    """
    # print(f"--- Starting pipeline for {ticker} with feature set {SELECTED_feature_SET} ---")
    #function body
    ticker_symbol = ticker
    df = get_ticker_reset_index_relabel(ticker_dataframe,ticker_symbol)
    if df is None:
        return None  # This skips to the next ticker if the function failed.

#------------------------indicators computation and data frame labels rename

    # print('data from get_ticker_reset_index_relabel: \n',df)
    df = indicators_compute_DFlabels_adjust(df,ticker_symbol)
    # **THE FIX**: The loop now checks the function's output.
    if df is None:
        return None  # This skips to the next ticker if the function failed.

    ##################################################################################
    # 1. Your 'features' variable is now set automatically
    features = feature_sets[SELECTED_feature_SET]
    #################################################################################

    data = df[features].values # Convert selected features to numpy array

    # --- 2. Data Preprocessing (with multivariate input) ---
    # 2. Preprocess for LSTM
    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences for LSTM

    # Assuming 'close' is the 4th column (index 4) in the 'features' list above
    # Make sure this index is correct based on your 'features' list
    close_col_index = features.index('close')
    X, y = create_sequences_multivariate(scaled_data, look_back, close_col_index)

#------------------------------train and test data sets-------------------------

    train_size = int(len(X) * train_size_ratio)
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]

    # print(f"\nShape of X_train: {X_train.shape}")
    # print(f"Shape of y_train: {y_train.shape}")
    # print(f"Shape of X_test: {X_test.shape}")
    # print(f"Shape of y_test: {y_test.shape}")

    try:
        num_features = X_train.shape[2] # Number of features in your input sequences
        print(f"Number of features in input sequences: {num_features}")

    except Exception as e:
         print(f"Error in downloaded (index out of rsange) {ticker_symbol} data: {e}")
         print("Checking next ticker.")
         print("Skipping to the next ticker.")
         print("----------------------------------------------------------------------------------------\n\n\n\n")
         return None

    # 4. Build and Train Model
    # --- Define Model Filename --- to load saved model without training
    MODEL_FILENAME = f'{ticker_symbol}_Stock_Price_{SELECTED_feature_SET}_{time_interval}_Prediction_tsr{train_size_ratio}_.keras'
    # Get the directory of the current script (main_script.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the model file
    # Joins: my_project_directory + models + my_keras_model.keras
    model_path_name = os.path.join(script_dir,FOLDER_MODELS_SAVED,MODEL_FILENAME)


    # # --- 3.Load the LSTM Model -------------
    model = load_model_from_disk(model_path_name)
    # Check if the model failed to load
    if model is None:
        if ENABLE_ADVANCED_ANALYSIS:

            if cfg.CNN_LSTM_CREATE_SWITCH :
                model = create_cnnlstm_model(look_back, num_features)
            else:
                model = create_lstm_model(look_back, num_features)

            # --- 4. Train the Model ---
            history = train_model(model, X_train, y_train, ticker_symbol)
            if history is None:
                return None  # This skips to the next ticker if the function failed.
            # --- Save the Trained Model ---
            if ENABLE_MODEL_SAVE:
                 save_model_to_disk(model, model_path_name)
        else:
            print(f"No model found for {ticker}. Skipping to next iteration.")
            return None # This command skips the rest of the loop


    predictions_scaled = model.predict(X_test)

    dummy_array_predictions = np.zeros((len(predictions_scaled), num_features))
    dummy_array_predictions[:, close_col_index] = predictions_scaled.flatten()
    predictions = scaler.inverse_transform(dummy_array_predictions)[:, close_col_index]

    dummy_array_y_test = np.zeros((len(y_test), num_features))
    dummy_array_y_test[:, close_col_index] = y_test.flatten()
    y_test_actual = scaler.inverse_transform(dummy_array_y_test)[:, close_col_index]


    # Calculate RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    print(f"\nRoot Mean Squared Error on Test Set: {rmse:.4f}")

    # Visualize the predictions vs actual prices
    # Adjusting dates for chronological test set
    train_end_date_idx = len(df) - len(y_test_actual)
    test_dates = df.index[train_end_date_idx:]
    # print('These are X-axis test dates:\n',test_dates)
    ##
    plot_price_prediction_minute_tf(
        test_dates,
        y_test_actual,
        predictions,
        ticker_symbol,
        SELECTED_feature_SET,
        time_interval,
        train_size_ratio,
        start_date,
        FOLDER_PLOTS_SAVED,
        ENABLE_SAVE_PREDICTION_FIGURE,
        TIME_CONV_RATIO,
        df,
        VOLUME_PERIOD_MVA_MINUTE
    )

    # 7. Analyze trends and find advanced signals
    ##slope _03intersection_slope_signal
    last_n_days_df = ints.analyze_trends_and_generate_signals(test_dates,y_test_actual,predictions)
    # singnals_last_n_days_df = ints.analyze_trends_and_generate_confirmed_signals(test_dates,y_test_actual,predictions)
    # print('This is analyze_trends_and_generate_confirmed_signals output:\n',singnals_last_n_days_df.iloc[-50:,1:9])
    # print('This is analyze_trends_and_generate_confirmed_signals output:\n',singnals_last_n_days_df.iloc[-50:,9:])

    ####Checking last values to find strong signal###############
    # print('This is selected slope signal output:\n',last_n_days_df.iloc[-1:,7])
    singnals_last_n_days_df = ints.generate_advanced_signals(last_n_days_df)
    # print('This is generate_advanced_signals output:\n ',singnals_last_n_days_df.iloc[-100:,0:9])
    # print('This is generate_advanced_signals output:\n ',singnals_last_n_days_df.iloc[-100:,9:18])
    # print('This is generate_advanced_signals output:\n ',singnals_last_n_days_df.iloc[-100:,18:25])
    # print('This is generate_advanced_signals output:\n ',singnals_last_n_days_df.iloc[-100:,25:30])
    # print('This is generate_advanced_signals output:\n ',singnals_last_n_days_df.iloc[-100:,30:])

    ints.find_best_trend_signal_advanced(singnals_last_n_days_df,ticker,strong_sell_signals,tickers_strong_sell_signals,strong_buy_signals,tickers_strong_buy_signals,df,VOLUME_PERIOD_MVA_MINUTE, cfg.BEST_SIGNAL_LOOKBACK_DAYS)

    ###this plots signals
    if ENTRY_EXIT_SIMPLE_SERIES_PLOT :
        simple_series_plot(
        ticker_symbol,
        SELECTED_feature_SET,
        time_interval,
        train_size_ratio,
        start_date,
        FOLDER_PLOTS_SAVED,
        ENABLE_SAVE_PREDICTION_FIGURE,
        singnals_last_n_days_df,
        df,
        ENTRY_EXIT_SIMPLE_SERIES_PLOT,
        LOOKBACK_DAYS = cfg.BEST_SIGNAL_LOOKBACK_DAYS,
        VOLUME_PERIOD = cfg.VOLUME_PERIOD_MVA_MINUTE
        )


    # 8. Predict Next day price
    # --- Predicting Future Prices (one step ahead for the next trading day) ---
    # Get the last 'look_back' days from the original scaled data
    last_sequence_scaled = scaled_data[-look_back:]
    last_sequence_scaled = last_sequence_scaled.reshape(1, look_back, num_features) # Reshape for prediction

    # Predict the next day's price (still scaled)
    next_day_scaled_prediction = model.predict(last_sequence_scaled)

    # Inverse scale the prediction
    dummy_array_next_day = np.zeros((1, num_features))
    dummy_array_next_day[:, close_col_index] = next_day_scaled_prediction.flatten()
    next_day_price_prediction = scaler.inverse_transform(dummy_array_next_day)[:, close_col_index]

    # # ==============================================================================

    return (
        next_day_price_prediction,
        strong_sell_signals,
        tickers_strong_sell_signals,
        strong_buy_signals,
        tickers_strong_buy_signals
    )
