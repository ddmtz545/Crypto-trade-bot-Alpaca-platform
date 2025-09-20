import pandas as pd
import numpy as np
import pandas_ta as ta
import datetime
import os # Import os to check for saved model file existence
import time
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- SCRIPT START ---
# Equivalent to MATLAB's 'tic'
start_time = time.perf_counter()

# --- Your Custom Project Modules ---------------------
# These functions are called from within your pipeline and must be imported.
# from plotter import plot_price_prediction_day_tf
from data_handler import get_alpaca_timeframe,determine_asset_type,download_data_batch
import config as cfg
from config import ASSET_TYPE, ENABLE_ADVANCED_ANALYSIS
####loading thickers
import list_tickers as ltic
from list_tickers import portfolio_stock_tickers
import main

# This will ignore all FutureWarnings.
# You don't need to import FutureWarning because it's a built-in warning category.
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- ALPACA SPECIFIC IMPORTS AND CONFIGURATION ---
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.common.exceptions import APIError


# !!! IMPORTANT: REPLACE WITH YOUR ALPACA CREDENTIALS !!!
ALPACA_API_KEY_ID = cfg.ALPACA_API_KEY_ID
ALPACA_SECRET_KEY = cfg.ALPACA_SECRET_KEY
ALPACA_BASE_URL = cfg.ALPACA_BASE_URL

# Initialize Alpaca data clients
# You'll need to decide if you're fetching Stock or Crypto data
stock_data_client = StockHistoricalDataClient(ALPACA_API_KEY_ID, ALPACA_SECRET_KEY)
crypto_data_client = CryptoHistoricalDataClient(ALPACA_API_KEY_ID, ALPACA_SECRET_KEY)


# 1. assign tickers set to stock_tickers
stock_tickers= portfolio_stock_tickers

    # 1. Fetch and Prepare Data

#----------------setting dates -----------------
# start_date = datetime.datetime(2025, 7, 15)
# FIX: Set end_date to current UTC time minus a small buffer
end_date = datetime.datetime.utcnow() - datetime.timedelta(seconds = cfg.BUFFER_SECONDS) # 1000-second buffer


# 2. Initialize variables to count backwards
working_days_to_subtract = cfg.WORKING_DAYS_TO_SUBTRACT
working_days_counted = 0
current_date = end_date ###correct,it caclulates start_date from end_date by counting backwards

# 3. Loop backwards until 7 working days are found
while working_days_counted < working_days_to_subtract:
    current_date -= datetime.timedelta(days=1)
    # The weekday() method returns 0 for Monday and 6 for Sunday.
    # We only count the day if it's a weekday (Mon-Fri).
    if current_date.weekday() < 5:
        working_days_counted += 1

# 4. The loop ends on the correct date
start_date = current_date

print(f"End Date: {end_date}")
print(f"Start Date (7 working days prior): {start_date}")
#-----------------------------------------------------------------------

time_interval = cfg.TIME_INTERVAL_MINUTE # This will be a yfinance-style interval like "1d", "1h", "5m"

alpaca_timeframe = get_alpaca_timeframe(time_interval)

bars = None # Initialize bars to None
if not alpaca_timeframe:
    print(f"Error: Alpaca timeframe not found for interval '{time_interval}'.")
else:
    ticker_symbols_list = stock_tickers
    bars = download_data_batch(ASSET_TYPE, ticker_symbols_list, alpaca_timeframe, start_date, end_date, stock_data_client, crypto_data_client)

# --- Proceed only if data was successfully downloaded ---
if bars:
    # --- FIX: Get the list of tickers from the dictionary's keys ---
    available_tickers = list(bars.keys())
    print(f"\nSuccessfully downloaded data for {len(available_tickers)} of {len(stock_tickers)} requested tickers.")

##---------------------Fetching control values from config.py----------------
# Select features for the model
feature_sets = cfg.FEATURE_SETS
SELECTED_feature_SETs = cfg.SELECTED_FEATURE_SETS_MIN##this line loads settings from congig.py

# Create sequences for LSTM
look_back = cfg.LOOK_BACK # Number of previous days to consider for prediction
FOLDER_MODELS_SAVED = cfg.FOLDER_MODELS_SAVED
ENABLE_ADVANCED_ANALYSIS = cfg.ENABLE_ADVANCED_ANALYSIS
ENABLE_MODEL_SAVE = cfg.ENABLE_MODEL_SAVE

###plot function arguments
FOLDER_PLOTS_SAVED = cfg.FOLDER_PLOTS_SAVED
ENABLE_SAVE_PREDICTION_FIGURE = cfg.ENABLE_SAVE_PREDICTION_FIGURE
TIME_CONV_RATIO = cfg.TIME_CONV_RATIO
VOLUME_PERIOD_MVA_MINUTE = cfg.VOLUME_PERIOD_MVA_MINUTE
ENTRY_EXIT_SIMPLE_SERIES_PLOT=cfg.ENTRY_EXIT_SIMPLE_SERIES_PLOT
##--------------------------------------------------------------------------
    # 2. Engineer Features,# 3. Preprocess for LSTM,# 4. Build and Train Model
    ## Optional: Plot training loss,# 5. Evaluate Model,# 6. Plot Results
    # 7. Analyze trends and find advanced signals,# 8. Predict Next day price

# For time series, it's often better to split chronologically
for train_size_ratio in cfg.TRAIN_SIZE_RATIOS_MINUTE:

    for SELECTED_feature_SET in SELECTED_feature_SETs:

        strong_sell_signals , tickers_strong_sell_signals = [] , [] # List to store sell signals
        strong_buy_signals , tickers_strong_buy_signals = [] , []   # List to store buy signals


        for ticker in available_tickers:
            ticker_dataframe = bars[ticker]

            # Replace the invalid '/' with a valid '_' for BTC/USD crpto
            # First, check if '/' is in the string
            if "/" in ticker:
                # If it is, replace it with '_'
                ticker = ticker.replace("/", "_")
                print(f"Character '/' found. New name: {ticker}")

            result = main.run_prediction_pipeline_create_minute_model(
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
            )

            # If the function failed (e.g., for ASPSW), it returned None.
            # Check for this and continue to the next ticker.
            if result is None:
                print(f"--> Skipping {ticker} due to an error in the prediction pipeline.")
                continue

            # If the result is valid, now we can safely unpack it.
            (
                next_day_price_prediction,
                strong_sell_signals,
                tickers_strong_sell_signals,
                strong_buy_signals,
                tickers_strong_buy_signals
            ) = result

            # It's also good practice to check if the prediction itself is valid
            if next_day_price_prediction is None:
                print(f"--> No prediction was generated for {ticker}. Skipping.")
                continue

            print(f"\nPredicted closing price for the next trading day: ${next_day_price_prediction[0]:.2f}\n")
            print("----------------------------------------------------------------------------------------\n\n\n\n")

        ######saving strong signals tickers into a jason file
        # Combine the lists into a dictionary
        data_to_save = {
            'buy_signals': strong_buy_signals,
            'sell_signals': strong_sell_signals,
            'buy_tickers' : tickers_strong_buy_signals,
            'sell_tickers' :tickers_strong_sell_signals
        }
        print('strong signals tickers into the jason file :\n',data_to_save)
        # This checks if any of the lists in the dictionary's values are not empty
        if any(data_to_save.values()):
            print("Data found. Proceeding to save the file...")
            # Your file saving logic here
            # 1. Get the current date and time
            now = datetime.datetime.now()
            # 1. Format it into a string suitable for a filename (YYYY-MM-DD_HH-MM-SS)
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            # Get the directory of the current script (main_script.py)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filename = f'Strong_signal_{timestamp}_Stock_Price_{SELECTED_feature_SET}_{time_interval}_Prediction_tsr{train_size_ratio}_{start_date.date()}.json'
            signal_file_path_name = os.path.join(script_dir,cfg.FOLDER_TRADE_SIGNAL_SAVED,filename)
            # Write the dictionary to a JSON file
            with open(signal_file_path_name, 'w') as f:
                json.dump(data_to_save, f)

            print("✅ Signals have been saved to signals.json")
        else:
            print("No data to save. Skipping.")


# --- SCRIPT END ---
# Equivalent to MATLAB's 'toc'
end_time = time.perf_counter()

# Calculate and print the total elapsed time
elapsed_time = end_time - start_time
print("\n-------------------------------------------------")
print(f"Total script execution time: {elapsed_time:.2f} seconds")
print("-------------------------------------------------")
