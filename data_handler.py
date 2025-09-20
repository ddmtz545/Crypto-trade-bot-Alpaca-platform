
# Required imports from the Alpaca library
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import pandas as pd
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.common.exceptions import APIError
import config as cfg
import time
import os
import datetime

###for clear folder contents
import pathlib
from typing import List, Tuple
# Mapping of common intervals to Alpaca TimeFrame objects
# Note: Alpaca has specific timeframes. Not all yfinance intervals map directly.
# For example, Alpaca supports 1Min, 5Min, 15Min, 30Min, 1Hour, 1Day, 1Week, 1Month.
# You may need to adjust your `cfg.TIME_INTERVAL_MINUTE` in _01list_tickers.py
# based on Alpaca's supported granularities.
# ALPACA_INTERVAL_MAP = {
#     "1m": TimeFrame.Minute,
#     "5m": TimeFrame.Minute, # For 5m, you'd set TimeFrame.Minute and a 'timeframe_amount=5'
#     "15m": TimeFrame.Minute, # For 15m, you'd set TimeFrame.Minute and a 'timeframe_amount=15'
#     "30m": TimeFrame.Minute, # For 30m, you'd set TimeFrame.Minute and a 'timeframe_amount=30'
#     "1h": TimeFrame.Hour,
#     "1d": TimeFrame.Day,
#     "1wk": TimeFrame.Week,
#     "1mo": TimeFrame.Month,
#     # Alpaca also has specific units like TimeFrame.Hour, TimeFrame.Day etc.
#     # For custom intervals like "2m", "90m", you might need to fetch 1m or 1h and resample locally.
# }

# Helper to get timeframe unit and amount
def get_alpaca_timeframe(interval_str):
    unit_map = {
        "m": TimeFrameUnit.Minute,
        "h": TimeFrameUnit.Hour,
        "d": TimeFrameUnit.Day,
        "wk": TimeFrameUnit.Week,
        "mo": TimeFrameUnit.Month,
    }

    if interval_str.endswith('m') and interval_str != '1m': # e.g., "5m", "15m", "30m"
        amount = int(interval_str[:-1])
        return TimeFrame(amount, TimeFrameUnit.Minute)
    elif interval_str.endswith('h') and interval_str != '1h': # e.g., "4h"
        amount = int(interval_str[:-1])
        return TimeFrame(amount, TimeFrameUnit.Hour)
    elif interval_str == "1m":
        return TimeFrame.Minute
    elif interval_str == "1h":
        return TimeFrame.Hour
    elif interval_str == "1d":
        return TimeFrame.Day
    elif interval_str == "1wk":
        return TimeFrame.Week
    elif interval_str == "1mo":
        return TimeFrame.Month
    else:
        raise ValueError(f"Unsupported interval for Alpaca: {interval_str}")

#this function downloads data for a single ticker
def determine_asset_type(ASSET_TYPE,ticker_symbol,alpaca_timeframe, start_date, end_date, stock_data_client, crypto_data_client):
    ## Determine which client to use based on ASSET_TYPE from config.py
    try:
        # Determine which client to use based on ASSET_TYPE from _01list_tickers.py
        if ASSET_TYPE == "STOCK":
            request_params = StockBarsRequest(
                symbol_or_symbols=[ticker_symbol],
                timeframe=alpaca_timeframe,
                start=start_date,
                end=end_date
            )

            bars = stock_data_client.get_stock_bars(request_params)
            # return bars

        elif ASSET_TYPE == "CRYPTO":
             request_params = CryptoBarsRequest(
                symbol_or_symbols=[ticker_symbol],
                timeframe=alpaca_timeframe,
                start=start_date,
                end=end_date
            )
             bars = crypto_data_client.get_crypto_bars(request_params)
             # return bars
        else:
            print(f"Error: Unsupported ASSET_TYPE '{ASSET_TYPE}'. Skipping {ticker_symbol}.")
            return None

        # Convert Alpaca bars to pandas DataFrame
        # Alpaca's get_bars returns a BarSet object, which can be converted to DataFrame
        try:
            df = bars.df #.df: This is a convenient property of the BarSet object. The Alpaca library developers included it to give you a one-step way to access the data as a standard pandas DataFrame.
            # print(df)
            if df.empty:
                raise ValueError("No data reading from Alpaca. Check ticker symbol, dates, or interval.")
            print(f"\n\nSuccessfully reading historical data for {ticker_symbol}.")
        except Exception as e:
            print(f"Error reading {ticker_symbol} data: {e}")
            print("Checking next ticker.")
            print("Skipping to the next ticker.")
            print("----------------------------------------------------------------------------------------\n\n\n\n")

            return None # THIS IS THE KEY CHANGE


        # Alpaca's DataFrame typically has a multi-index (symbol, timestamp)
        # Reset index and rename columns for consistency with your existing code
        df = df.reset_index()
        # df.rename(columns={'timestamp': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        df.rename(columns={'timestamp': 'Date'}, inplace=True)

        df.set_index('Date', inplace=True) # Set Date as index again for consistency

        # Drop the 'symbol' column if it exists after reset_index
        if 'symbol' in df.columns:
            df.drop(columns=['symbol'], inplace=True)

        # Ensure Date column is datetime type
        df.index = pd.to_datetime(df.index)
        return df

        if df.empty:
            raise ValueError("No data downloaded after processing. Check ticker symbol and dates.")
        print(f"\n\nSuccessfully downloaded historical data for {ticker_symbol} from Alpaca.")

    except APIError as e:
        print(f"Alpaca API Error for {ticker_symbol}: {e}")
        print("Check your ALPACA_API_KEY_ID, ALPACA_SECRET_KEY, and ALPACA_BASE_URL.")
        print("Also ensure the instrument name (ticker) is correct and available on Alpaca.")
        print("Skipping to the next ticker.")
        print("----------------------------------------------------------------------------------------\n\n\n\n")
        return None

    except Exception as e:
        print(f"General Error downloading {ticker_symbol} data from Alpaca: {e}")
        print("Skipping to the next ticker.")
        print("----------------------------------------------------------------------------------------\n\n\n\n")
        return None

###this function download data in batch
def download_data_batch(asset_type,ticker_symbols_list, timeframe, start_date, end_date, stock_client, crypto_client):
    """
    Downloads historical data for a BATCH of tickers.

    Args:
        asset_type (str): "STOCK" or "CRYPTO".
        ticker_symbols_list (list): A list of ticker symbols.
        timeframe (TimeFrame): The Alpaca TimeFrame object.
        start_date (datetime): The start date for the data.
        end_date (datetime): The end date for the data.
        stock_client: The Alpaca stock data client.
        crypto_client: The Alpaca crypto data client.

    Returns:
        pd.DataFrame: A DataFrame containing the data for all requested symbols,
                      or None on failure.
    """
    # --- 2. Batch API Data Fetching ---
    batch_size = 5000 # Alpaca's limit for a multi-symbol request
    all_data = {}
    for i in range(0, len(ticker_symbols_list), batch_size):
        batch = ticker_symbols_list[i:i + batch_size]
        try:
            print(f"Fetching data for batch {i//batch_size + 1} of {len(ticker_symbols_list)//batch_size + 1}...")

            if asset_type == "STOCK":
                request_params = StockBarsRequest(
                    symbol_or_symbols=ticker_symbols_list, # Pass the whole list here
                    timeframe=timeframe,
                    start=start_date,
                    end=end_date
                )
                bars = stock_client.get_stock_bars(request_params)
            elif asset_type == "CRYPTO":
                request_params = CryptoBarsRequest(
                    symbol_or_symbols=ticker_symbols_list, # Pass the whole list here
                    timeframe=timeframe,
                    start=start_date,
                    end=end_date
                )
                bars = crypto_client.get_crypto_bars(request_params)
            else:
                print(f"Error: Unsupported ASSET_TYPE '{asset_type}'.")
                return None

            if not bars:
                print("No data returned from API for this batch.")
                return None

            print(f"Successfully downloaded data for batch of {len(ticker_symbols_list)} tickers.")
            bars = bars.df
            # Group the combined DataFrame by symbol into a dictionary
            all_data.update({symbol: group for symbol, group in bars.groupby('symbol')})
            time.sleep(cfg.SLEEP_TIME) # Pause for 1 second between batches to be respectful to the API

        except APIError as e:
            print(f"Alpaca API Error for batch: {e}")
            return None
        except Exception as e:
            print(f"A general error occurred during batch download: {e}")
            return None

    return all_data # Return the multi-symbol DataFrame



def get_ticker_reset_index_relabel(df,ticker_symbol):

    try:
        #df = bars.df #.df: This is a convenient property of the BarSet object. The Alpaca library developers included it to give you a one-step way to access the data as a standard pandas DataFrame.

        if df.empty:
            print(f"Warning: DataFrame for {ticker_symbol} is empty. Skipping.")
            return None # THIS IS THE KEY CHANGE
        print(f"\n\nSuccessfully downloaded historical data for {ticker_symbol}.")



        # Alpaca's DataFrame typically has a multi-index (symbol, timestamp)
        # Reset index and rename columns for consistency with your existing code
        # 2. Replace Date indexing with a sequence of integer numbers.
        # This part remains mostly the same, but now df.index is already the Date.
        # .reset_index() moves the current index (Date) into a column
        # and creates a new default integer index (0, 1, 2, ...).
        ##In short, it's a three-step workaround to achieve a simple goal: changing the name of the index from timestamp to the more intuitive name Date for consistency
        df = df.reset_index()
        # df.rename(columns={'timestamp': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        df.rename(columns={'timestamp': 'Date'}, inplace=True)

        # df.set_index('Date', inplace=True) # Set Date as index again for consistency

        # Drop the 'symbol' column if it exists after reset_index
        if 'symbol' in df.columns:
            df.drop(columns=['symbol'], inplace=True)

        # Ensure Date column is datetime type
        # df.index = pd.to_datetime(df.index)
        return df


    except Exception as e:
        print(f"Error reading {ticker_symbol} data: {e}")
        print("Checking next ticker.")
        print("Skipping to the next ticker.")
        print("----------------------------------------------------------------------------------------\n\n\n\n")

        return None # THIS IS THE KEY CHANGE


#-------------------------------------------------------------------------------
####saving downloaded data form alpaca
def save_market_data(market_data, file_path, file_format='parquet'):
    """
    Consolidates a dictionary of market data DataFrames and saves it to a file.

    Args:
        market_data (dict): A dictionary where keys are ticker symbols and
                            values are their corresponding pandas DataFrames.
        file_path (str): The full path where the file will be saved
                         (e.g., 'data/my_stocks.parquet').
        file_format (str): The desired output format. Can be 'parquet' or 'csv'.
                           Defaults to 'parquet'.
    """
    # 1. Validate the input data
    if not isinstance(market_data, dict) or not market_data:
        print("Error: 'market_data' must be a non-empty dictionary.")
        return

    # --- Filter out empty or invalid DataFrames before processing ---
    valid_data = {
        symbol: df
        for symbol, df in market_data.items()
        if isinstance(df, pd.DataFrame) and not df.empty
    }

    if not valid_data:
        print("Error: No valid data found in the dictionary to save.")
        return

    print("Consolidating data into a single DataFrame...")

    # 2. Convert the dictionary of DataFrames into a single DataFrame
    try:
        # --- FIX: Concatenate the values directly. ---
        # The DataFrames in the dictionary already contain a multi-index,
        # so we don't need to create a new one with 'keys'.
        combined_df = pd.concat(valid_data.values())

        # Ensure the index names are set correctly, which is good practice.
        combined_df.index.names = ['symbol', 'timestamp']
        print(f"Data for {len(valid_data)} symbols consolidated successfully.")
    except Exception as e:
        print(f"Error: Failed to consolidate data. {e}")
        return

    # 3. Save the DataFrame to the specified format
    try:
        # Ensure the directory for the file exists
        output_dir = os.path.dirname(file_path)
        if output_dir: # Check if there is a directory part in the path
            os.makedirs(output_dir, exist_ok=True)

        print(f"Attempting to save data to '{file_path}' as {file_format}...")

        if file_format == 'parquet':
            # Parquet is efficient and preserves data types/index structure
            combined_df.to_parquet(file_path)
        elif file_format == 'csv':
            # CSV is human-readable but less efficient for large data
            combined_df.to_csv(file_path, index=True)
        else:
            print(f"Error: Unsupported file format '{file_format}'. Please choose 'parquet' or 'csv'.")
            return

        print(f"✅ Successfully saved data to {file_path}")

    except Exception as e:
        print(f"❌ Failed to save the file. Error: {e}")

def load_market_data(file_path):
    """
    Loads market data from a Parquet or CSV file into a pandas DataFrame.

    Args:
        file_path (str): The full path to the file to be loaded.

    Returns:
        pd.DataFrame: A DataFrame containing the market data, or None if loading fails.
    """
    # 1. Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return None

    # 2. Determine the file format from the extension
    _, file_extension = os.path.splitext(file_path)

    print(f"\nAttempting to load data from '{file_path}'...")

    try:
        if file_extension == '.parquet':
            # Loading from Parquet is straightforward and efficient
            df = pd.read_parquet(file_path)
            print("✅ Successfully loaded data from Parquet file.")
            return df
        elif file_extension == '.csv':
            # For CSV, we must specify how to reconstruct the index and parse dates
            df = pd.read_csv(
                file_path,
                index_col=['symbol', 'timestamp'],
                parse_dates=['timestamp']
            )
            print("✅ Successfully loaded data from CSV file.")
            return df
        else:
            print(f"Error: Unsupported file format '{file_extension}'. Please use '.parquet' or '.csv'.")
            return None
    except Exception as e:
        print(f"❌ Failed to load the file. Error: {e}")
        return None
#bars is the ouput of download_data_batch

##deletes files inside a specified folder
def clear_folder_content(
    folder_path: str,
    pattern: str = "*",
    dry_run: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Deletes files inside a specified folder that match a given pattern.

    Args:
        folder_path (str): The absolute path to the folder.
        pattern (str, optional): The glob pattern to match files (e.g., "*.tmp", "data_*.*").
                                 Defaults to "*" which matches all files.
        dry_run (bool, optional): If True, lists files that would be deleted
                                  without actually deleting them. Defaults to True.
                                  **Set is to False to really delete the files**

    Returns:
        A tuple containing two lists:
        - A list of successfully deleted file names.
        - A list of file names that failed to delete.
    """
    # 1. Create a Path object and check if the folder exists
    directory = pathlib.Path(folder_path)
    if not directory.is_dir():
        print(f"Error: Folder not found at '{folder_path}'")
        return [], []

    print(f"Searching for files matching '{pattern}' in '{folder_path}'...")
    if dry_run:
        print("--- DRY RUN MODE: No files will be deleted. ---")

    deleted_files = []
    failed_files = []

    # 2. Find and loop through all files matching the pattern
    for file in directory.glob(pattern):
        # 3. Ensure it's a file (and not a directory) before proceeding
        if file.is_file():
            try:
                if not dry_run:
                    file.unlink()  # This is the actual deletion command

                print(f"{'Would delete' if dry_run else 'Deleted'}: {file.name}")
                deleted_files.append(file.name)
            except Exception as e:
                print(f"Error deleting {file.name}: {e}")
                failed_files.append(file.name)

    print("\n--- Summary ---")
    print(f"Files processed: {len(deleted_files) + len(failed_files)}")
    print(f"Files deleted (or would be): {len(deleted_files)}")
    print(f"Failures: {len(failed_files)}")

    return deleted_files, failed_files

# --- HOW TO USE THE FUNCTION ---
if __name__ == "__main__":
    # ⚠️ IMPORTANT: Change this to the path of the folder you want to clear.
    # It is highly recommended to use an absolute path.
    target_folder = '/path/to/your/folder'

    # --- Example 1: Safe dry run to see what would be deleted ---
    # This will only list all files and will NOT delete anything.
    print("--- Starting Dry Run (all files) ---")
    clear_folder_content(target_folder, pattern="*", dry_run=True)

    # --- Example 2: Dry run for specific files (e.g., all .txt files) ---
    print("\n--- Starting Dry Run (.txt files only) ---")
    clear_folder_content(target_folder, pattern="*.txt", dry_run=True)

    # --- Example 3: ACTUAL DELETION ---
    # To actually delete files, you must set dry_run=False.
    # UNCOMMENT THE LINES BELOW ONLY WHEN YOU ARE SURE.
    # print("\n--- Starting ACTUAL DELETION (.tmp files) ---")
    # deleted, failed = clear_folder_content(target_folder, pattern="*.tmp", dry_run=False)
    # if failed:
    #     print(f"Could not delete the following files: {failed}")



