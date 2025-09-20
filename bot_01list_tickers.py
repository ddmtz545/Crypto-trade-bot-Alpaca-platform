###this file contains list of different groupr of tickers for my current portfolio
####and alos deliberate lists for searching potential stocks

import json
import os
import glob
import config as cfg
import time


###############################################################################
#### LOGIC FOR HANDLING NEW AND HISTORICAL SIGNALS ############################

###this new func gets the file which is created in last 20 seconds
###to avoid unwanted orders form previous saved files
###file should be created in recent_seconds time
def get_latest_signal_filepath():
    """
    Finds the path of the most recent signal JSON file if it was modified
    within the last 20 seconds.
    """
    recent_seconds =360
    script_dir = os.path.dirname(os.path.abspath(__file__))
    signal_dir = os.path.join(script_dir, cfg.FOLDER_TRADE_SIGNAL_SAVED)

    list_of_files = glob.glob(os.path.join(signal_dir, 'Strong_signal_*.json'))

    # Return None if no matching files are found
    if not list_of_files:
        return None

    # Find the most recently modified file
    latest_file = max(list_of_files, key=os.path.getmtime)

    # Check if the file's modification time is within the last 20 seconds
    file_mod_time = os.path.getmtime(latest_file)
    if time.time() - file_mod_time <= recent_seconds:
        return latest_file

    # Otherwise, return None
    return None

def load_and_filter_signals():
    """
    Loads latest signals, compares them to a historical log, returns only new
    signals, and updates the historical log for the next run.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    historical_log_path = os.path.join(script_dir, cfg.BOT_FOLDER_TRADE_SIGNAL_SAVED, 'historical_signals.json')

    # --- Step 1: Load historical signals that have been processed before ---
    ###uncomment below if you do not want to have repeatitive tickers
    ###modified for bot trading 16/08/2025

    historical_buy, historical_sell = set(), set()
    if os.path.exists(historical_log_path):
        try:
            with open(historical_log_path, 'r') as f:
                historical_data = json.load(f)
                historical_buy = set(historical_data.get('buy_tickers', []))
                historical_sell = set(historical_data.get('sell_tickers', []))
            print(f"✅ Loaded {len(historical_buy)} historical buy and {len(historical_sell)} historical sell signals.")
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Could not read historical log file. Starting fresh.")
    else:
        print("ℹ️ No historical signal file found. Will create a new one.")



    # --- Step 2: Load the latest signals from the screener ---
    latest_signal_file = get_latest_signal_filepath()
    #I enter the file name manuall, above line get the latest file correctly.
    # latest_signal_file = '.json'

    if not latest_signal_file:
        print("⚠️ No new signal file found. No tickers to process.")
        return [], []

    print(f"✅ Loading latest signals from: {os.path.basename(latest_signal_file)}")
    try:
        with open(latest_signal_file, 'r') as f:
            latest_data = json.load(f)
        latest_buy = set(latest_data.get('buy_tickers', []))
        latest_sell = set(latest_data.get('sell_tickers', []))
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"⚠️ Could not process latest signal file. No tickers to process.")
        return [], []

    # --- Step 3: Filter for ONLY NEW signals by finding the difference ---
    # new_tickers_to_buy = list(latest_buy - historical_buy)
    # new_tickers_to_sell = list(latest_sell - historical_sell)

    ###uncomment above and comment below if you do not want to have repeatitive tickers
    ###modified for bot trading 16/08/2025 below line is a cheat!
    new_tickers_to_buy = list(latest_buy )
    new_tickers_to_sell = list(latest_sell)


    print("\n--- Signal Comparison ---")
    print(f"New Buy Signals to Trade Now: {len(new_tickers_to_buy)} tickers")
    print(f"New Sell Signals to Trade Now: {len(new_tickers_to_sell)} tickers")


    # --- Step 4: Update the historical log with the latest signals for the next run ---
    # updated_historical_buy = historical_buy.union(latest_buy)
    # updated_historical_sell = historical_sell.union(latest_sell)
    ##uncomment above 2 lines and comment below if you do not want repeated values
    updated_historical_buy = list(historical_buy)+list(latest_buy)
    updated_historical_sell = list(historical_sell)+list(latest_sell)



    data_to_save = {
        'buy_tickers': sorted(list(updated_historical_buy)),
        'sell_tickers': sorted(list(updated_historical_sell))
    }
    with open(historical_log_path, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    print(f"✅ Historical log updated. Total signals tracked: {len(updated_historical_buy)} buys, {len(updated_historical_sell)} sells.")

    return new_tickers_to_buy, new_tickers_to_sell

# --- Load the pre-filtered signals to be used by the trading script ---
imaybuy, imay_short = load_and_filter_signals()


# The rest of your script can now safely use the filtered imaybuy and imay_short
print("\n--- Current State for Trading ---")
print(f"Tickers to consider buying (NEW ONLY): {len(imaybuy)} tickers\n {imaybuy}")
print(f"Tickers to consider shorting (NEW ONLY): {len(imay_short)} tickers\n {imay_short}")
print("----------------------------------------------------------------------------------------")


#### LOGIC FOR HANDLING NEW AND HISTORICAL BOT EXECUTED TRADES ############################

def get_latest_bot_executed_filepath():
    """Finds the path of the most recent signal JSON file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    signal_dir = os.path.join(script_dir, cfg.BOT_FOLDER_TRADE_SIGNAL_SAVED)

    list_of_files = glob.glob(os.path.join(signal_dir, 'bot_executed_trades_*.json'))
    if not list_of_files:
        return None

    # Find and return the most recently modified file among the signal files
    latest_signal_file = max(list_of_files, key=os.path.getmtime)

    return latest_signal_file

def load_and_filter_bot_executed_trades():
    """
    Loads latest signals, compares them to a historical log, returns only new
    signals, and updates the historical log for the next run.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    historical_log_path = os.path.join(script_dir, cfg.BOT_FOLDER_TRADE_SIGNAL_SAVED, 'bot_executed_trades_historical_log.json')

    # --- Step 1: Load historical signals that have been processed before ---
    ###uncomment below if you do not want to have repeatitive tickers
    ###modified for bot trading 16/08/2025

    historical_buy, historical_sell,historical_buy_time,historical_sell_time = set(), set(),set(),set()
    if os.path.exists(historical_log_path):
        try:
            with open(historical_log_path, 'r') as f:
                historical_data = json.load(f)

                # --- FIX STARTS HERE ---
                # Handle timestamp correctly, whether it's a list or a single string
                raw_times = historical_data.get('buy_bot_executed_time', [])
                historical_buy_time = set(raw_times if isinstance(raw_times, list) else [raw_times])

                raw_times2 = historical_data.get('sell_bot_executed_time', [])
                historical_sell_time = set(raw_times2 if isinstance(raw_times2, list) else [raw_times2])
                # --- FIX ENDS HERE ---

                historical_buy = set(historical_data.get('buy_bot_executed', []))
                historical_sell = set(historical_data.get('sell_bot_executed', []))
            print(f"✅ Loaded {len(historical_buy)} historical take profit and {len(historical_sell)} historical stop loss signals.")
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Could not read historical log file. Starting fresh.")
    else:
        print("ℹ️ No historical signal file found. Will create a new one.")


    # --- Step 2: Load the latest signals from the screener ---
    latest_signal_file = get_latest_bot_executed_filepath()
    #I enter the file name manuall, above line get the latest file correctly.
    # latest_signal_file = '.json'

    if not latest_signal_file:
        print("⚠️ No new signal file found. No tickers to process.")
        return [], [], [],[]

    print(f"✅ Loading latest signals from: {os.path.basename(latest_signal_file)}")
    try:
        with open(latest_signal_file, 'r') as f:
            latest_data = json.load(f)

        # --- FIX STARTS HERE ---
        raw_latest_times = latest_data.get('buy_bot_executed_time', [])
        latest_buy_time = set(raw_latest_times if isinstance(raw_latest_times, list) else [raw_latest_times])

        raw_latest_times2 = latest_data.get('sell_bot_executed_time', [])
        latest_sell_time = set(raw_latest_times2 if isinstance(raw_latest_times2, list) else [raw_latest_times2])
        # --- FIX ENDS HERE ---
        latest_buy = set(latest_data.get('buy_bot_executed', []))
        latest_sell = set(latest_data.get('sell_bot_executed', []))
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"⚠️ Could not process latest signal file. No tickers to process.")
        return [], [] , [], []

    # --- Step 3: Filter for ONLY NEW signals by finding the difference ---
    # new_sell_trades_take_profit = list(latest_buy - historical_buy)
    # new_sell_trades_stop_loss = list(latest_sell - historical_sell)

    ###uncomment above and comment below if you do not want to have repeatitive tickers
    ###modified for bot trading 16/08/2025 below line is a cheat!
    new_buy_trades_time = list(latest_buy_time )
    new_sell_trades_time = list(latest_sell_time )
    new_buy_trades_take_profit = list(latest_buy )
    new_sell_trades_stop_loss = list(latest_sell)


    print("\n--- Signal Comparison ---")
    print(f"New Time of buy bot executed trades Now: {len(new_buy_trades_time)} tickers")
    print(f"New Time of sell bot executed trades Now: {len(new_sell_trades_time)} tickers")
    print(f"New Buy bot executed trades Now: {len(new_buy_trades_take_profit)} tickers")
    print(f"New Sell bot executed trades  Now: {len(new_sell_trades_stop_loss)} tickers")


    # --- Step 4: Update the historical log with the latest signals for the next run ---
    # updated_historical_buy = historical_buy.union(latest_buy)
    # updated_historical_sell = historical_sell.union(latest_sell)
    # # Also apply the fix to the timestamp sets
    # updated_historical_buy_time = historical_buy_time.union(latest_buy_time)
    # updated_historical_sell_time = historical_sell_time.union(latest_sell_time)
    ##uncomment above 4 lines and comment below if you do not want repeated values
    updated_historical_buy_time = list(historical_buy_time)+list(latest_buy_time)
    updated_historical_sell_time = list(historical_sell_time)+list(latest_sell_time)
    updated_historical_buy = list(historical_buy)+list(latest_buy)
    updated_historical_sell = list(historical_sell)+list(latest_sell)



    data_to_save = {
        'buy_bot_executed_time': sorted(list(updated_historical_buy_time)),
        'buy_bot_executed': sorted(list(updated_historical_buy)),
        'sell_bot_executed_time': sorted(list(updated_historical_sell_time)),
        'sell_bot_executed': sorted(list(updated_historical_sell))
    }
    with open(historical_log_path, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    print(f"✅ Historical log updated. Total signals tracked: {len(updated_historical_buy)} buys, {len(updated_historical_sell)} sells.")

    return new_buy_trades_take_profit, new_sell_trades_stop_loss,new_buy_trades_time,new_sell_trades_time

# --- Load the pre-filtered signals to be used by the trading script ---
new_buy_trades_take_profit, new_sell_trades_stop_loss,new_buy_trades_time,new_sell_trades_time = load_and_filter_bot_executed_trades()
