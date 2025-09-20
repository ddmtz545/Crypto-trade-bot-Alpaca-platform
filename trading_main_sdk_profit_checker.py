### This is for ordering without leverage, using percentage-based SL/TP.
### Corrected to work with the legacy alpaca-trade-api.
import json
import os
import glob
import math
import time
import datetime
import trading_advanced_sdk as trad
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
# Here’s what each one represents:
# OrderSide: Defines if an order is a 'buy' or 'sell'.
# TimeInForce: Defines how long an order is active (e.g., 'day', 'gtc').
# OrderClass: Defines the order strategy (e.g., 'simple', 'bracket', 'oto').
# QueryOrderStatus: Used to filter orders by their status (e.g., 'open', 'closed').

# --- Import Your Project's Config ---
import config as cfg
# import _01list_tickers as ltic

###
#-----------------------------------logging trades----------------------------------
###############################################################################
#### LOGIC FOR HANDLING NEW AND HISTORICAL SIGNALS ############################

def get_latest_signal_filepath():
    """Finds the path of the most recent signal JSON file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    signal_dir = os.path.join(script_dir, cfg.BOT_FOLDER_TRADE_SIGNAL_SAVED)

    list_of_files = glob.glob(os.path.join(signal_dir, 'checker_closed_positions_*.json'))
    if not list_of_files:
        return None

    # Find and return the most recently modified file among the signal files
    latest_signal_file = max(list_of_files, key=os.path.getmtime)

    return latest_signal_file


###this new func gets the file which is created in last 20 seconds
###to avoid unwanted orders form previous saved files
###file should be created in recent_seconds time
'''
def get_latest_signal_filepath():
    """
    Finds the path of the most recent signal JSON file if it was modified
    within the last 20 seconds.
    """
    recent_seconds =360
    script_dir = os.path.dirname(os.path.abspath(__file__))
    signal_dir = os.path.join(script_dir, cfg.BOT_FOLDER_TRADE_SIGNAL_SAVED)

    list_of_files = glob.glob(os.path.join(signal_dir, 'profit_loss_*.json'))

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
'''

def load_and_filter_signals():
    """
    Loads latest signals, compares them to a historical log, returns only new
    signals, and updates the historical log for the next run.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    historical_log_path = os.path.join(script_dir, cfg.BOT_FOLDER_TRADE_SIGNAL_SAVED, 'checker_profit_loss_historical_log.json')

    # --- Step 1: Load historical signals that have been processed before ---
    ###uncomment below if you do not want to have repeatitive tickers
    ###modified for bot trading 16/08/2025

    historical_buy, historical_sell,historical_time = set(), set(),set()
    if os.path.exists(historical_log_path):
        try:
            with open(historical_log_path, 'r') as f:
                historical_data = json.load(f)

                # --- FIX STARTS HERE ---
                # Handle timestamp correctly, whether it's a list or a single string
                raw_times = historical_data.get('timestamp', [])
                historical_time = set(raw_times if isinstance(raw_times, list) else [raw_times])
                # --- FIX ENDS HERE ---

                historical_buy = set(historical_data.get('take_profit', []))
                historical_sell = set(historical_data.get('stop_loss', []))
            print(f"✅ Loaded {len(historical_buy)} historical take profit and {len(historical_sell)} historical stop loss signals.")
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
        return [], [], []

    print(f"✅ Loading latest signals from: {os.path.basename(latest_signal_file)}")
    try:
        with open(latest_signal_file, 'r') as f:
            latest_data = json.load(f)

            # # --- FIX STARTS HERE ---
            # raw_latest_times = latest_data.get('timestamp', [])
            # latest_time = set(raw_latest_times if isinstance(raw_latest_times, list) else [raw_latest_times])
            # # --- FIX ENDS HERE ---
            #
            # latest_buy = set(latest_data.get('take_profit', []))
            # latest_sell = set(latest_data.get('stop_loss', []))
            # CORRECTED: Use 'or []' to handle both missing keys and 'None' values gracefully.
            latest_time_data = latest_data.get('timestamp') or []
            latest_time = set(latest_time_data if isinstance(latest_time_data, list) else [latest_time_data])

            latest_buy = set(latest_data.get('take_profit') or [])
            latest_sell = set(latest_data.get('stop_loss') or [])

    except (json.JSONDecodeError, FileNotFoundError):
        print(f"⚠️ Could not process latest signal file. No tickers to process.")
        return [], [] , []

    # --- Step 3: Filter for ONLY NEW signals by finding the difference ---
    # new_sell_trades_take_profit = list(latest_buy - historical_buy)
    # new_sell_trades_stop_loss = list(latest_sell - historical_sell)

    ###uncomment above and comment below if you do not want to have repeatitive tickers
    ###modified for bot trading 16/08/2025 below line is a cheat!
    new_sell_trades_time = list(latest_time )
    new_sell_trades_take_profit = list(latest_buy )
    new_sell_trades_stop_loss = list(latest_sell)


    print("\n--- Signal Comparison ---")
    print(f"New Time of sell trades Now: {len(new_sell_trades_time)} tickers")
    print(f"New Take Trofit sell trades Now: {len(new_sell_trades_take_profit)} tickers")
    print(f"New Stop Loss Sell trades  Now: {len(new_sell_trades_stop_loss)} tickers")


    # --- Step 4: Update the historical log with the latest signals for the next run ---
    # updated_historical_buy = historical_buy.union(latest_buy)
    # updated_historical_sell = historical_sell.union(latest_sell)
    ##uncomment above 2 lines and comment below if you do not want repeated values
    updated_historical_time = list(historical_time)+list(latest_time)
    updated_historical_buy = list(historical_buy)+list(latest_buy)
    updated_historical_sell = list(historical_sell)+list(latest_sell)



    data_to_save = {
        'timestamp': sorted(list(updated_historical_time)),
        'take_profit': sorted(list(updated_historical_buy)),
        'stop_loss': sorted(list(updated_historical_sell))
    }
    with open(historical_log_path, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    print(f"✅ Historical log updated. Total signals tracked: {len(updated_historical_buy)} buys, {len(updated_historical_sell)} sells.")

    return new_sell_trades_take_profit, new_sell_trades_stop_loss,new_sell_trades_time





#----------------------------------------------------------------
#----------------------------------------------
#cancels all positions instantly
# # 2. Now, hit the panic button
# trad.panic_button_cancel_all_orders()
#------------------------------------------


# ======================================================================
# EXAMPLE: AUTOMATICALLY TAKE PROFIT ON POSITIONS
# ======================================================================
# This is the main logic to check and close profitable positions.
# You could run this inside a loop with a sleep timer to check periodically.

def run_profit_checker():
    """
    A simple loop to periodically check for profits.
    """
    while True:
        print("\n" + "="*50)
        print("RUNNING PERIODIC PROFIT CHECK...")
        print("="*50)

        # This single function does all the work:
        # 1. Gets all open positions.
        # 2. Checks the P/L % for each one.
        # 3. Closes any position where P/L % > 0.5.
##---------------main section to set take profit and stop loss values----------------
##-----------------------------------------------------------------------------------
        tp_closed_positions = trad.check_and_take_profit(profit_threshold_percentage =1.3)
        print(f"✅ Closed take profit positions: {tp_closed_positions}")

        sl_closed_positions = trad.check_and_stop_loss(-0.60)
        print(f"✅ Closed stop loss positions: {sl_closed_positions}")
##--------------------------------------------------------------------------------
##--------------------------------------------------------------------------------
        # Wait for 5 minutes before checking again.
        # print("\nCheck complete. Waiting for 1 minutes before the next run...")
        current_time = datetime.datetime.now()
        print('Current Time is:',current_time)


        ## ---------saving closed positions into a json file
        # 1. Get the current date and time
        now = datetime.datetime.now()
        # 1. Format it into a string suitable for a filename (YYYY-MM-DD_HH-MM-SS)
        timestamp2 = now.strftime("%Y-%m-%d_%H-%M-%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = f'checker_closed_positions_log_{timestamp2}.json'
        closed_positions_log_path = os.path.join(script_dir, cfg.BOT_FOLDER_TRADE_SIGNAL_SAVED, filename)

        if tp_closed_positions:
            with open(closed_positions_log_path, 'w') as f:
                json.dump({
                    'timestamp': current_time.isoformat(),
                    'take_profit': tp_closed_positions,
                    'stop_loss': 'N'
                }, f, indent=4)
                print(f"✅ TP Closed positions logged to {closed_positions_log_path}")
        if sl_closed_positions:
            with open(closed_positions_log_path, 'w') as f:
                json.dump({
                    'timestamp': current_time.isoformat(),
                    'take_profit': 'N',
                    'stop_loss': sl_closed_positions
                }, f, indent=4)
                print(f"✅ SL Closed positions logged to {closed_positions_log_path}")

            # --- Load the pre-filtered signals to be used by the trading script ---
            ##giving time for the system to save file
            time.sleep(1)
            new_sell_trades_tp, new_sell_trades_sl,new_sell_trades_time = load_and_filter_signals()


            # The rest of your script can now safely use the filtered new_sell_trades_tp and new_sell_trades_sl
            print("\n--- Current State for Trading ---")
            print(f"Tickers to consider buying (NEW ONLY): {len(new_sell_trades_tp)} tickers\n {new_sell_trades_tp}")
            print(f"Tickers to consider shorting (NEW ONLY): {len(new_sell_trades_sl)} tickers\n {new_sell_trades_sl}")
            print("----------------------------------------------------------------------------------------")



        ### Iterations checking interval
        interval_time=30
        print(f"Waiting {interval_time} seconds for the next iteration...\n" + "="*50)
        time.sleep(interval_time) # 300 seconds = 5 minutes


# --- Main execution ---
if __name__ == "__main__":

# #---------------------------------
    # trad.close_positions_before_market_close()
    # trad.schedule_close_all_positions(
    #     close_datetime_str="2025-08-18 20:29:00",
    #     timezone_str="America/New_York"
    # )
    # If you wanted to run it continuously, you would uncomment the line below:
    run_profit_checker()
