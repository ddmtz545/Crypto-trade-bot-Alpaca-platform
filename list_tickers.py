###this file contains list of different groupr of tickers for my current portfolio
####and alos deliberate lists for searching potential stocks

import json
import os
import config as cfg


# --- NEW: Import ticker lists from the 'tickers' package ---
# This is much cleaner and more organized.


###############################################################################
####load tickers from json files###########################################

script_dir = os.path.dirname(os.path.abspath(__file__))
##note: filename should not have '.' in its name to open with json
filename = []
# filename = 'nany_lbk1_noVolume_27082025.json'

imaybuy = []
imay_short = []

# 2. Proceed only if the filename is not empty
if filename:
    signal_file_path = os.path.join(script_dir, cfg.FOLDER_TRADE_SIGNAL_SAVED, filename)

    # 3. Check if the file actually exists before trying to open it
    if os.path.exists(signal_file_path):
        try:
            with open(signal_file_path, 'r') as f:
                loaded_data = json.load(f)

            # 4. Use .get() for safer dictionary access
            imaybuy = loaded_data.get('buy_tickers', [])
            imay_short = loaded_data.get('sell_tickers', [])

            print(f"✅ Successfully loaded {len(imaybuy) + len(imay_short)} signals from {filename}.")

        except json.JSONDecodeError:
            # Catches errors from malformed JSON (e.g., empty file)
            print(f"⚠️ Warning: Could not decode JSON from {filename}. File might be empty or corrupt.")
        except Exception as e:
            # Catches other unexpected errors during file processing
            print(f"An unexpected error occurred while processing {filename}: {e}")
    else:
        # Handle the case where the file doesn't exist
        print(f"ℹ️ Info: File not found at '{signal_file_path}'. Continuing without loading.")
else:
    # Handle the case where the filename string is empty
    print("ℹ️ Info: Filename is empty. Continuing without loading.")

# The rest of your script can now safely use imaybuy and imay_short
print("\n--- Current State ---")
print(f"Tickers to consider buying (json file){len(imaybuy)} tickers:\n {imaybuy}")
print(f"Tickers to consider shorting (json file){len(imay_short)} tickers:\n {imay_short}")
print("----------------------------------------------------------------------------------------")

#########################################################################
###############################################################################

###################assign tickers set to stock_tickers#########################
##############parameters for 01portfolio_tickers_price_prediction.py (01ptpp.py)

# portfolio_stock_tickers = ["XRP/USD"]

portfolio_stock_tickers = ["BTC/USD","ETH/USD","XRP/USD","LINK/USD","BCH/USD","LTC/USD","UNI/USD","AAVE/USD"]##VXX for representing market fear,A rising VXX price generally indicates increasing market uncertainty.


