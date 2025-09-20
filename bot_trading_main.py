### This script places bracket orders based on buy/sell signals from a ticker list.
### It uses an ATR-based method to calculate stop-loss and take-profit levels.
import time
import datetime
import os
import json
import config as cfg
import math
# --- MODIFIED: Import the new SDK module ---
import trading_advanced_sdk as sdk
# --- MODIFIED: Import OrderSide enum for specifying trade direction ---
from alpaca.trading.enums import OrderSide
# --- Import the pre-filtered signal lists ---
from bot_01list_tickers import imaybuy, imay_short

# --- The SDK module (trading_advanced_sdk.py) now handles client initialization.
# --- The connect_to_alpaca() call is no longer needed.

# --- Trade Sizing Parameters ---
# These parameters determine the dollar value of shares to buy or sell for each trade.
cash_to_buy = 500
cash_to_sell = 500

# --- ATR-Based Risk Management Parameters ---
# SL_MULTIPLIER determines how far the stop-loss is placed, measured in multiples of the ATR.
# For example, 1.0 means the stop-loss will be one ATR value away from the current price.
SL_MULTIPLIER = 0.75

# RISK_REWARD_RATIO defines the target profit relative to the risk.
# For example, a ratio of 1.5 means the take-profit distance from the entry
# will be 1.5 times the stop-loss distance from the entry.
RISK_REWARD_RATIO = 1.5


# --- Main Trading Logic ---
# to avoid repeated similar buy orders first get the open positions
all_position_symbols = sdk.get_all_position_symbols()
print('All current position symbols:', all_position_symbols)

# --- Process Buy Signals ---
tickers_performed_bot_buy,tickers_performed_bot_buy_time = [],[]##for saving in log

if not imaybuy:
    print("\nNo new buy signals to process for this run.")
else:
    print(f"\nProcessing {len(imaybuy)} NEW buy signals...")


    for ticker in imaybuy:

        if ticker not in all_position_symbols:
            print(f"\n--- Processing BUY signal for: {ticker} ---")

            # '''
            #----------------------market price order
            calculated_prices = sdk.calculate_sl_tp_prices_atr(
                symbol=ticker,
                side=OrderSide.BUY,
                sl_atr_multiplier=SL_MULTIPLIER,
                risk_reward_ratio=RISK_REWARD_RATIO
                )
            if calculated_prices:
                current_price = calculated_prices['current_price']
                # Ensure the price is valid before calculating quantity
                if current_price > 0:
                    # Determine the number of shares based on available cash
                    # qty_to_trade = math.ceil(cash_to_buy / current_price)
                    qty_to_trade = (cash_to_buy / current_price)

                    ###uncomment below if you buy stocks
                    # # --- CORRECTED QUANTITY LOGIC ---
                    # if '/' in ticker:
                    #     # It's crypto, use a float quantity
                    #     qty_to_trade = cash_to_buy / current_price
                    # else:
                    #     # It's a stock, use an integer quantity
                    #     qty_to_trade = math.floor(cash_to_buy / current_price)
                    # # --- END OF CORRECTION ---

                    print(f"  - Calculated trade quantity: {qty_to_trade} shares.")

                    ##simple order added by me
                    # ticker = ticker.replace('/', '') Gemini says we do not need this
                    sdk.submit_simple_order(symbol=ticker, qty=qty_to_trade, side=OrderSide.BUY)
                    # Your file saving logic here
                    # 1. Get the current date and time
                    now = datetime.datetime.now()
                    # 1. Format it into a string suitable for a filename (YYYY-MM-DD_HH-MM-SS)
                    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
                    # Get the directory of the current script (main_script.p
                    tickers_performed_bot_buy.append(ticker)
                    tickers_performed_bot_buy_time.append(timestamp)
                else:
                    print(f"  - Error: Current price for {ticker} is zero. Skipping order.")
            #---------------------------------------------

# #-------------------limit price buy order--------------
#             maker_prices = sdk.calculate_maker_prices(symbol=ticker)
#
#             if maker_prices:
#
#                 qty_to_trade = (cash_to_buy / maker_prices['buy_limit_price'])
#                 print(f"\nTo place a MAKER buy order, set limit price to: {maker_prices['buy_limit_price']}")
#                 # print(f"To place a MAKER sell order, set limit price to: {maker_prices['sell_limit_price']}")
#                 # #If price calculation is successful, calculate quantity and place the order
#                 sdk.submit_limit_order(symbol=ticker, qty=qty_to_trade, side=OrderSide.BUY, limit_price=maker_prices['buy_limit_price'])
#
#
#                 # Your file saving logic here
#                 # 1. Get the current date and time
#                 now = datetime.datetime.now()
#                 # 1. Format it into a string suitable for a filename (YYYY-MM-DD_HH-MM-SS)
#                 timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
#                 # Get the directory of the current script (main_script.p
#                 tickers_performed_bot_buy.append(ticker)
#                 tickers_performed_bot_buy_time.append(timestamp)
#             else:
#                 print(f"  - Error: Current price for {ticker} is zero. Skipping order.")
# #-------------------------


# --- Process Sell (Short) Signals ---
tickers_performed_bot_sell,tickers_performed_bot_sell_time = [],[]##for saving in log

if not imay_short:
    print("\nNo new sell signals to process for this run.")
else:
    print(f"\nProcessing {len(imay_short)} NEW sell signals...")

    for ticker in imay_short:

        if ticker in all_position_symbols:
            print(f"\n--- Processing SELL(close position) signal for: {ticker} ---")
##------------uncomment below section for just closing the position

            position_symbol = ticker.replace('/', '')
            sdk.close_position_by_symbol(symbol= position_symbol)

            # Your file saving logic here
            # 1. Get the current date and time
            now = datetime.datetime.now()
            # 1. Format it into a string suitable for a filename (YYYY-MM-DD_HH-MM-SS)
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            # Get the directory of the current script (main_script.p
            tickers_performed_bot_sell.append(ticker)
            tickers_performed_bot_sell_time.append(timestamp)

###-------------------------------------------


##---------------------------limit price sell order----------------------------
            # maker_prices = sdk.calculate_maker_prices(symbol=ticker)
            #
            # if maker_prices:
            #
            #     ticker = ticker.replace('/', '')
            #     position = sdk.get_open_position_by_symbol(ticker)
            #     print(position)
            #     qty_to_trade = position.qty
            #     # print(f"\nTo place a MAKER buy order, set limit price to: {maker_prices['buy_limit_price']}")
            #     print(f"To place a MAKER sell order, set limit price to: {maker_prices['sell_limit_price']}")
            #     # #If price calculation is successful, calculate quantity and place the order
            #     sdk.submit_limit_order(symbol=ticker, qty=qty_to_trade, side=OrderSide.SELL, limit_price=maker_prices['sell_limit_price'])
            #
            #     # Your file saving logic here
            #     # 1. Get the current date and time
            #     now = datetime.datetime.now()
            #     # 1. Format it into a string suitable for a filename (YYYY-MM-DD_HH-MM-SS)
            #     timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            #     # Get the directory of the current script (main_script.p
            #     tickers_performed_bot_sell.append(ticker)
            #     tickers_performed_bot_sell_time.append(timestamp)
            # #-------------------------
            #
            # ##removes '/' from api symbol to close position by close_position_by_symbol()
            # # ticker is 'BTC/USD'






######saving bot performed trade tickers into a jason file
# Combine the lists into a dictionary
data_to_save = {
    'buy_bot_executed_time': tickers_performed_bot_buy_time,
    'buy_bot_executed': tickers_performed_bot_buy,
    'sell_bot_executed_time' : tickers_performed_bot_sell_time,
    'sell_bot_executed' :tickers_performed_bot_sell
}
print('saving bot performed trade tickers into the jason file :\n',data_to_save)

if any(data_to_save.values()):
    print("Bot executed trades data found. Proceeding to save the file...")
    # Your file saving logic here
    # 1. Get the current date and time
    now = datetime.datetime.now()
    # 1. Format it into a string suitable for a filename (YYYY-MM-DD_HH-MM-SS)
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    # Get the directory of the current script (main_script.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f'bot_executed_trades_{timestamp}.json'
    signal_file_path_name = os.path.join(script_dir,cfg.BOT_FOLDER_TRADE_SIGNAL_SAVED,filename)
    # Write the dictionary to a JSON file
    with open(signal_file_path_name, 'w') as f:
        json.dump(data_to_save, f)

    print("✅ Bot executed trades tickers and timestamp have been saved to bot_executed_trades_*.json")
else:
    print("No Bot executed trades data to save. Skipping.")
print("\n--- Trading script execution finished. ---")
