# Alpaca Trading API - Python Script with Advanced Order Types

import alpaca_trade_api as tradeapi
import os
import time
import config as cfg
import list_tickers as ltic

import pandas_ta as ta
import pandas as pd
import datetime
from alpaca_trade_api.rest import TimeFrame

# --- Configuration ---
# It's best practice to use environment variables for your API keys.
# For Paper Trading: Use your paper trading keys and the paper trading URL.
# For Live Trading: Use your live trading keys and the live trading URL.
API_KEY = os.environ.get('APCA_API_KEY_ID', cfg.ALPACA_API_KEY_ID)
API_SECRET = os.environ.get('APCA_API_SECRET_KEY', cfg.ALPACA_SECRET_KEY)
BASE_URL = os.environ.get('APCA_API_BASE_URL', cfg.ALPACA_BASE_URL) # Use 'https://api.alpaca.markets' for live trading

# --- Alpaca API Connection ---
def connect_to_alpaca():
    """
    Establishes and tests the connection to the Alpaca API.

    Returns:
        tradeapi.REST: An authenticated Alpaca API object, or None if connection fails.
    """
    try:
        api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL, api_version='v2')
        account = api.get_account()
        print(f"Successfully connected to Alpaca. Account Status: {account.status}")
        print(f"Buying Power (Leverage is included): {account.buying_power}")
        return api
    except Exception as e:
        print(f"Error connecting to Alpaca: {e}")
        return None

# --- NEW FUNCTION: Calculate Stop Loss / Take Profit from Market Price ---
def calculate_sl_tp_from_market(api, symbol, side, stop_loss_pct, take_profit_pct):
    """
    Gets the current market price and calculates stop loss and take profit levels.

    Args:
        api (tradeapi.REST): The authenticated Alpaca API object.
        symbol (str): The stock symbol.
        side (str): The side of the trade, 'buy' for long, 'sell' for short.
        stop_loss_pct (float): The stop loss percentage (e.g., 2.0 for 2%).
        take_profit_pct (float): The take profit percentage (e.g., 4.0 for 4%).

    Returns:
        dict: A dictionary containing 'current_price', 'stop_loss_price',
              and 'take_profit_price', or None on failure.
    """
    if not api:
        print("API connection not available.")
        return None
    try:
        # Get the latest trade price for the symbol
        latest_trade = api.get_latest_trade(symbol)
        current_price = latest_trade.price
        print(f"Current market price for {symbol} is ${current_price:.4f}")

        if side == 'buy':
            stop_loss_price = current_price * (1 - stop_loss_pct / 100)
            take_profit_price = current_price * (1 + take_profit_pct / 100)
        elif side == 'sell':
            stop_loss_price = current_price * (1 + stop_loss_pct / 100)
            take_profit_price = current_price * (1 - take_profit_pct / 100)
        else:
            print(f"Error: Invalid order side '{side}'. Must be 'buy' or 'sell'.")
            return None

        # Round to 2 decimal places for typical stock prices
        prices = {
            "current_price": round(current_price, 2),
            "stop_loss_price": round(stop_loss_price, 2),
            "take_profit_price": round(take_profit_price, 2)
        }
        return prices

    except Exception as e:
        print(f"An error occurred fetching price or calculating SL/TP: {e}")
        return None

# --- NEW ATR-BASED SL/TP CALCULATION ---
def calculate_sl_tp_with_atr(api, symbol, side, sl_atr_multiplier=2.0, tp_atr_multiplier=4.0, atr_period=14):
    """
    Gets the current price and calculates stop-loss/take-profit based on ATR.

    Args:
        api (tradeapi.REST): Authenticated Alpaca API object.
        symbol (str): The stock symbol.
        side (str): 'buy' or 'sell'.
        sl_atr_multiplier (float): Multiplier for ATR to set stop loss.
        tp_atr_multiplier (float): Multiplier for ATR to set take profit.
        atr_period (int): The lookback period for ATR (usually 14).

    Returns:
        dict: A dictionary with prices, or None on failure.
    """
    if not api:
        print("API connection not available.")
        return None
    try:
        # 1. Get current market price
        latest_trade = api.get_latest_trade(symbol)
        current_price = latest_trade.price
        print(f"Current market price for {symbol} is ${current_price:.4f}")

        # 2. Get recent historical data to calculate ATR
        end_date = datetime.datetime.now(datetime.timezone.utc)
        start_date = end_date - datetime.timedelta(days=atr_period * 2) # Get enough data
        bars = api.get_bars(symbol, TimeFrame.Day, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), adjustment='raw').df

        if bars.empty or len(bars) < atr_period:
            print(f"Not enough historical data for {symbol} to calculate ATR.")
            return None

        # 3. Calculate ATR
        bars['atr'] = ta.atr(high=bars['high'], low=bars['low'], close=bars['close'], length=atr_period)
        latest_atr = bars['atr'].iloc[-1]
        print(f"Latest 14-Day ATR for {symbol} is: {latest_atr:.4f}")

        # 4. Calculate SL/TP prices based on ATR and side
        if side == 'buy':
            stop_loss_price = current_price - (latest_atr * sl_atr_multiplier)
            take_profit_price = current_price + (latest_atr * tp_atr_multiplier)
        elif side == 'sell':
            stop_loss_price = current_price + (latest_atr * sl_atr_multiplier)
            take_profit_price = current_price - (latest_atr * tp_atr_multiplier)
        else:
            raise ValueError(f"Invalid order side '{side}'. Must be 'buy' or 'sell'.")

        prices = {
            "current_price": round(current_price, 2),
            "stop_loss_price": round(stop_loss_price, 2),
            "take_profit_price": round(take_profit_price, 2),
            "atr": round(latest_atr, 2)
        }
        return prices

    except Exception as e:
        print(f"An error occurred in calculate_sl_tp_with_atr: {e}")
        return None



# --- Simple Market Order Function ---
def place_market_order(api, symbol, qty, side):
    """
    Places a simple market order (buy or sell).

    Args:
        api (tradeapi.REST): The authenticated Alpaca API object.
        symbol (str): The stock symbol (e.g., 'AAPL').
        qty (int): The number of shares.
        side (str): 'buy' or 'sell'.
    """
    if not api:
        print("API connection not available.")
        return None
    try:
        print(f"Placing a market {side} order for {qty} shares of {symbol}...")
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='day'
        )
        print(f"Market {side} order for {symbol} submitted successfully.")
        print(f"Order ID: {order.id}, Status: {order.status}")
        return order
    except Exception as e:
        print(f"An error occurred placing the market order: {e}")
        return None

# --- Bracket Order Function (Take Profit and Stop Loss for Long/Short) ---
def place_bracket_order(api, symbol, qty, side, entry_price, take_profit_price, stop_loss_price):
    """
    Places a bracket order for either a long ('buy') or short ('sell') position.
    This consists of a primary entry order (limit) and two exit orders (take-profit
    and stop-loss) that are active once the primary order fills.

    The take-profit and stop-loss orders are "One-Cancels-Other" (OCO).

    Args:
        api (tradeapi.REST): The authenticated Alpaca API object.
        symbol (str): The stock symbol.
        qty (int): The number of shares to trade.
        side (str): The side of the trade, 'buy' for long, 'sell' for short.
        entry_price (float): The limit price for the initial order.
        take_profit_price (float): The price at which to exit for a profit.
        stop_loss_price (float): The price at which to exit for a loss.
    """
    if not api:
        print("API connection not available.")
        return None
    # --- NEW: Minimum Price Gap Check ---
    # Ensure there is at least a $0.01 difference between each price point
    min_gap = 0.01

    # --- NEW: Intelligent Price Gap Adjustment ---
    # Adjust stop loss if it's too close to the entry price
    if abs(entry_price - stop_loss_price) < min_gap:
        print(f"Warning: Original stop loss for {symbol} is too close. Adjusting to minimum ${min_gap} gap.")
        if side == 'buy':
            stop_loss_price = entry_price - min_gap
        else: # side == 'sell'
            stop_loss_price = entry_price + min_gap
        print(f"  New Adjusted Stop Loss: {stop_loss_price:.4f}")

    # Adjust take profit if it's too close to the entry price
    if abs(entry_price - take_profit_price) < min_gap:
        print(f"Warning: Original take profit for {symbol} is too close. Adjusting to minimum ${min_gap} gap.")
        if side == 'buy':
            take_profit_price = entry_price + min_gap
        else: # side == 'sell'
            take_profit_price = entry_price - min_gap
        print(f"  New Adjusted Take Profit: {take_profit_price:.4f}")


    # --- Price Validation based on trade side ---
    if side == 'buy':
        if not (take_profit_price > entry_price > stop_loss_price):
            print("Error: For a 'buy' bracket order, price relationship must be Take Profit > Entry > Stop Loss.")
            return None
    elif side == 'sell':
        if not (stop_loss_price > entry_price > take_profit_price):
            print("Error: For a 'sell' bracket order, price relationship must be Stop Loss > Entry > Take Profit.")
            return None
    else:
        print(f"Error: Invalid order side '{side}'. Must be 'buy' or 'sell'.")
        return None

    try:
        print(f"Placing a {side} bracket order for {qty} shares of {symbol}...")
        print(f"Entry: {entry_price}, Take Profit: {take_profit_price}, Stop Loss: {stop_loss_price}")

        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='limit',
            time_in_force='gtc',
            limit_price=entry_price,
            order_class='bracket',
            take_profit={
                'limit_price': take_profit_price
            },
            stop_loss={
                'stop_price': stop_loss_price
            }
        )
        print(f"Bracket order for {symbol} submitted successfully.")
        print(f"Order ID: {order.id}, Status: {order.status}")
        return order
    except Exception as e:
        print(f"An error occurred placing the bracket order: {e}")
        return None




# --- Universal Bracket Order Function ---
def universal_place_bracket_order(
    api: tradeapi.REST,
    symbol: str,
    qty: int,
    side: str,
    take_profit_price: float,
    stop_loss_price: float,
    order_type: str = 'market',
    time_in_force: str = 'gtc',
    limit_price: float = None,
    stop_price: float = None,
    trail_price: float = None,
    trail_percent: float = None
):
    """
    Places a versatile bracket order with a market, limit, stop, stop_limit, or trailing_stop entry.

    Args:
        api (tradeapi.REST): The authenticated Alpaca API object.
        symbol (str): The stock symbol.
        qty (int): The number of shares to trade.
        side (str): The side of the trade, 'buy' for long, 'sell' for short.
        take_profit_price (float): The price at which to exit for a profit.
        stop_loss_price (float): The price at which to exit for a loss.
        order_type (str, optional): 'market', 'limit', 'stop', 'stop_limit', or 'trailing_stop'. Defaults to 'market'.
        time_in_force (str, optional): The time in force for the order. Defaults to 'gtc'.
        limit_price (float, optional): Required for 'limit' and 'stop_limit' orders.
        stop_price (float, optional): Required for 'stop' and 'stop_limit' orders.
        trail_price (float, optional): Required for 'trailing_stop' orders if trail_percent is not set.
        trail_percent (float, optional): Required for 'trailing_stop' orders if trail_price is not set.
    """
    if not api:
        print("API connection not available.")
        return None

    order_data = {}  # To store final order parameters

    try:
        # Base payload shared by all bracket orders
        base_order_payload = {
            'symbol': symbol,
            'qty': qty,
            'side': side,
            'time_in_force': time_in_force,
            'order_class': 'bracket',
            'take_profit': {'limit_price': take_profit_price},
            'stop_loss': {'stop_price': stop_loss_price}
        }

        # --- Construct payload based on order_type ---

        if order_type == 'market':
            if side == 'buy' and not (take_profit_price > stop_loss_price):
                raise ValueError("For a 'buy' market order, TP must be > SL.")
            if side == 'sell' and not (stop_loss_price > take_profit_price):
                raise ValueError("For a 'sell' market order, SL must be > TP.")
            order_data = {**base_order_payload, 'type': 'market'}

        elif order_type == 'limit':
            if limit_price is None:
                raise ValueError("'limit_price' is required for a 'limit' order.")
            if side == 'buy' and not (take_profit_price > limit_price > stop_loss_price):
                raise ValueError("For a 'buy' limit order, price must be TP > Limit > SL.")
            if side == 'sell' and not (stop_loss_price > limit_price > take_profit_price):
                raise ValueError("For a 'sell' limit order, price must be SL > Limit > TP.")
            order_data = {**base_order_payload, 'type': 'limit', 'limit_price': limit_price}

        elif order_type == 'stop':
            if stop_price is None:
                raise ValueError("'stop_price' is required for a 'stop' order.")
            if side == 'buy' and not (take_profit_price > stop_price > stop_loss_price):
                 raise ValueError("For a 'buy' stop order, price must be TP > Stop Price > SL.")
            if side == 'sell' and not (stop_loss_price > stop_price > take_profit_price):
                 raise ValueError("For a 'sell' stop order, price must be SL > Stop Price > TP.")
            order_data = {**base_order_payload, 'type': 'stop', 'stop_price': stop_price}

        elif order_type == 'stop_limit':
            if not all([limit_price, stop_price]):
                raise ValueError("'limit_price' and 'stop_price' are required for a 'stop_limit' order.")
            if side == 'buy' and not (take_profit_price > limit_price >= stop_price > stop_loss_price):
                 raise ValueError("For a 'buy' stop_limit, price must be TP > Limit >= Stop > SL.")
            if side == 'sell' and not (stop_loss_price > limit_price <= stop_price > take_profit_price):
                 raise ValueError("For a 'sell' stop_limit, price must be SL > Limit <= Stop > TP.")
            order_data = {**base_order_payload, 'type': 'stop_limit', 'limit_price': limit_price, 'stop_price': stop_price}

        elif order_type == 'trailing_stop':
            if trail_price is None and trail_percent is None:
                raise ValueError("Either 'trail_price' or 'trail_percent' is required for a 'trailing_stop' order.")
            if trail_price is not None and trail_percent is not None:
                raise ValueError("Provide either 'trail_price' or 'trail_percent', not both.")

            trail_params = {'trail_price': trail_price} if trail_price else {'trail_percent': trail_percent}
            order_data = {**base_order_payload, 'type': 'trailing_stop', **trail_params}

        else:
            valid_types = "'market', 'limit', 'stop', 'stop_limit', 'trailing_stop'"
            raise ValueError(f"Invalid order_type '{order_type}'. Must be one of {valid_types}.")

        # --- Submit the Order ---
        print(f"Placing a {side} {order_type} bracket order for {qty} shares of {symbol}...")
        order = api.submit_order(**order_data)
        print(f"✅ Bracket order for {symbol} submitted successfully.")
        print(f"   Order ID: {order.id}, Status: {order.status}")
        return order

    except (ValueError, Exception) as e:
        print(f"❌ An error occurred placing the bracket order: {e}")
        return None

# --- CORRECTED Replace Order Functions ---
def replace_order_leg(api, parent_order_id, new_price, leg_type):
    """
    A helper function to replace a specific leg (stop_loss or take_profit) of a bracket order.
    """
    try:
        # 1. Get the parent order to find its legs
        parent_order = api.get_order(parent_order_id)
        if not hasattr(parent_order, 'legs') or parent_order.legs is None:
            print(f"Error: Order {parent_order_id} is not a bracket order or has no legs.")
            return None

        # 2. Find the ID of the specific leg to replace based on order type
        leg_to_replace = None
        if leg_type == 'stop_loss':
            # A stop-loss order can be of type 'stop' or 'stop_limit'
            leg_to_replace = next((leg for leg in parent_order.legs if leg.type in ['stop', 'stop_limit']), None)
        elif leg_type == 'take_profit':
            # A take-profit order is always of type 'limit'
            leg_to_replace = next((leg for leg in parent_order.legs if leg.type == 'limit'), None)

        if not leg_to_replace:
            print(f"Error: Could not find the {leg_type} leg for order {parent_order_id}.")
            return None

        leg_id = leg_to_replace.id
        print(f"Found {leg_type} leg with ID: {leg_id}")

        # 3. Prepare the parameters for replacement
        replace_params = {}
        if leg_type == 'stop_loss':
            replace_params['stop_price'] = new_price
        elif leg_type == 'take_profit':
            replace_params['limit_price'] = new_price

        # 4. Call the replace_order method on the LEG's ID
        print(f"Attempting to replace {leg_type} leg {leg_id} with new price: {new_price}...")
        new_leg_order = api.replace_order(order_id=leg_id, **replace_params)

        print(f"✅ {leg_type.replace('_', ' ').title()} leg replaced successfully.")
        print(f"   New Order ID for leg: {new_leg_order.id}")
        return new_leg_order

    except Exception as e:
        print(f"❌ An error occurred replacing the order leg: {e}")
        return None

#------------------------Orders modification-----------------------------------#
################################################################################
# --- NEW FUNCTION: Add Protection to an Existing Position ---
# --- CORRECTED FUNCTION: Add Protection to an Existing Position ---
def add_protection_to_position(api, symbol, qty, take_profit_price, stop_loss_price):
    """
    Adds a protective OCO (One-Cancels-Other) bracket to an existing open position.
    This is used when a position was opened without an initial stop-loss or take-profit.

    Args:
        api (tradeapi.REST): The authenticated Alpaca API object.
        symbol (str): The symbol of the existing position.
        qty (int): The number of shares in the position.
        take_profit_price (float): The price to sell for a profit.
        stop_loss_price (float): The price to sell for a loss.
    """
    if not api:
        print("API connection not available.")
        return None
    try:
        print(f"Adding OCO protection for {qty} shares of {symbol}...")
        print(f"  Take Profit: {take_profit_price}, Stop Loss: {stop_loss_price}")

        # To close a long position, the OCO order must be a 'sell' order.
        # The primary order must be of type 'limit', which serves as the take-profit.
        # The stop-loss is then attached as the second leg of the OCO.
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell', # This order is to CLOSE the existing long position
            type='limit', # THIS IS THE CRITICAL FIX: The primary order is the take-profit limit.
            limit_price=take_profit_price, # The price for the take-profit limit order.
            time_in_force='gtc',
            order_class='oco',
            take_profit={ # THIS IS THE CRITICAL FIX: The take_profit leg must be a dictionary.
                'limit_price': take_profit_price
            },
            stop_loss={ # The stop-loss is the second leg of the OCO order.
                'stop_price': stop_loss_price
            }
        )
        print(f"✅ OCO protection order for {symbol} submitted successfully.")
        print(f"   Order ID: {order.id}, Status: {order.status}")
        return order
    except Exception as e:
        print(f"❌ An error occurred adding OCO protection: {e}")
        return None

def replace_order_stop_loss(api, order_id, new_stop_loss_price):
    """
    Replaces the stop-loss leg of an existing bracket order.
    """
    return replace_order_leg(api, order_id, new_stop_loss_price, 'stop_loss')

def replace_order_take_profit(api, order_id, new_take_profit_price):
    """
    Replaces the take-profit leg of an existing bracket order.
    """
    return replace_order_leg(api, order_id, new_take_profit_price, 'take_profit')

# Other functions (cancel_order etc.) would be here...


# --- Cancel Order Function ---
def cancel_order(api, order_id):
    """
    Cancels a specific open order.

    Args:
        api (tradeapi.REST): The authenticated Alpaca API object.
        order_id (str): The unique ID of the order to be canceled.
    """
    if not api:
        print("API connection not available.")
        return
    try:
        print(f"Attempting to cancel order ID: {order_id}...")
        api.cancel_order(order_id)
        print(f"Request to cancel order {order_id} sent successfully.")
    except Exception as e:
        print(f"An error occurred while trying to cancel the order: {e}")
        print("The order may have already been filled, canceled, or expired.")

'''
# --- Main Execution ---
if __name__ == '__main__':
    alpaca_api = connect_to_alpaca()

    if alpaca_api:
        # --- Example Usage ---
        # Make sure to replace with valid symbols and realistic prices.
        # To run an example, uncomment the corresponding function call(s).

        # Example 1: Calculate SL/TP and place a long bracket order for AAPL
        # Note: We use the calculated current price as the limit entry price.
        print("\n--- Example: Calculate SL/TP and Place Long Bracket Order ---")
        symbol_to_buy = 'AAPL'
        trade_side = 'buy'
        stop_loss_percent = 1.5  # 1.5% stop loss
        take_profit_percent = 3.0 # 3.0% take profit

        # Calculate the prices based on the current market price
        calculated_prices = calculate_sl_tp_from_market(
            alpaca_api, symbol_to_buy, trade_side, stop_loss_percent, take_profit_percent
        )

        # If calculation is successful, place the order
        if calculated_prices:
            print(f"Calculated Prices: {calculated_prices}")
            place_bracket_order(
                alpaca_api,
                symbol=symbol_to_buy,
                qty=5,
                side=trade_side,
                entry_price=calculated_prices['current_price'],
                take_profit_price=calculated_prices['take_profit_price'],
                stop_loss_price=calculated_prices['stop_loss_price']
            )

        print("\nScript execution finished. Uncomment other example function calls to test functionality.")
'''
