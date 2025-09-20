import os
import json
import time
import pytz
import pandas as pd
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestTradeRequest,CryptoLatestQuoteRequest
import requests
from alpaca.trading.requests import (
    MarketOrderRequest,
    ReplaceOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
    LimitOrderRequest,
    GetOrdersRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus, PositionIntent, OrderType
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.common.exceptions import APIError

from typing import Optional, Tuple, List
from alpaca.trading.models import Order

# --- Import Your Project's Config ---
import config as cfg


# --- Your API Credentials ---
# Make sure to set these as environment variables for security
API_KEY = os.environ.get('APCA_API_KEY_ID', cfg.ALPACA_API_KEY_ID)
SECRET_KEY = os.environ.get('APCA_API_SECRET_KEY', cfg.ALPACA_SECRET_KEY)
# Use paper=True for paper trading, paper=False for live trading
# PAPER = False #True
PAPER = cfg.PAPER

# --- Initialize Clients ---
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
data_client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)
crypto_data_client = data_client


#-------------------------------------------------------------------------------
#-------------------------------price calculation-------------------------------
#-------------------------------------------------------------------------------

###Average True Range (ATR) price CALCULATION
###problem with time frame module importing
def calculate_sl_tp_prices_atr(symbol: str, side: OrderSide, sl_atr_multiplier: float, risk_reward_ratio: float):
    """
    Calculates stop-loss and take-profit prices based on the Average True Range (ATR).

    Args:
        symbol (str): The Crypto symbol.
        side (OrderSide): The side of the trade (BUY or SELL).
        sl_atr_multiplier (float): The multiplier for the ATR to determine the stop loss distance.
        risk_reward_ratio (float): The desired risk-to-reward ratio for the take profit.

    Returns:
        dict: A dictionary with 'stop_loss_price' and 'take_profit_price', or None on error.
    """
    print(f"\nCalculating SL/TP prices for a {side.value} on {symbol} using ATR...")
    try:
        # 1. Get historical data to calculate ATR
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30) # Fetch enough data for a 14-period ATR
        bars_request = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        # --- CORRECTED METHOD NAME ---
        bars = data_client.get_crypto_bars(bars_request).df

        # Calculate True Range (TR)
        bars['high-low'] = bars['high'] - bars['low']
        bars['high-prev_close'] = abs(bars['high'] - bars['close'].shift(1))
        bars['low-prev_close'] = abs(bars['low'] - bars['close'].shift(1))
        bars['tr'] = bars[['high-low', 'high-prev_close', 'low-prev_close']].max(axis=1)

        # Calculate 14-day ATR
        bars['atr'] = bars['tr'].ewm(alpha=1/14, adjust=False).mean()
        latest_atr = bars['atr'].iloc[-1]
        print(f"  - Latest 14-Day ATR for {symbol}: ${latest_atr:.2f}")

        # 2. Get the latest trade price
        trade_request = CryptoLatestTradeRequest(symbol_or_symbols=symbol)
        # --- CORRECTED METHOD NAME ---
        latest_trade = data_client.get_crypto_latest_trade(trade_request)
        current_price = latest_trade[symbol].price
        print(f"  - Current Price for {symbol}: ${current_price:.2f}")

        # 3. Calculate SL/TP prices
        risk_distance = latest_atr * sl_atr_multiplier

        if side == OrderSide.BUY:
            stop_loss_price = current_price - risk_distance
            take_profit_price = current_price + (risk_distance * risk_reward_ratio)
        elif side == OrderSide.SELL:
            stop_loss_price = current_price + risk_distance
            take_profit_price = current_price - (risk_distance * risk_reward_ratio)
        else:
            print("Error: Invalid side provided.")
            return None

        prices = {
            'current_price' :round(current_price, 2),
            'stop_loss_price': round(stop_loss_price, 2),
            'take_profit_price': round(take_profit_price, 2)
        }
        print(f"  - Calculated Stop Loss: ${prices['stop_loss_price']:.2f}")
        print(f"  - Calculated Take Profit: ${prices['take_profit_price']:.2f}")

        return prices

    except Exception as e:
        print(f"An error occurred during ATR price calculation for {symbol}: {e}")
        return None
###regular price calculation
def calculate_sl_tp_prices(symbol: str, side: OrderSide, sl_percentage: float, tp_percentage: float):
    """
    Calculates stop-loss and take-profit prices based on the current market price.

    Args:
        symbol (str): The Crypto symbol.
        side (OrderSide): The side of the trade (BUY or SELL).
        sl_percentage (float): The percentage away from the current price to set the stop loss.
        tp_percentage (float): The percentage away from the current price to set the take profit.

    Returns:
        dict: A dictionary with 'stop_loss_price' and 'take_profit_price', or None on error.
    """
    print(f"\nCalculating SL/TP prices for a {side.value} on {symbol}...")
    try:
        # 1. Get the latest trade price for the symbol
        trade_request = CryptoLatestTradeRequest(symbol_or_symbols=symbol)
        # --- CORRECTED METHOD NAME ---
        latest_trade = data_client.get_crypto_latest_trade(trade_request)
        current_price = latest_trade[symbol].price
        print(f"  - Current Price for {symbol}: ${current_price:.2f}")

        # 2. Calculate prices based on the side of the trade
        if side == OrderSide.BUY:
            stop_loss_price = current_price * (1 - sl_percentage / 100)
            take_profit_price = current_price * (1 + tp_percentage / 100)
        elif side == OrderSide.SELL:
            stop_loss_price = current_price * (1 + sl_percentage / 100)
            take_profit_price = current_price * (1 - tp_percentage / 100)
        else:
            print("Error: Invalid side provided. Must be OrderSide.BUY or OrderSide.SELL.")
            return None

        # Round to 2 decimal places for typical Crypto prices
        prices = {
            'current_price' :round(current_price, 2),
            'stop_loss_price': round(stop_loss_price, 2),
            'take_profit_price': round(take_profit_price, 2)
        }
        print(f"  - Calculated Stop Loss: ${prices['stop_loss_price']:.2f}")
        print(f"  - Calculated Take Profit: ${prices['take_profit_price']:.2f}")

        return prices

    except APIError as e:
        print(f"Alpaca API Error calculating prices for {symbol}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while calculating prices for {symbol}: {e}")
        return None


###calculates limit price for crypto to avoid taker fee
##this function does not work
def get_latest_quote(symbol: str) -> dict | None:
    """
    Fetches the latest real-time bid and ask price for a crypto symbol.

    Args:
        symbol (str): The symbol to get a quote for (e.g., 'BTC/USD').

    Returns:
        dict: The latest quote object from the Alpaca API, or None if it fails.
    """
    try:
        # Define the request parameters
        request_params = CryptoLatestQuoteRequest(symbol_or_symbols=[symbol])

        # Fetch the latest quote using the data client
        latest_quote = crypto_data_client.get_crypto_latest_quote(request_params)

        # The response is a dictionary, so we access the quote by symbol
        if symbol in latest_quote:
            return latest_quote[symbol]
        else:
            return None

    except Exception as e:
        print(f"Error fetching quote for {symbol} from API: {e}")
        return None


def calculate_maker_prices(symbol: str) -> dict | None:
    """
    Calculates limit prices to help ensure an order is a maker order.

    Args:
        symbol (str): The symbol to get prices for (e.g., 'BTC/USD').

    Returns:
        dict: A dictionary with 'buy_limit_price' and 'sell_limit_price', or None if quote fails.
    """
    print(f"\nFetching latest quote for {symbol} to calculate maker prices...")
    try:
        # Call our newly defined function to get the latest quote data
        latest_quote = get_latest_quote(symbol)

        if not latest_quote:
            print(f"  - Could not retrieve quote for {symbol}.")
            return None

        # Extract the bid and ask prices from the quote object
        bid_price = latest_quote.bid_price
        ask_price = latest_quote.ask_price

        print(f"  - ✅ Success! Best Bid: {bid_price}, Best Ask: {ask_price}")

        prices = {
            'buy_limit_price': bid_price,
            'sell_limit_price': ask_price
        }
        return prices

    except Exception as e:
        print(f"An error occurred in calculate_maker_prices for {symbol}: {e}")
        return None


#-------------------------------------------------------------------------------
#------------------------------Order submission---------------------------------
#-------------------------------------------------------------------------------

###limit bracket order
def submit_limit_bracket_order(symbol: str, qty: float, side: OrderSide, limit_price: float, take_profit_price: float, stop_loss_price: float):
    """
    Submits a new limit order with specified take-profit and stop-loss levels.

    Args:
        symbol (str): The symbol to trade.
        qty (float): The quantity of shares to trade.
        side (OrderSide): The side of the order (BUY or SELL).
        limit_price (float): The limit price for the main order.
        take_profit_price (float): The take-profit limit price.
        stop_loss_price (float): The stop-loss price.

    Returns:
        Order: The new order object if successful, otherwise None.
    """
    print(f"\nAttempting to submit a new limit bracket order for {qty} shares of {symbol}...")
    print(f"  - Side: {side.value}")
    print(f"  - Limit Price: {limit_price}")
    print(f"  - Take Profit: {take_profit_price}")
    print(f"  - Stop Loss: {stop_loss_price}")

    try:
        # Prepare the request objects for the bracket legs
        take_profit_req = TakeProfitRequest(limit_price=take_profit_price)
        stop_loss_req = StopLossRequest(stop_price=stop_loss_price)

        # Prepare the main limit order request with the bracket legs
        limit_bracket_order_data = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.GTC,
            limit_price=limit_price,
            order_class=OrderClass.BRACKET,
            take_profit=take_profit_req,
            stop_loss=stop_loss_req
        )

        # Submit the order
        new_order = trading_client.submit_order(order_data=limit_bracket_order_data)

        print(f"Successfully submitted new limit bracket order: {new_order.id}")
        return new_order

    except APIError as e:
        print(f"Alpaca API Error submitting limit bracket order: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while submitting limit bracket order: {e}")
        return None

###simple market order
def submit_simple_order(symbol: str, qty: float, side: OrderSide):
    """
    Submits a simple market order without stop-loss or take-profit.

    Args:
        symbol (str): The symbol to trade.
        qty (float): The quantity of shares to trade.
        side (OrderSide): The side of the order (BUY or SELL).

    Returns:
        Order: The new order object if successful, otherwise None.
    """
    print(f"\nAttempting to submit a simple market order for {qty} shares of {symbol}...")
    print(f"  - Side: {side.value}")
    try:
        # Prepare the market order request
        simple_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.GTC
        )
        # Submit the order
        new_order = trading_client.submit_order(order_data=simple_order_data)
        print(f"Successfully submitted simple market order: {new_order.id}")
        return new_order
    except APIError as e:
        print(f"Alpaca API Error submitting simple order: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while submitting simple order: {e}")
        return None

###simple limit order
def submit_limit_order(symbol: str, qty: float, side: OrderSide, limit_price: float):
    """
    Submits a simple limit order.

    Args:
        symbol (str): The symbol to trade.
        qty (float): The quantity of shares to trade.
        side (OrderSide): The side of the order (BUY or SELL).
        limit_price (float): The price at which to execute the order or better.

    Returns:
        Order: The new order object if successful, otherwise None.
    """
    print(f"\nAttempting to submit a LIMIT order for {qty} shares of {symbol} at ${limit_price}...")
    print(f"  - Side: {side.value}")
    try:
        # Prepare the limit order request
        limit_order_data = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            limit_price=limit_price,
            time_in_force=TimeInForce.GTC  # Good-Til-Canceled is a common choice
        )
        # Submit the order
        new_order = trading_client.submit_order(order_data=limit_order_data)
        print(f"Successfully submitted limit order: {new_order.id}")
        return new_order
    except APIError as e:
        print(f"Alpaca API Error submitting limit order: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while submitting limit order: {e}")
        return None


def submit_bracket_order(symbol: str, qty: float, side: OrderSide, take_profit_price: float, stop_loss_price: float):
    """
    Submits a new market order with specified take-profit and stop-loss levels (a bracket order).

    Args:
        symbol (str): The symbol to trade.
        qty (float): The quantity of shares to trade.
        side (OrderSide): The side of the order (BUY or SELL).
        take_profit_price (float): The take-profit limit price.
        stop_loss_price (float): The stop-loss price.

    Returns:
        Order: The new order object if successful, otherwise None.
    """
    print(f"\nAttempting to submit a new bracket order for {qty} shares of {symbol}...")
    print(f"  - Side: {side.value}")
    print(f"  - Take Profit: {take_profit_price}")
    print(f"  - Stop Loss: {stop_loss_price}")

    try:
        # Prepare the request objects for the bracket legs
        take_profit_req = TakeProfitRequest(limit_price=take_profit_price)
        stop_loss_req = StopLossRequest(stop_price=stop_loss_price)

        # Prepare the main market order request with the bracket legs
        bracket_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=take_profit_req,
            stop_loss=stop_loss_req
        )

        # Submit the order
        new_order = trading_client.submit_order(order_data=bracket_order_data)

        print(f"Successfully submitted new bracket order: {new_order.id}")
        return new_order

    except APIError as e:
        print(f"Alpaca API Error submitting bracket order: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while submitting bracket order: {e}")
        return None


#-------------------------------------------------------------------------------
#-----------------------------getting order info--------------------------------
#-------------------------------------------------------------------------------
###
def get_open_crypto_orders(trading_client: TradingClient) -> list:
    """
    Fetches all open orders from an Alpaca account and filters for crypto assets.

    Args:
        trading_client (TradingClient): An authenticated Alpaca trading client instance.

    Returns:
        list: A list of open Order objects for crypto assets. Returns an empty list if none are found.
    """
    print("🔎 Fetching all open orders...")
    try:
        request_params = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        all_open_orders = trading_client.get_orders(filter=request_params)

        # --- MODIFIED LOGIC ---
        # Filter the list by checking the asset_class attribute directly.
        crypto_orders = [
            order for order in all_open_orders if order.asset_class == AssetClass.CRYPTO
        ]

        print(f"✅ Found {len(crypto_orders)} open crypto order(s).")
        return crypto_orders

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return []


###
def get_open_position_by_symbol(symbol: str):
    """
    Checks if a position is currently open for a given symbol.

    Args:
        symbol (str): The Crypto symbol to check.

    Returns:
        Position: The Position object if a position exists, otherwise None.
    """
    print(f"\nChecking for an open position in {symbol}...")
    try:
        position = trading_client.get_open_position(symbol)
        print(f"  - Position found for {symbol}: {position.qty} shares @ avg entry ${position.avg_entry_price}")
        return position
    except APIError as e:
        if "position not found" in str(e):
            print(f"  - No position found for {symbol}.")
            return None
        else:
            print(f"Alpaca API Error checking position for {symbol}: {e}")
            return None
    except Exception as e:
        print(f"An unexpected error occurred while checking position for {symbol}: {e}")
        return None

###
def get_all_open_orders_by_symbol(symbol: str):
    """
    Retrieves all open orders for a specific symbol.

    Args:
        symbol (str): The Crypto symbol to check for open orders.

    Returns:
        list[Order]: A list of open Order objects, or an empty list if none are found.
    """
    print(f"\nGetting all open orders for {symbol}...")
    try:
        # Create a request object to filter orders
        request_params = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol]
        )
        orders = trading_client.get_orders(filter=request_params)
        print(f"  - Found {len(orders)} open order(s) for {symbol}.")
        return orders
    except APIError as e:
        print(f"Alpaca API Error getting orders for {symbol}: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while getting orders for {symbol}: {e}")
        return []

###
def get_all_position_symbols() -> List[str]:
    """
    note: this only returns crypto symbols
    Retrieves a list of symbols for all currently open positions.

    Returns:
        List[str]: A list of Crypto symbols, or an empty list if no positions are open.
    """
    print("\nGetting all open position symbols...")
    try:
        positions = trading_client.get_all_positions()
        if not positions:
            print("  - No open positions found.")
            return []

        symbols = []
        for position in positions:
            # --- NEW LOGIC: Differentiate between stocks and crypto ---
            if position.asset_class == 'crypto':

                # --- FIX: Reformat symbol for the data client ---
                # Assumes a 3-character quote currency like USD, EUR, etc.
                api_symbol = f"{position.symbol[:-3]}/{position.symbol[-3:]}"
                symbols = symbols + [api_symbol]

            elif position.asset_class == 'stock':
                symbols = symbols + [position.symbol]

        print(f"  - Found {len(symbols)} position(s): {', '.join(symbols)}")
        return symbols

    except APIError as e:
        print(f"Alpaca API Error getting position symbols: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while getting position symbols: {e}")
        return []


#-------------------------------------------------------------------------------
#------------position and order cancellation or close --------------------------
#-------------------------------------------------------------------------------


##cancel order by id
###
def cancel_order_by_id(order_id: str):
    """
    Cancels a specific order by its ID.

    Args:
        order_id (str): The ID of the order to cancel.

    Returns:
        bool: True if cancellation was successfully requested, otherwise False.
    """
    print(f"\nAttempting to cancel order {order_id}...")
    try:
        # The cancel_order_by_id method does not return anything on success,
        # but will raise an APIError if it fails.
        trading_client.cancel_order_by_id(order_id=order_id)
        print(f"Successfully submitted cancellation request for order {order_id}.")
        return True
    except APIError as e:
        print(f"Alpaca API Error cancelling order {order_id}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while cancelling order {order_id}: {e}")
        return False


#----------improved function to avoid fast-moving market problem majid 2/9/1979
def close_position_by_symbol(symbol: str, timeout_seconds: int = 10):
    """
    Closes the entire open position for a symbol using a robust polling method.
    It first cancels any open orders and confirms their cancellation before proceeding.

    Args:
        symbol (str): The symbol of the position to close.
        timeout_seconds (int): Maximum seconds to wait for order cancellations to be confirmed.

    Returns:
        Order: The closing order object if successful, otherwise None.
    """
    print(f"\nAttempting to robustly close position for {symbol}...")
    try:
        # 1. Find and cancel all open orders for the symbol
        print(f"  - First, finding and cancelling any open orders for {symbol}.")
        open_orders = get_all_open_orders_by_symbol(symbol)

        if not open_orders:
            print(f"  - No open orders found for {symbol} to cancel.")
        else:
            order_ids_to_check = [order.id for order in open_orders]
            for order_id in order_ids_to_check:
                cancel_order_by_id(order_id) # This function should handle its own print statements

            # 2. Poll for cancellation confirmation
            print(f"  - Waiting for cancellation confirmation (timeout: {timeout_seconds}s)...")
            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                all_orders_closed = True
                # Re-fetch orders to check their live status
                orders_to_verify = [trading_client.get_order_by_id(oid) for oid in order_ids_to_check]

                for order in orders_to_verify:
                    # An order is considered "done" if it's canceled, filled, expired, etc.
                    if order.status not in ['canceled', 'filled', 'expired', 'rejected']:
                        all_orders_closed = False
                        break # No need to check others in this iteration

                if all_orders_closed:
                    print("  - All open orders have been confirmed as closed.")
                    break

                time.sleep(1) # Wait 1 second before polling again
            else: # This 'else' belongs to the 'while' loop, it runs if the loop finishes without a 'break'
                print(f"  - TIMEOUT: Not all orders were confirmed closed within {timeout_seconds} seconds.")
                print("  - Aborting position close to avoid potential errors.")
                return None

        # 3. Proceed to close the position
        print(f"  - Proceeding to close the position for {symbol}.")
        closing_order = trading_client.close_position(symbol)
        print(f"Successfully submitted closing order for {symbol}: {closing_order.id}")
        return closing_order

    except APIError as e:
        if "position not found" in str(e):
            print(f"  - No position to close for {symbol}.")
            return None
        else:
            print(f"Alpaca API Error closing position for {symbol}: {e}")
            return None
    except Exception as e:
        print(f"An unexpected error occurred while closing position for {symbol}: {e}")
        return None
#-----------
###
## NEW FUNCTION TO CHECK AND TAKE PROFIT
##this functions ckeck total profit/loss percentage of all postions and close those above threshold
##modified for crypto majid 2/9/2025
def check_and_take_profit(profit_threshold_percentage: float, crypto_data_client: CryptoHistoricalDataClient = data_client ):
    """
    Checks all open positions (both stock and crypto) and closes any
    that have a profit/loss percentage above the specified threshold.
    """
        # --- ADD THIS DIAGNOSTIC CODE AT THE TOP OF YOUR SCRIPT ---
    try:
        print("\n--- Verifying All Open Positions ---")
        all_positions = trading_client.get_all_positions()
        if not all_positions:
            print("No open positions found at all.")
        else:
            position_symbols = [p.symbol for p in all_positions]
            print(f"API is reporting positions for: {position_symbols}")
        print("------------------------------------\n")
    except Exception as e:
        print(f"Error fetching positions: {e}")
    # -------------------------------------------------------------

    # # If no specific client is passed in, use the global default.
    # if crypto_data_client is None:
    #     crypto_data_client = data_client

    if profit_threshold_percentage >= 0:
        print(f"\nChecking all open positions for take-profit opportunity (Threshold: {profit_threshold_percentage}%)...")
        try:
            positions = trading_client.get_all_positions()
            if not positions:
                print("  - No open positions found.")
                return

            profit_threshold_decimal = profit_threshold_percentage / 100.0
            closed_positions = []
            for position in positions:
                unrealized_plpc = 0.0 # Initialize P/L percentage

                # --- NEW LOGIC: Differentiate between stocks and crypto ---
                if position.asset_class == 'crypto':
                    # Manually calculate P/L for crypto because crypto does not have it like stocks
                    # --- FIX: Reformat symbol for the data client ---
                    # Assumes a 3-character quote currency like USD, EUR, etc.
                    api_symbol = f"{position.symbol[:-3]}/{position.symbol[-3:]}"

                    print(f"  - Checking crypto {position.symbol} (using API symbol {api_symbol}):")
                    entry_price = float(position.avg_entry_price)

                    # Fetch the current market price using the CORRECTED symbol
                    trade_request = CryptoLatestTradeRequest(symbol_or_symbols=api_symbol)
                    latest_trade = crypto_data_client.get_crypto_latest_trade(trade_request)

                    # Use the CORRECTED symbol to access the price data
                    current_price = latest_trade[api_symbol].price

                    if position.side == 'long':
                        unrealized_plpc = (current_price - entry_price) / entry_price
                    else: # short
                        unrealized_plpc = (entry_price - current_price) / entry_price
                else:
                    # For non-crypto assets like stocks and options
                    print(f"  - Checking stock/other: {position.symbol}")

                    # --- FIX: Check if P/L data is available before using it ---
                    if position.unrealized_plpc is not None:
                        unrealized_plpc = float(position.unrealized_plpc)
                    else:
                        print(f"    - WARNING: P/L data not available for {position.symbol}. Skipping.")
                        continue # Skips to the next position in the loop
                    # --- END OF FIX ---
                # --- END OF NEW LOGIC ---

                unrealized_pl_percent = unrealized_plpc * 100
                print(f"    - Current P/L is {unrealized_pl_percent:.2f}%")

                if unrealized_plpc > profit_threshold_decimal:
                    print(f"    - PROFIT TARGET MET for {position.symbol}! "
                          f"({unrealized_pl_percent:.2f}% > {profit_threshold_percentage}%)")
                    print(f"    - Closing position for {position.symbol} to take profit.")
                    close_position_by_symbol(position.symbol)
                    closed_positions.append(position.symbol)
                else:
                    print(f"    - Profit target not met for {position.symbol}.")
            return closed_positions

        except APIError as e:
            print(f"Alpaca API Error when checking positions for take-profit: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during take-profit check: {e}")
    else:
        print("Profit percentage threshold should be positive or zero.")

###
# NEW FUNCTION TO CHECK AND TAKE PROFIT
##this functions ckeck total profit/loss percentage of all postions and closes those below threshold
# modified for crypto trade 2/9/2025
def check_and_stop_loss(loss_threshold_percentage: float, crypto_data_client: CryptoHistoricalDataClient = data_client):
    """
    Checks all open positions (both stock and crypto) and closes any
    that have a profit/loss percentage below the specified threshold.
    """
    # # If no specific client is passed in, use the global default.
    # if crypto_data_client is None:
    #     crypto_data_client = data_client

    if loss_threshold_percentage < 0:
        print(f"\nChecking all open positions for stop-loss opportunity (Threshold: {loss_threshold_percentage}%)...")
        try:
            positions = trading_client.get_all_positions()
            if not positions:
                print("  - No open positions found.")
                return

            loss_threshold_decimal = loss_threshold_percentage / 100.0
            closed_positions = []
            for position in positions:
                unrealized_plpc = 0.0 # Initialize P/L percentage

                # --- NEW LOGIC: Differentiate between stocks and crypto ---
                if position.asset_class == 'crypto':
                    # Manually calculate P/L for crypto because crypto does not have it like stocks
                    # --- FIX: Reformat symbol for the data client ---
                    # Assumes a 3-character quote currency like USD, EUR, etc.
                    api_symbol = f"{position.symbol[:-3]}/{position.symbol[-3:]}"

                    print(f"  - Checking crypto {position.symbol} (using API symbol {api_symbol}):")
                    entry_price = float(position.avg_entry_price)

                    # Fetch the current market price using the CORRECTED symbol
                    trade_request = CryptoLatestTradeRequest(symbol_or_symbols=api_symbol)
                    latest_trade = crypto_data_client.get_crypto_latest_trade(trade_request)

                    # Use the CORRECTED symbol to access the price data
                    current_price = latest_trade[api_symbol].price

                    if position.side == 'long':
                        unrealized_plpc = (current_price - entry_price) / entry_price
                    else: # short
                        unrealized_plpc = (entry_price - current_price) / entry_price
                else:
                    # For non-crypto assets like stocks and options
                    print(f"  - Checking stock/other: {position.symbol}")

                    # --- FIX: Check if P/L data is available before using it ---
                    if position.unrealized_plpc is not None:
                        unrealized_plpc = float(position.unrealized_plpc)
                    else:
                        print(f"    - WARNING: P/L data not available for {position.symbol}. Skipping.")
                        continue # Skips to the next position in the loop
                    # --- END OF FIX ---
                # --- END OF NEW LOGIC ---

                unrealized_pl_percent = unrealized_plpc * 100
                print(f"    - Current P/L is {unrealized_pl_percent:.2f}%")

                if unrealized_plpc < loss_threshold_decimal:
                    print(f"    - LOSS TARGET MET for {position.symbol}! "
                          f"({unrealized_pl_percent:.2f}% < {loss_threshold_percentage}%)")
                    print(f"    - Closing position for {position.symbol} to stop loss.")
                    close_position_by_symbol(position.symbol)
                    closed_positions.append(position.symbol)
                else:
                    print(f"    - Loss target not met for {position.symbol}.")
            return closed_positions
        except APIError as e:
            print(f"Alpaca API Error when checking positions for stop-loss: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during stop-loss check: {e}")
    else:
        print("Loss percentage threshold should be negative.")
###
def panic_button_cancel_all_orders():
    """
    Acts as a panic button to cancel all open orders on the account.

    Returns:
        bool: True if the cancellation request was successful, otherwise False.
    """
    print("\n" + "!"*10 + " PANIC BUTTON ACTIVATED " + "!"*10)
    print("Attempting to cancel ALL open orders...")
    try:
        # cancel_orders() returns a list of responses for each order cancellation attempt.
        cancel_responses = trading_client.cancel_orders()

        if not cancel_responses:
            print("No open orders to cancel.")
        else:
            print(f"Successfully submitted cancellation requests for {len(cancel_responses)} order(s).")
            # You can optionally iterate through responses to check individual statuses
            for response in cancel_responses:
                print(f"  - Order ID: {response.id}, Status Code: {response.status}")

        return True
    except APIError as e:
        print(f"Alpaca API Error during panic cancel: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during panic cancel: {e}")
        return False

###
def schedule_close_all_positions(close_datetime_str: str, timezone_str: str = "America/New_York"):
    """
    Schedules a task to close all open positions at a specific date and time.

    Args:
        close_datetime_str (str): The target date and time in 'YYYY-MM-DD HH:MM:SS' format.
        timezone_str (str): The timezone for the target time (e.g., 'America/New_York').
    """
    try:
        # 1. Set up the timezone and target close time
        target_timezone = pytz.timezone(timezone_str)
        target_close_time = target_timezone.localize(datetime.strptime(close_datetime_str, '%Y-%m-%d %H:%M:%S'))
        print(f"Scheduled to close all positions on: {target_close_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        # 2. Loop until it's time to close
        while True:
            current_time = datetime.now(target_timezone)
            if current_time >= target_close_time:
                print("\n" + "="*50)
                print(f"TARGET TIME REACHED: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}. Closing all positions.")
                print("="*50)

                # Get all position symbols
                symbols_to_close = get_all_position_symbols()

                if not symbols_to_close:
                    print("No positions to close.")
                    break

                # Close each position
                for symbol in symbols_to_close:
                    close_position_by_symbol(symbol)

                print("\nAll positions have been instructed to close.")
                break # Exit the loop
            else:
                time_remaining = target_close_time - current_time
                # Use \r to return to the beginning of the line and overwrite it
                print(f"\rNot time yet. Current time: {current_time.strftime('%H:%M:%S')}. Time remaining: {str(time_remaining).split('.')[0]}", end="")
                time.sleep(60) # Wait for 60 seconds before checking again

    except Exception as e:
        print(f"\nAn error occurred in the scheduler: {e}")

###
def close_positions_before_market_close(client: TradingClient = trading_client, minutes_before_close: int = 5) -> bool:
    """
    Checks if the market is about to close and liquidates all positions if it is.

    Args:
        client (TradingClient): An authenticated Alpaca trading client instance.
        minutes_before_close (int): The number of minutes before the close to trigger liquidation. Defaults to 5.

    Returns:
        bool: True if positions were closed, False otherwise.
    """
    try:
        clock = client.get_clock()

        if not clock.is_open:
            print("Market is currently closed. No action taken.")
            return False

        market_close_time = clock.next_close
        time_until_close = market_close_time - clock.timestamp

        print(f"Market is open. Time until close: {time_until_close}")

        # Check if we are within the liquidation window
        if time_until_close.total_seconds() <= minutes_before_close * 60:
            print(f"Within {minutes_before_close} minutes of market close. Closing all positions.")

            # Liquidate all positions and cancel open orders
            responses = client.close_all_positions(cancel_orders=True)

            print("Liquidation responses received.")
            # Optional: You can still loop through responses if you want to log them
            # for resp in responses:
            #     print(f"  - Symbol: {resp.symbol}, Status: {resp.status}")

            return True # Indicate that the close action was taken
        else:
            print(f"Not within the closing window. No action taken.")
            return False

    except APIError as e:
        print(f"An error occurred with the Alpaca API: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

###for preventing bot to leave unexecuted orders more than any minutes determined in bot master code
###note this function cancels all SL and TP limits for all positions but keep the positions unprotected
def cancel_all_pending_orders():
    """
    Cancels all open orders that have not been fully executed, without
    affecting any existing open positions.

    This is ideal for clearing out pending limit, stop, or the take-profit/stop-loss
    legs of bracket orders at the end of a trading session. It leaves your
    actual held positions untouched.

    Returns:
        bool: True if the cancellation request was successful, otherwise False.
    """
    print("\n--- Attempting to cancel all pending (unexecuted) orders ---")
    try:
        # The cancel_orders() method targets all orders in an 'open' state.
        # It returns a list of responses, one for each order it attempted to cancel.
        # We use this response as the single source of truth.
        cancel_responses = trading_client.cancel_orders()

        if not cancel_responses:
            print("  - No open orders were found to cancel.")
            return True

        print(f"  - Submitted cancellation requests for {len(cancel_responses)} order(s).")

        # Iterate through the responses to confirm the status of each cancellation
        successful_cancellations = 0
        failed_cancellations = 0

        for response in cancel_responses:
            # A successful response object contains the order ID and its status code.
            # A status code of 200 means the cancellation was accepted.
            if hasattr(response, 'status') and response.status == 200:
                print(f"    - Confirmed cancellation for Order ID: {response.id}")
                successful_cancellations += 1
            else:
                # This part handles cases where a specific order couldn't be cancelled.
                print(f"    - Could not confirm cancellation for Order ID: {response.id}. Status: {response.status}")
                failed_cancellations += 1

        print(f"\nFinished. Successfully cancelled {successful_cancellations} order(s).")
        if failed_cancellations > 0:
            print(f"Failed to cancel {failed_cancellations} order(s). Please check your dashboard.")

        return True

    except APIError as e:
        print(f"An Alpaca API Error occurred during cancellation: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during cancellation: {e}")
        return False



###finds open ordres by get_open_crypto_orders and cancel them
###
def cancel_all_open_crypto_orders(trading_client: TradingClient = trading_client):
    """
    Finds all open crypto orders and cancels them.

    Args:
        trading_client (TradingClient): An authenticated Alpaca trading client instance.
    """
    # 1. First, get the list of open crypto orders
    orders_to_cancel = get_open_crypto_orders(trading_client)

    if not orders_to_cancel:
        print("\nNo open crypto orders to cancel.")
        return

    print(f"\nPreparing to cancel the following {len(orders_to_cancel)} order(s):")
    for order in orders_to_cancel:
        print(f"  - ID: {order.id}, Symbol: {order.symbol}, Qty: {order.qty}, Side: {order.side.value}")

    # 2. Iterate through the list and cancel each order by its ID
    print("\nSending cancellation requests...")
    for order in orders_to_cancel:
        try:
            # The API call to cancel a specific order
            trading_client.cancel_order_by_id(order_id=order.id)
            print(f"✅ Successfully cancelled order {order.id} for {order.symbol}.")
        except Exception as e:
            print(f"❌ Failed to cancel order {order.id}. Reason: {e}")

###this is for quick crypto market liquidatoin
def panic_button_crypto():
    """
    A panic button that cancels all open orders and closes all crypto positions.

    This function performs two critical actions in sequence:
    1. Cancels ALL open orders (both stock and crypto) to prevent any new trades.
    2. Submits market orders to close every open CRYPTO position.
    """
    print("\n🚨 PANIC BUTTON ACTIVATED FOR CRYPTO 🚨")

    # --- Step 1: Cancel ALL open orders ---
    # This is the safest first step in a panic situation to prevent unwanted fills.
    print("\n--- Step 1: Cancelling ALL open orders... ---")
    try:
        # cancel_statuses = trading_client.cancel_orders()
        cancel_all_open_crypto_orders()
        # if cancel_statuses:
        #     print(f"Successfully requested cancellation for {len(cancel_statuses)} open order(s).")
        # else:
        #     print("No open orders to cancel.")
    except APIError as e:
        print(f"An API error occurred while cancelling orders: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


    # --- Step 2: Close all open CRYPTO positions ---
    print("\n--- Step 2: Closing all open crypto positions... ---")
    try:
        positions = trading_client.get_all_positions()
        # Filter for crypto positions, which typically have a '/' in the symbol
        # print('These are positions from trading_client.list_positions:',positions)
        crypto_positions = [p for p in positions if p.asset_class == 'crypto']

        if not crypto_positions:
            print("No open crypto positions found to close.")
            return

        print(f"Found {len(crypto_positions)} crypto position(s) to close.")
        for position in crypto_positions:
            try:
                print(f"  - Closing position for {position.symbol}...")
                trading_client.close_position(position.symbol)
                print(f"    - Successfully submitted market sell order for {position.symbol}.")
            except APIError as e:
                print(f"    - FAILED to close position for {position.symbol}: {e}")

        print("\n✅ Panic button process complete.")

    except APIError as e:
        print(f"An API error occurred while fetching or closing positions: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



#-----------------------------------------------------------------------------#
#---------------------------order modification----------------------------------
#-------------------------------------------------------------------------------
###
def find_and_modify_stop_loss(symbol: str, new_stop_price: float):
    """
    Finds the specific stop-loss order for a symbol and modifies its stop price.

    Args:
        symbol (str): The Crypto symbol of the order to find and modify.
        new_stop_price (float): The new stop-loss price to set.

    Returns:
        Order: The updated order object if successful, otherwise None.
    """
    print(f"\nAttempting to find and modify stop-loss for {symbol}...")

    # 1. Find all open orders for the symbol
    open_orders = get_all_open_orders_by_symbol(symbol)

    # 2. Find the specific stop-loss order from the list
    stop_loss_order = None
    for order in open_orders:
        # A stop-loss order is identified by having a stop_price
        if order.stop_price is not None:
            stop_loss_order = order
            break # Assume only one stop-loss per symbol for simplicity

    if not stop_loss_order:
        print(f"  - No open stop-loss order found for {symbol}.")
        return None

    order_id = stop_loss_order.id
    print(f"  - Found stop-loss order {order_id} to modify.")

    # 3. Check if the price actually needs to be changed
    if stop_loss_order.stop_price == new_stop_price:
        print(f"  - No change needed. Stop-loss for order {order_id} is already at ${new_stop_price}.")
        return stop_loss_order

    # 4. Call the replace order method with only the new stop price
    try:
        replace_order_data = ReplaceOrderRequest(stop_price=new_stop_price)

        new_order = trading_client.replace_order_by_id(
            order_id=order_id,
            order_data=replace_order_data
        )
        print(f"Successfully modified stop-loss for order {order_id} to ${new_stop_price}.")
        return new_order

    except APIError as e:
        print(f"Alpaca API Error modifying stop-loss for order {order_id}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while modifying stop-loss for order {order_id}: {e}")
        return None

###this one looks for child legs, use when there is no parent_order
def update_tp_sl_for_open_position(
    symbol: str,
    *,
    trading_client: TradingClient = trading_client,
    new_take_profit: Optional[float] = None,  # new TP limit price
    new_stop_price: Optional[float] = None,   # new SL stop price
    new_stop_limit: Optional[float] = None,   # optional SL limit for STOP_LIMIT
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find any OPEN take-profit / stop-loss child orders (legs) for `symbol`
    and replace them with the provided prices. Works even if the parent order
    is filled/closed or if Alpaca doesn't return legs on the parent.

    Returns: (tp_leg_id_replaced, sl_leg_id_replaced)
    Raises:  ValueError with a clear message if nothing is found/updated.
    """
    symbol = symbol.upper()

    # 1) Ensure there is an open position (raises if not found)
    try:
        pos = trading_client.get_open_position(symbol)
    except Exception:
        raise ValueError(f"No open position found for {symbol}.")

    if float(pos.qty) == 0:
        raise ValueError(f"No open position quantity for {symbol}.")

    # 2) Pull OPEN orders for this symbol and find child legs (those with parent_order_id)
    open_orders = trading_client.get_orders(
        GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol],
            nested=True  # ask Alpaca to include legs where applicable
        )
    )

    # Identify TP and SL children directly
    tp_child = None
    sl_child = None

    for o in open_orders:
        # We only care about children of a multi-leg structure
        if getattr(o, "parent_order_id", None) is None:
            continue

        # Heuristics:
        # - Take-profit legs are LIMIT orders without a stop price
        # - Stop-loss legs have a stop price (STOP or STOP_LIMIT)
        if o.type == OrderType.LIMIT and getattr(o, "stop_price", None) is None:
            tp_child = tp_child or o
        if getattr(o, "stop_price", None) is not None:
            sl_child = sl_child or o

    if tp_child is None and sl_child is None:
        raise ValueError(
            "No open TP/SL child orders found. "
            "This usually means the position was opened without a bracket/OCO, "
            "or both exits were already filled/canceled."
        )

    # 3) Replace what the user asked for
    replaced_tp_id = None
    replaced_sl_id = None

    if tp_child is not None and new_take_profit is not None:
        tp_update = ReplaceOrderRequest(limit_price=new_take_profit)
        tp_replaced = trading_client.replace_order_by_id(tp_child.id, tp_update)
        replaced_tp_id = tp_replaced.id

    if sl_child is not None and (new_stop_price is not None or new_stop_limit is not None):
        sl_update = ReplaceOrderRequest(
            stop_price=new_stop_price if new_stop_price is not None else sl_child.stop_price,
            limit_price=new_stop_limit if new_stop_limit is not None else getattr(sl_child, "limit_price", None),
        )
        sl_replaced = trading_client.replace_order_by_id(sl_child.id, sl_update)
        replaced_sl_id = sl_replaced.id

    if replaced_tp_id is None and replaced_sl_id is None:
        raise ValueError("No updates applied. Provide new_take_profit and/or new_stop_price/new_stop_limit.")

    return replaced_tp_id, replaced_sl_id


###this look for the parent order
def update_tp_sl_for_symbol(
    trading_client: TradingClient,
    symbol: str,
    *,
    new_take_profit: Optional[float] = None,
    new_stop_price: Optional[float] = None,
    new_stop_limit: Optional[float] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Update the take-profit and/or stop-loss legs for the active bracket/OCO
    order associated with an open position in `symbol`.

    Args:
        trading_client: An authenticated TradingClient instance.
        symbol: The asset symbol (e.g., "AAPL").
        new_take_profit: New limit price for the take-profit leg (optional).
        new_stop_price: New stop price for the stop-loss leg (optional).
        new_stop_limit: Optional new limit price for a stop-limit SL leg.

    Returns:
        (tp_order_id, sl_order_id): IDs of the replaced legs (None if not updated).

    Raises:
        ValueError: If no open position or no bracket/OCO parent order is found,
                    or if there are no modifiable legs.
    """
    symbol = symbol.upper()

    # 1) Ensure there is an open position
    try:
        pos = trading_client.get_open_position(symbol)
    except Exception:
        raise ValueError(f"No open position found for {symbol}.")

    if float(pos.qty) == 0:
        raise ValueError(f"No open position quantity for {symbol}.")

    # 2) Find the parent order for this symbol (bracket/OCO/OTO) and fetch legs
    # First, look for an open parent order directly.
    parent: Optional[Order] = None

    orders = trading_client.get_orders(
        GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol],
        )
    )

    # Try to find an order that has legs or is a known multi-leg class
    for o in orders:
        if (o.order_class and o.order_class.lower() in ("bracket", "oco", "oto")):
            # Re-fetch this order by id to ensure legs are populated
            parent = trading_client.get_order_by_id(o.id)
            if getattr(parent, "legs", None):
                break

    # If not found, try discovering via children that reference a parent
    if parent is None:
        # Pull all open orders for the symbol and see if any have parent_order_id
        for o in orders:
            if getattr(o, "parent_order_id", None):
                parent = trading_client.get_order_by_id(o.parent_order_id)
                # Ensure legs are populated
                parent = trading_client.get_order_by_id(parent.id)
                if getattr(parent, "legs", None):
                    break

    if parent is None or not getattr(parent, "legs", None):
        raise ValueError(
            f"Could not locate a bracket/OCO parent order with legs for {symbol}."
        )

    # 3) Identify TP and SL legs
    tp_leg: Optional[Order] = None
    sl_leg: Optional[Order] = None

    for leg in parent.legs:
        # Take-profit legs are LIMIT orders (sell for long / buy for short)
        if leg.type == OrderType.LIMIT:
            tp_leg = leg
        # Stop-loss legs have a stop price (type STOP or STOP_LIMIT)
        if leg.stop_price is not None:
            sl_leg = leg

    if tp_leg is None and sl_leg is None:
        raise ValueError("No modifiable TP/SL legs found on the parent order.")

    # 4) Replace legs with new prices
    replaced_tp_id: Optional[str] = None
    replaced_sl_id: Optional[str] = None

    if tp_leg is not None and new_take_profit is not None:
        tp_update = ReplaceOrderRequest(limit_price=new_take_profit)
        tp_replaced = trading_client.replace_order_by_id(tp_leg.id, tp_update)
        replaced_tp_id = tp_replaced.id

    if sl_leg is not None and (new_stop_price is not None or new_stop_limit is not None):
        sl_update = ReplaceOrderRequest(
            stop_price=new_stop_price if new_stop_price is not None else sl_leg.stop_price,
            limit_price=new_stop_limit if new_stop_limit is not None else getattr(sl_leg, "limit_price", None),
        )
        sl_replaced = trading_client.replace_order_by_id(sl_leg.id, sl_update)
        replaced_sl_id = sl_replaced.id

    if replaced_tp_id is None and replaced_sl_id is None:
        raise ValueError("No updates were applied (provide new_take_profit and/or new_stop_price).")

    return replaced_tp_id, replaced_sl_id

###
def find_and_modify_sl_tp_by_symbol(symbol: str, new_stop_loss: float, new_take_profit: float):
    """
    Finds the parent order for a symbol and modifies its stop-loss and take-profit.

    Args:
        symbol (str): The Crypto symbol of the order to find and modify.
        new_stop_loss (float): The new stop-loss price to set.
        new_take_profit (float): The new take-profit price to set.

    Returns:
        Order: The updated order object if successful, otherwise None.
    """
    print(f"\nAttempting to find and modify parent order for {symbol}...")

    # 1. Find all open orders for the symbol
    open_orders = get_all_open_orders_by_symbol(symbol)

    # 2. Check if any open orders were found
    if not open_orders:
        print(f"  - No open orders found for {symbol}. Cannot modify.")
        return None

    # 3. Find the PARENT order. A parent bracket order has associated legs.
    #    A simple order that can be modified into a bracket won't be an exit order.
    parent_order = None
    for order in open_orders:
        if order.legs:  # This identifies a parent bracket order
            parent_order = order
            break

    # If no parent with legs is found, find a simple order that is not an exit leg
    if not parent_order:
        potential_parents = [
            o for o in open_orders
            if o.position_intent not in [PositionIntent.SELL_TO_CLOSE, PositionIntent.BUY_TO_CLOSE]
        ]
        if potential_parents:
            parent_order = potential_parents[0]

    if not parent_order:
        print(f"  - Could not determine a modifiable parent order for {symbol} from the open orders.")
        return None

    order_id = parent_order.id
    print(f"  - Found parent/main order {order_id} to modify.")

    # 4. Call the modification function with the correct parent order ID and new prices
    return modify_or_add_sl_tp(
        order_id=order_id,
        stop_loss_price=new_stop_loss,
        take_profit_price=new_take_profit
    )


###
def modify_or_add_sl_tp(order_id: str, stop_loss_price: float = None, take_profit_price: float = None):
    """
    Modifies an existing order to update or add stop-loss and/or take-profit levels.

    This function replaces the specified order with a new one that includes the
    desired stop-loss and take-profit prices, turning it into a bracket order.

    Args:
        order_id (str): The ID of the order to modify.
        stop_loss_price (float, optional): The new stop-loss price. Defaults to None.
        take_profit_price (float, optional): The new take-profit price. Defaults to None.

    Returns:
        Order: The new, updated order object if successful, otherwise None.
    """
    if not stop_loss_price and not take_profit_price:
        print("Error: You must provide a new stop_loss_price or take_profit_price.")
        return None

    print(f"\nAttempting to modify order {order_id}...")

    try:
        # --- ADDED: Check current order status before modifying ---
        existing_order = trading_client.get_order_by_id(order_id)
        if existing_order.status not in ['new', 'accepted', 'partially_filled']:
            print(f"Cannot modify order {order_id}. Its status is '{existing_order.status}'.")
            return None
        # --- END OF ADDED CODE ---

        print(f"  - New Stop Loss: {stop_loss_price}")
        print(f"  - New Take Profit: {take_profit_price}")

        # Prepare the request objects for stop loss and take profit if prices are provided
        stop_loss_req = StopLossRequest(stop_price=stop_loss_price) if stop_loss_price else None
        take_profit_req = TakeProfitRequest(limit_price=take_profit_price) if take_profit_price else None

        # Create the replacement order request object.
        # Providing stop_loss and take_profit legs will convert the order to a bracket order.
        replace_order_data = ReplaceOrderRequest(
            stop_loss=stop_loss_req,
            take_profit=take_profit_req
        )

        # Replace the order using the correct method name: replace_order_by_id
        new_order = trading_client.replace_order_by_id(
            order_id=order_id,
            order_data=replace_order_data
        )

        print(f"Successfully replaced order {order_id} with new order {new_order.id}.")
        print(f"  - New Stop Loss: {new_order.stop_loss}")
        print(f"  - New Take Profit: {new_order.take_profit}")
        return new_order

    except APIError as e:
        # Catch specific Alpaca API errors for better debugging
        print(f"Alpaca API Error modifying order {order_id}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while modifying order {order_id}: {e}")
        return None

def add_stop_loss_to_order(order_id: str, stop_loss_price: float):
    """
    Adds a stop-loss to an existing simple order, turning it into an OTO (One-Triggers-Other) order.

    Args:
        order_id (str): The ID of the simple order to add a stop-loss to.
        stop_loss_price (float): The stop-loss price.

    Returns:
        Order: The new, updated order object if successful, otherwise None.
    """
    print(f"\nAttempting to add stop loss to order {order_id}...")

    try:
        # --- ADDED: Check current order status before modifying ---
        existing_order = trading_client.get_order_by_id(order_id)
        if existing_order.status not in ['new', 'accepted', 'partially_filled']:
            print(f"Cannot add stop loss to order {order_id}. Its status is '{existing_order.status}'.")
            return None
        # --- END OF ADDED CODE ---

        print(f"  - Stop Loss Price: {stop_loss_price}")

        # Prepare the request object for the stop loss
        stop_loss_req = StopLossRequest(stop_price=stop_loss_price)

        # Create the replacement order request object.
        # Providing only a stop_loss leg will convert the order to an OTO order.
        replace_order_data = ReplaceOrderRequest(
            stop_loss=stop_loss_req
        )

        # Replace the order
        new_order = trading_client.replace_order_by_id(
            order_id=order_id,
            order_data=replace_order_data
        )

        print(f"Successfully added stop loss to order {order_id}. New order ID: {new_order.id}.")
        print(f"  - New Stop Loss: {new_order.stop_loss}")
        return new_order

    except APIError as e:
        # Catch specific Alpaca API errors for better debugging
        print(f"Alpaca API Error adding stop loss to order {order_id}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while adding stop loss to order {order_id}: {e}")
        return None

def add_take_profit_to_order(order_id: str, take_profit_price: float):
    """
    Adds a take-profit to an existing simple order, turning it into an OTO (One-Triggers-Other) order.

    Args:
        order_id (str): The ID of the simple order to add a take-profit to.autom
        take_profit_price (float): The take-profit limit price.

    Returns:
        Order: The new, updated order object if successful, otherwise None.
    """
    print(f"\nAttempting to add take profit to order {order_id}...")

    try:
        # --- ADDED: Check current order status before modifying ---
        existing_order = trading_client.get_order_by_id(order_id)
        if existing_order.status not in ['new', 'accepted', 'partially_filled']:
            print(f"Cannot add take profit to order {order_id}. Its status is '{existing_order.status}'.")
            return None
        # --- END OF ADDED CODE ---

        print(f"  - Take Profit Price: {take_profit_price}")

        # Prepare the request object for the take profit
        take_profit_req = TakeProfitRequest(limit_price=take_profit_price)

        # Create the replacement order request object.
        # Providing only a take_profit leg will convert the order to an OTO order.
        replace_order_data = ReplaceOrderRequest(
            take_profit=take_profit_req
        )

        # Replace the order
        new_order = trading_client.replace_order_by_id(
            order_id=order_id,
            order_data=replace_order_data
        )

        print(f"Successfully added take profit to order {order_id}. New order ID: {new_order.id}.")
        print(f"  - New Take Profit: {new_order.take_profit}")
        return new_order

    except APIError as e:
        # Catch specific Alpaca API errors for better debugging
        print(f"Alpaca API Error adding take profit to order {order_id}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while adding take profit to order {order_id}: {e}")
        return None



# --- RESET PAPER ACCOUNT ------------------------------------------------------
####I have not applied this
# The specific endpoint for resetting the account
ACCOUNT_URL = f"{cfg.ALPACA_BASE_URL}/v2/account"
def reset_paper_account_definitively():
    """
    Resets the Alpaca paper trading account using a direct API call.
    This method is independent of SDK changes.
    """
    if not API_KEY or not SECRET_KEY:
        print("Error: API keys not found in environment variables.")
        print("Please set 'APCA_API_KEY_ID' and 'APCA_API_SECRET_KEY'.")
        return

    print("Attempting to reset your paper trading account...")
    print("WARNING: This will liquidate all positions and clear all activity. This is IRREVERSIBLE.")

    confirm = input("Type 'RESET' to confirm this action: ")

    if confirm != "RESET":
        print("Confirmation failed. Account reset has been cancelled.")
        return

    try:
        print("\nSending reset request to Alpaca API...")
        # Make the direct DELETE request to the API endpoint
        response = requests.delete(ACCOUNT_URL, headers=HEADERS)

        # This will raise an exception if the request failed (e.g., for bad keys)
        response.raise_for_status()

        # If the request was successful, process the response
        account_info = response.json()

        print("SUCCESS: Account has been reset.")
        print(f"  Account ID: {account_info.get('id')}")
        print("  Your account equity and buying power have been restored to the default value.")

    except requests.exceptions.HTTPError as http_err:
        print(f"\nAn HTTP error occurred: {http_err}")
        print(f"Status Code: {http_err.response.status_code}")
        print(f"Response Body: {http_err.response.text}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
