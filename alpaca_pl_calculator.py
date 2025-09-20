import os
import argparse
from datetime import datetime, timezone
from collections import defaultdict, deque
import alpaca_trade_api as tradeapi

import config as cfg


# --- Configuration ---
# It's best practice to use environment variables for your API keys.
# For Paper Trading: Use your paper trading keys and the paper trading URL.
# For Live Trading: Use your live trading keys and the live trading URL.
API_KEY = os.environ.get('APCA_API_KEY_ID', cfg.ALPACA_API_KEY_ID)
API_SECRET = os.environ.get('APCA_API_SECRET_KEY', cfg.ALPACA_SECRET_KEY)
BASE_URL = os.environ.get('APCA_API_BASE_URL', cfg.ALPACA_BASE_URL) # Use 'https://api.alpaca.markets' for live trading



def calculate_pnl(start_date, end_date, is_paper_trading,
    api_key_id = API_KEY,
     api_secret_key = API_SECRET,
     base_url = BASE_URL,
     all_data = False
     ):
    """
    Connects to the Alpaca API, fetches historical and current position data,
    and calculates both Realized (with fees) and Unrealized P&L.
    all_data = True fetch all hostorical data from the date of account opening
    """
   

    # --- 1. Get API Keys from Environment Variables ---
    # api_key_id = os.getenv('APCA_API_KEY_ID')
    # api_secret_key = os.getenv('APCA_API_SECRET_KEY')
    # base_url = os.getenv('APCA_API_BASE_URL')

    if not all([api_key_id, api_secret_key, base_url]):
        print("Error: Alpaca API keys or Base URL are not set.")
        return

    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)

    print(f"Connecting to Alpaca ({'Paper' if is_paper_trading else 'Live'} Trading)...")

    # --- 2. Initialize Alpaca API Client ---
    try:
        api = tradeapi.REST(api_key_id, api_secret_key, base_url=base_url, api_version='v2')
        account = api.get_account()
        print(f"Successfully connected to account: {account.account_number}")
    except Exception as e:
        print(f"Error connecting to Alpaca API: {e}")
        return

    # """
    if all_data == True:
        # --- 3. Fetch Full Account History for P&L and Fees ---
        print("Fetching full account trade history for P&L and Fee calculation...")

        all_activities = []
        page_token = None

        ##this while loop get all of the activities from account openning time
        ##I modified it you can set number of pages Majid 06/09/2025
        ##keep direction 'asc'
        page_counter = 0
        while True:
            try:
                batch = api.get_activities(direction='asc', page_size=100, page_token=page_token)#direction='asc' or'desc'
                # page_counter += 1
                # if page_counter < 10:
                if batch:
                    all_activities.extend(batch)
                    page_token = batch[-1].id
                else:
                    break
                # else:
                #     break

            except Exception as e:
                print(f"An error occurred while fetching activities: {e}")
                break
    else:
        # """
        #uncomment above if you want all of the activities Majid 05/09/2025
        #below fetch does not work properly, needs to be investigated
        # --- 3. Fetch Account History for the Specified Period ---
        print(f"Fetching activities from {start_date} to {end_date}...")

        try:
            # Convert datetime objects to ISO 8601 format strings for the API
            start_iso = start_date_dt.isoformat()
            end_iso = end_date_dt.isoformat()

            # Fetch all activities within the date range in a single, efficient call
            all_activities = api.get_activities(
                after=start_iso,
                until=end_iso,
                direction='asc' ## Keep 'asc' to process trades chronologically
            )
            print(f"Found {len(all_activities)} activities in the specified period.")

        except Exception as e:
            print(f"An error occurred while fetching activities: {e}")
            # Return zero values if the fetch fails
            return 0, 0, 0, 0
        # """

    # --- 4. Separate Fills from Fees ---
    fills = []
    fees_by_symbol = defaultdict(float)

    for a in all_activities:
        if a.activity_type == 'FILL':
            fills.append(a)
        # JNLF is a common activity type for fees (Journal Fee).
        # We check for a negative net_amount and a symbol.
        elif a.activity_type == 'JNLF' and hasattr(a, 'net_amount') and hasattr(a, 'symbol') and a.symbol:
            fee_amount = float(a.net_amount)
            # Only count fees that are debits (costs) and occurred in the period
            if fee_amount < 0 and hasattr(a, 'date') and start_date_dt <= a.date.replace(tzinfo=timezone.utc) <= end_date_dt:
                fees_by_symbol[a.symbol] += abs(fee_amount)

    if not fills: print("No trade fills found in the entire account history.")
    else: print(f"Found {len(fills)} total fills and processed fee activities in history.")


    # --- 5. Calculate Realized P&L (Gross) using FIFO ---
    trades_by_symbol = defaultdict(list)
    for fill in fills:
        if fill.transaction_time <= end_date_dt:
            trades_by_symbol[fill.symbol].append(fill)

    # MODIFIED: Store a dictionary of results instead of just a float
    stock_gross_pnl = defaultdict(dict)
    crypto_gross_pnl = defaultdict(dict)

    for symbol, trades in trades_by_symbol.items():
        buy_queue = deque()
        realized_pnl_in_period = 0.0
        cost_basis_in_period = 0.0  # NEW: Variable to track the cost of sold shares

        for trade in trades:
            qty, price = float(trade.qty), float(trade.price)
            if trade.side == 'buy':
                buy_queue.append({'qty': qty, 'price': price})
            elif trade.side == 'sell':
                sell_qty = qty
                while sell_qty > 0 and buy_queue:
                    oldest_buy = buy_queue[0]
                    match_qty = min(sell_qty, oldest_buy['qty'])

                    # Only calculate for sales that happened within the period
                    if start_date_dt <= trade.transaction_time <= end_date_dt:
                        pnl = (price - oldest_buy['price']) * match_qty
                        realized_pnl_in_period += pnl
                        # NEW: Add the cost of the shares being sold to the total cost basis
                        cost_basis_in_period += oldest_buy['price'] * match_qty

                    sell_qty -= match_qty
                    oldest_buy['qty'] -= match_qty
                    if oldest_buy['qty'] == 0:
                        buy_queue.popleft()

        # NEW: Check if there's a cost basis to avoid division by zero
        if cost_basis_in_period > 0:
            # NEW: Calculate the P&L percentage
            pnl_percentage = (realized_pnl_in_period / cost_basis_in_period) * 100

            # NEW: Store all results in a dictionary for later use
            pnl_data = {
                'pnl': realized_pnl_in_period,
                'pnl_pct': pnl_percentage
            }

            if '/' in symbol:
                crypto_gross_pnl[symbol] = pnl_data
            else:
                stock_gross_pnl[symbol] = pnl_data

    # --- 6. Fetch Open Positions for Unrealized P&L ---
    print("\nFetching current open positions for Unrealized P&L calculation...")
    stock_unrealized = {}
    crypto_unrealized = {}
    try:
        positions = api.list_positions()
        print(f"Found {len(positions)} open positions.")
        for pos in positions:
            avg_cost = float(pos.avg_entry_price)
            qty = float(pos.qty)
            total_cost = avg_cost * qty  # NEW: Calculate total cost of the position
            pnl = float(pos.unrealized_pl)

            # NEW: Calculate unrealized P&L percentage, handle division by zero
            pnl_percentage = (pnl / total_cost * 100) if total_cost > 0 else 0

            unrealized_data = {
                'qty': qty,
                'avg_cost': avg_cost,
                'current_price': float(pos.current_price),
                'pnl': pnl,
                'pnl_pct': pnl_percentage # NEW: Add percentage to the dictionary
            }
            if '/' in pos.symbol:
                crypto_unrealized[pos.symbol] = unrealized_data
            else:
                stock_unrealized[pos.symbol] = unrealized_data
    except Exception as e:
        print(f"Could not fetch open positions: {e}")

    # --- 7. Display the Results ---

    # --- REALIZED P&L REPORT ---
    print("\n--- Realized P&L Report (FIFO Method) ---")
    print(f"Period: {start_date} to {end_date}\n")

    total_net_stock_pnl = 0
    if stock_gross_pnl:
        # MODIFIED: Added '% P&L' column and adjusted spacing
        print(f"{'Stock Ticker':<15} {'Gross P&L ($)':>15} {'P&L (%)':>10} {'Fees ($)':>15} {'Net P&L ($)':>15}")
        print("-" * 75)
        for symbol, data in sorted(stock_gross_pnl.items()):
            gross_pnl = data['pnl']
            pnl_pct = data['pnl_pct']
            fee = fees_by_symbol.get(symbol, 0.0)
            net_pnl = gross_pnl - fee
            # MODIFIED: Added pnl_pct to the formatted print string with a '%' sign
            print(f"{symbol:<15} {gross_pnl:>15.2f} {pnl_pct:>9.2f}% {fee:>15.2f} {net_pnl:>15.2f}")
            total_net_stock_pnl += net_pnl
        print("-" * 75)
        print(f"{'Stock Subtotal':<57} {total_net_stock_pnl:>15.2f}\n")
    else:
        print("No realized stock P&L in the specified period.\n")

    total_net_crypto_pnl = 0
    if crypto_gross_pnl:
        # MODIFIED: Added '% P&L' column and adjusted spacing
        print(f"{'Crypto Ticker':<15} {'Gross P&L ($)':>15} {'P&L (%)':>10} {'Fees ($)':>15} {'Net P&L ($)':>15}")
        print("-" * 75)
        #to calculate averages
        sum_pnl_pct=0
        sum_gross_pnl=0
        # counter = 0
        for symbol, data in sorted(crypto_gross_pnl.items()):
            gross_pnl = data['pnl']
            pnl_pct = data['pnl_pct']
            fee = fees_by_symbol.get(symbol, 0.0)
            net_pnl = gross_pnl - fee
            # MODIFIED: Added pnl_pct to the formatted print string with a '%' sign
            #Averages
            # counter += 1
            sum_pnl_pct += pnl_pct
            sum_gross_pnl += gross_pnl
            print(f"{symbol:<15} {gross_pnl:>15.2f} {pnl_pct:>9.2f}% {fee:>15.2f} {net_pnl:>15.2f}")
            total_net_crypto_pnl += net_pnl
        print("-" * 75)
        print(f"{'Crypto Subtotal':<10} {sum_gross_pnl:>15.2f} {sum_pnl_pct:>9.2f}% {total_net_crypto_pnl:>31.2f}\n")
    else:
        print("No realized crypto P&L in the specified period.\n")

    # --- UNREALIZED P&L REPORT ---
    total_stock_unrealized = sum(p['pnl'] for p in stock_unrealized.values())
    total_crypto_unrealized = sum(p['pnl'] for p in crypto_unrealized.values())
    print("\n--- Unrealized P&L Report (Current Holdings) ---")
    if stock_unrealized:
        # MODIFIED: Added 'Unrlzd P&L (%)' column
        print(f"{'Stock Ticker':<15} {'Qty':>10} {'Avg Cost':>12} {'Current':>12} {'Unrlzd P&L (%)':>18} {'Unrealized P&L ($)':>22}")
        print("-" * 95)
        for s, d in sorted(stock_unrealized.items()):
            # MODIFIED: Added pnl_pct to the formatted print string
            print(f"{s:<15} {d['qty']:>10.2f} {d['avg_cost']:>12.2f} {d['current_price']:>12.2f} {d['pnl_pct']:>17.2f}% {d['pnl']:>22.2f}")
        print("-" * 95)
        print(f"{'Stock Subtotal':<71} {total_stock_unrealized:>22.2f}\n")
    else:
        print("No open stock positions found.\n")

    if crypto_unrealized:
        # MODIFIED: Added 'Unrlzd P&L (%)' column
        print(f"{'Crypto Ticker':<15} {'Qty':>10} {'Avg Cost':>12} {'Current':>12} {'Unrlzd P&L (%)':>18} {'Unrealized P&L ($)':>22}")
        print("-" * 95)
        for s, d in sorted(crypto_unrealized.items()):
            # MODIFIED: Added pnl_pct to the formatted print string
            print(f"{s:<15} {d['qty']:>10.4f} {d['avg_cost']:>12.2f} {d['current_price']:>12.2f} {d['pnl_pct']:>17.2f}% {d['pnl']:>22.2f}")
        print("-" * 95)
        print(f"{'Crypto Subtotal':<71} {total_crypto_unrealized:>22.2f}\n")
    else:
        print("No open crypto positions found.\n")

    # --- TOTAL P&L SUMMARY ---
    # (This section remains unchanged)
    total_net_realized = total_net_stock_pnl + total_net_crypto_pnl
    total_unrealized = total_stock_unrealized + total_crypto_unrealized
    total_fees = sum(fees_by_symbol.values())
    total_pnl = total_net_realized + total_unrealized

    print("\n--- Grand Total P&L Summary ---")
    print("=" * 45)
    print(f"{'Total Net Realized P&L':<25} {total_net_realized:>18.2f}")
    print(f"{'Total Unrealized P&L':<25} {total_unrealized:>18.2f}")
    print(f"{'Total Fees In Period':<25} {total_fees:>18.2f}")
    print("-" * 45)
    print(f"{'TOTAL P&L':<25} {total_pnl:>18.2f}")
    print("=" * 45)

    return total_net_realized,total_unrealized,total_fees,total_pnl


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate Realized & Unrealized P&L from Alpaca trading data.')
    default_end = datetime.now().strftime('%Y-%m-%d')
    parser.add_argument('--start', dest='start_date', required=False, help='Start date in YYYY-MM-DD format.')
    parser.add_argument('--end', dest='end_date', default=default_end, help=f'End date in YYYY-MM-DD format (default: today, {default_end}).')
    parser.add_argument('--live', action='store_false', dest='is_paper', default=True, help='Use the live trading environment instead of paper. Defaults to paper.')
    args = parser.parse_args()
    if args.start_date:
        calculate_pnl(args.start_date, args.end_date, args.is_paper)
    else:
        print("No --start argument provided. Running with a hardcoded example.")
        # calculate_pnl(default_end, default_end, True)
        calculate_pnl("2025-09-10", default_end, True, all_data = False)
