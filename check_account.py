import os
from alpaca.trading.client import TradingClient
import config as cfg # Your config file with keys and PAPER setting

# --- Initialize Clients (same as your main script) ---
API_KEY = os.environ.get('APCA_API_KEY_ID', cfg.ALPACA_API_KEY_ID)
SECRET_KEY = os.environ.get('APCA_API_SECRET_KEY', cfg.ALPACA_SECRET_KEY)
PAPER = cfg.PAPER
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)

# --- The Sanity Check ---
print("--- Checking API Connection Details ---")
try:
    account = trading_client.get_account()
    print(f"✅ Connection Successful!")
    print(f"   - Account Number: {account.account_number}")
    print(f"   - Is Paper Trading: {account.paper_trading}")
    print(f"   - Status: {account.status}")
    print(f"   - Buying Power: {account.buying_power}")

    print("\n--- Verifying Positions from THIS Account ---")
    positions = trading_client.get_all_positions()
    if not positions:
        print("   - No open positions found in this account.")
    else:
        symbols = [p.symbol for p in positions]
        print(f"   - Positions found in this account: {symbols}")

except Exception as e:
    print(f"❌ ERROR: Could not connect or get account details: {e}")