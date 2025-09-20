import subprocess
import sys
import time
# import datetime
from datetime import datetime,timedelta
import trading_advanced_sdk as sdk
from alpaca_pl_calculator import calculate_pnl

def run_script(script_name):
    """Runs a python script and returns its exit code."""
    print(f"\n{'='*25}")
    print(f"▶️  STARTING: {script_name}")
    print(f"{'='*25}")

    process = subprocess.Popen([sys.executable, script_name])
    process.wait() # Wait for the script to complete

    print(f"\n{'='*25}")
    if process.returncode == 0:
        print(f"✅ FINISHED: {script_name} completed successfully.")
    else:
        print(f"❌ ERROR: {script_name} failed with exit code {process.returncode}.")
    print(f"{'='*25}")

    return process.returncode

def main():
    """
    Main function to run the trading bot workflow.
    1. Runs the stock screener.
    2. Runs the trading execution script.
    """
    while True:
        start_time = time.time()
        print("🚀 LAUNCHING AUTOMATED TRADING BOT WORKFLOW 🚀")

        # --- Step 1: Run the stock screener ---
        screener_script = "alpaca_ptpp_model.py"
        # screener_script = "alpaca_ptpp_model_hp.py"
        screener_exit_code = run_script(screener_script)

        ##added this to test Majid 12/09/2025
        if screener_exit_code != 0 :
            Print(f"❌🚀There is a problem with 02alpaca_ptpp_model: check it")
            break
        #------------------------

        time.sleep(1)
        # --- Step 2: Run the trading script only if the screener was successful ---
        if screener_exit_code == 0:
            trading_script = "bot_trading_main.py"
            run_script(trading_script)
        else:
            print("\nSkipping trading script due to errors in the screener.")

        end_time = time.time()
        print(f"\n✨ Workflow Complete. Total execution time: {end_time - start_time:.2f} seconds.")
        print("\Codes run is complete. Waiting for 305 seconds before the next run...")

        ###claculating net profit and loss for master stop loss
        #---------------------------------------------------------------
        # Get the current date and time for the end point
        end_date = datetime.now()
        # Calculate the start date by subtracting one day (24 hours)
        start_date = end_date - timedelta(days=1)

        # Format both dates into the 'YYYY-MM-DD' string format
        # default_end = end_date.strftime('%Y-%m-%d')
        default_end = datetime.now().strftime('%Y-%m-%d')
        default_start = start_date.strftime('%Y-%m-%d')
        total_net_realized,total_unrealized,total_fees,total_pnl= \
        calculate_pnl(default_end,default_end, True)###true is for paper account

        ####Stop Loss threshold to stop the bot
        Loss_threshold = 150
        if total_net_realized < -Loss_threshold:
            print(f"Your Loss is more than threshold {Loss_threshold}, trading bot stopped.")
            sdk.panic_button_crypto()
            break
        ##---------------------------------------

        # Get the current date and time
        current_time = datetime.now()
        print('\n\n Current Time is:',current_time)
        time.sleep(305) # 300 seconds = 5 minutes
        sdk.cancel_all_open_crypto_orders()###there is small bug here , this lines cancels SL and TP limits,
        #only prtotectoin is profit_ckecker program which runs every 1 minute (dangerous 1 minute)

if __name__ == '__main__':

    main()

