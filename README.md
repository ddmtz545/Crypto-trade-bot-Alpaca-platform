# Crypto-trade-bot-Alpaca-platform
This is bot for trading crypto on Alpaca platform. It is based on a combination of technical analysis indicators as input of a LSTM neural network and signal processing  trend search and verification. You can modify the default and add your technical indicators choice in config.py.
Disclaimer: This code is for experimental backtesting on Alpaca paper account.

For a basic use you need to provide the following info and run the following scripts:
1. Enter your Alpaca API_KEY and SECRET_KEY in config.py
2. Set your favorite crypto ticker in list_tickers.py
3. Run bot_master_run.py; the default time frame is 5m for downloading historical data. It gets the buy and sell signals and then submits the buy and sell orders.
4. As a second layer of profit protection, set the stop loss and take profit values in trading_main_sdk_profit_checker.py and run it. It checks all of the open positions.
5. Copy all folders available in the main or create folders with the same name in your scripts directory.
