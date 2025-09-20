# Crypto-trade-bot-Alpaca-platform
This is bot for trading crypto on Alpaca platform. It is based on combination of technical analysis indicators as input of a LSTM neural network and signal processing  trend finding. This code is for experimantal backtesting for Alpaca paper account.

For a basic use you need to provide the following info and run the following codes:
1. Enter your Alpaca API_KEY and SECRET_KEY in config.py
2. Set your favorite crypto ticker in list_tickers.py
3. run bot_master_run.py
4. As a second layer of profit protection set the stop loss and take profit values in trading_main_sdk_profit_checker.py and run it. it checks all of the open positions.
