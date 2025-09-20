# Crypto-trade-bot-Alpaca-platform
This is bot for trading crypto on Alpaca platform. It is based on combination of technical analysis indicators as input of a LSTM neural network and signal processing  trend search and verification. 
Disclaimer: This code is for experimental backtesting on Alpaca paper account.

For a basic use you need to provide the following info and run the following scripts:
1. Enter your Alpaca API_KEY and SECRET_KEY in config.py
2. Set your favorite crypto ticker in list_tickers.py
3. run bot_master_run.py, the default time frame is 5m for historical data. It gets the buy and sell signals and then submit the buy and sell orders.
4. As a second layer of profit protection, set the stop loss and take profit values in trading_main_sdk_profit_checker.py and run it. it checks all of the open positions.
