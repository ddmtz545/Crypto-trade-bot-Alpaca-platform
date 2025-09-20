# config.py
from indicators_compute_DFlabels_adjust import is_market_bullish

# --- ALPACA API CREDENTIALS ---
##Paper account
ALPACA_API_KEY_ID = ""#paper account# Use os.getenv("ALPACA_API_KEY_ID") for better security
ALPACA_SECRET_KEY = ""#paper account# Use os.getenv("ALPACA_SECRET_KEY") for better security
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"# For paper trading
PAPER = True ##this if for sdk trading module
##live account
# ALPACA_API_KEY_ID = ""#live account
# ALPACA_SECRET_KEY = ""#live account
# ALPACA_BASE_URL = "https://api.alpaca.markets"#live account
# PAPER = False

# FOLDER_DAILY_DATA = "daily_data_cache" # For storing downloaded daily data#no code written for it
# --- FILE/FOLDER PATHS -----------------------------------------

FOLDER_PLOTS_SAVED = 'alp_plots_saved'
#----------models

# ##two below lines should be commented and uncommented together
FOLDER_MODELS_SAVED = 'alp_models_saved'
CNN_LSTM_CREATE_SWITCH = False

# ##two below lines should be commented and uncommented together
# CNN_LSTM_CREATE_SWITCH = True
# FOLDER_MODELS_SAVED = 'alp_cnn_lstm_models_saved'###some times cnn not good for predicting the end of trend well

#-------------------
FOLDER_TRADE_SIGNAL_SAVED = 'alp_trade_signal_saved'
BOT_FOLDER_TRADE_SIGNAL_SAVED = 'bot_alp_trade_signal_saved'
FOLDER_MARKET_BARS_SAVED = 'market_bars_saved'
# ------------------- MODEL & TRAINING PARAMETERS ---------------------------
ASSET_TYPE = "CRYPTO"# "STOCK" # "CRYPTO"#
SLEEP_TIME = 0 #set 0 for paid membership #0.5 ##this is time.sleep() value to avoid Alpaca API rate limit (200 requests per minute on the free plan).
LOOK_BACK = 60
DATA_POINT_LIM = 50 ###minimum number of data points in df2['close'] to calculate the indicators
##download start_date
SD_YEAR = 2020
SD_MONTH = 1
SD_DAY = 1

BUFFER_SECONDS = 0#set to 0 for alpaca paid membership # 1000-second buffer
# data_download_intervals=[
#     "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
#     "1d", "5d", "1wk", "1mo", "3mo"]
TIME_INTERVAL_DAILY = "1d"
#below variables are for lower timeframes in minutes or hourly
TIME_INTERVAL_MINUTE = "5m" # "5m" #"1m" #"5m"#'2m'
TIME_CONV_RATIO = 1#12#1#12#30 ###converting test X axis from minutes to hours (30 is for 2m interval)in 02stpp and other 02 files
WORKING_DAYS_TO_SUBTRACT = 60#18#200#18 #7     ###number of working days to download data in minutes range
#this sets the volume period for has_volume_confirmation function used in plotter functions
VOLUME_PERIOD_MVA_DAILY = 7 #20 #50  #20-50 is standard for daily and 10-20 is standard for 5m time frame
VOLUME_PERIOD_MVA_MINUTE = 2 #156 ##78 sampls is one day form 5m tf([:79])
# --- FEATURE SETS -------------------------------------------------------------
# Define all available feature combinations here
##'set_01_buy_focus''set_02_sell_focus''set_03_less_features'set_04_predicts_well''set_05_all_features'

FEATURE_SETS = {
    'fset_01': ['volume', 'open','close','vwap','trade_count','RSI_14','ATR_14', 'MFI_14','WILLR_14'],
    'fset_02': ['volume', 'close','vwap','trade_count','RSI_14', 'MACD','MACDs','ATR_14', 'MFI_14','WILLR_14','ROC_10','AROONOSC_14'],
    'fset_03': ['close','vwap','trade_count','RSI_14','ATR_14', 'MFI_14','WILLR_14'],
    'fset_04': ['volume', 'open','close','vwap','trade_count', 'RSI_14','MACDh','ATR_14', 'MFI_14','WILLR_14'],
    'fset_05': ['open', 'high', 'low', 'volume', 'close','vwap','trade_count', 'SMA_10', 'EMA_20', 'RSI_14', 'MACD', 'MACDh','MACDs','MFI_14','OBV','ATR_14','WILLR_14','CCI_20','ROC_10','AROONOSC_14'],
    'fset_m01': ['open', 'high', 'low', 'volume', 'close','vwap','trade_count', 'SMA_5', 'EMA_10', 'RSI_7','MFI_7','ATR_7','WILLR_7','CCI_10','ROC_5'],
    'fset_m05': ['open', 'high', 'low', 'volume', 'close','vwap','trade_count', 'SMA_10', 'EMA_20', 'RSI_14','MFI_14','ATR_14','WILLR_14','CCI_20','ROC_10'],
    'fset_d05': ['open', 'high', 'low', 'volume', 'close','vwap','trade_count', 'EMA_50','AROONOSC_14','OBV','MACD','MACDh','MACDs'],
    'fset_d05_EMA_20': ['open', 'high', 'low', 'volume', 'close','vwap','trade_count', 'EMA_20','AROONOSC_14','OBV','MACD','MACDh','MACDs'],

    # ...add all other sets here #models should be trained for 'fset_d05_EMA_20'
}

###assign training data size ratio
###training size can be a list of numbers
TRAIN_SIZE_RATIOS_DAILY = [0.9]#[0.95]#[0.9]#[0.95]#[0.9]
# TRAIN_SIZE_RATIOS_DAILY = [0.8,0.9]

TRAIN_SIZE_RATIOS_MINUTE =[0.85]#[0.9]#
# Select which feature sets to run
###select indicator sets as model input

SELECTED_FEATURE_SETS =['fset_d05']#['fset_d05_EMA_20']#['fset_d05']#['fset_d05_EMA_20']#['fset_d05']# ['fset_d05_EMA_20']#['fset_d05']

SELECTED_FEATURE_SETS_MIN = ['fset_m05']#['fset_m05']###this is for minute timeframe

#------------------------------------------------------------------------------

#--- Market search and signals
#find_best_trend_signal_advanced function look back days
BEST_SIGNAL_LOOKBACK_DAYS = 1 #has no been used yet

# --- CONTROL FLAGS -----------------------------
# --- Model Control ---
# Control variable to enable or disable a specific feature/code block
# Set to True to ENABLE the code, False to DISABLE it.
ENABLE_MODEL_SAVE = True
# ENABLE_MODEL_SAVE = False

#------------save figure
#enable saveing figures for predictions
# ENABLE_SAVE_PREDICTION_FIGURE = True
ENABLE_SAVE_PREDICTION_FIGURE = False

#ENTRY_EXIT_SIMPLE_SERIES_PLOT below swith (feature flag) is sependent to above ENABLE_SAVE_PREDICTION_FIGURE
ENTRY_EXIT_SIMPLE_SERIES_PLOT = True
# ENTRY_EXIT_SIMPLE_SERIES_PLOT = False
#-------------

## For model creating if it can not be loaded
ENABLE_ADVANCED_ANALYSIS = True #True # False #
# ENABLE_ADVANCED_ANALYSIS = False

# --- Trading Control (VERY IMPORTANT FOR LIVE TRADING) ---
ENABLE_LIVE_TRADING = False # Set to True ONLY for paper/live trading
# ... other trading parameters like POSITION_SIZE, RISK_PER_TRADE

#----------market overall-------------------------
BULISH_CONFIRMED = is_market_bullish()###checks SPY moving Average
