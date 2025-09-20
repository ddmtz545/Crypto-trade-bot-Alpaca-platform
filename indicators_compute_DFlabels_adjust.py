
import pandas as pd
import numpy as np
import pandas_ta as ta # This should now import without error
import config as cfg
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical import StockHistoricalDataClient



##this script calculates all of the market indicators

def indicators_compute_DFlabels_adjust(df,ticker_symbol):
#####seperating columns my way
    High_prices_imab = df['high'].values
    High_prices_imab2 = pd.Series(np.squeeze(High_prices_imab), name='high')
    df_High = pd.DataFrame(High_prices_imab2)
    # print('this df_High:   \n ',df_High)

    Low_prices_imab = df['low'].values
    Low_prices_imab2 = pd.Series(np.squeeze(Low_prices_imab), name='low')
    df_Low = pd.DataFrame(Low_prices_imab2)
    # print('this df_Low:   \n ',df_Low)

    Volume_prices_imab = df['volume'].values
    Volume_prices_imab2 = pd.Series(np.squeeze(Volume_prices_imab), name='volume')
    df_Volume = pd.DataFrame(Volume_prices_imab2)
    # print('this df_Volume:   \n ',df_Volume)

    close_prices_imab = df['close'].values
    close_prices_imab2 = pd.Series(np.squeeze(close_prices_imab), name='close')
    # date_prices_imab = df['Price'].values
     # Create a DataFrame from the Series
    df2 = pd.DataFrame(close_prices_imab2)
    # print('this df2:   \n ',df2)
    #new data frame for theoutput
    new_df = {} # To store the separated Series


    # --- Feature Engineering: Add Technical Indicators --------------------
    # FIX: Add a check for sufficient data length#added by Gemini and Majid 20/07/2025
    # The default slow period for MACD is 26. We check for a bit more to be safe.

    # Place this right before your "if len(df2) > 26:" line
    # print(f"Checking {ticker_symbol} for null values: {df2['Close'].isnull().sum()}")
    # print(df2['Close'])
    if len(df2) > cfg.DATA_POINT_LIM:

        ####A Simple Moving Average (SMA) calculates the average price of a security over a specific number of periods.
        SMA10=ta.sma(df2['close'], length=10)
        # print('this is SMA10: \n  ',SMA10)
        new_df['SMA_10'] = SMA10

        SMA5=ta.sma(df2['close'], length=5)
        new_df['SMA_5'] = SMA5

        ####An Exponential Moving Average (EMA) is similar to an SMA in that it calculates an average price over a period.
        ###### However, the EMA gives more weight to recent prices.
        new_df['EMA_20'] = ta.ema(df2['close'], length=20)
        new_df['EMA_50'] = ta.ema(df2['close'], length=50)
        new_df['EMA_10'] = ta.ema(df2['close'], length=10)

        #####The Relative Strength Index (RSI) is a momentum oscillator developed by J. Welles Wilder Wilder Jr.
        ######It measures the speed and change of price movements, indicating whether a stock is potentially overbought or oversold.
        new_df['RSI_14'] = ta.rsi(df2['close'], length=14)
        new_df['RSI_7'] = ta.rsi(df2['close'], length=7)

        ####MACD_12_26_9: This is the MACD Line.
        ####MACDh_12_26_9: This is the MACD Histogram (the 'h' stands for histogram).
        ####MACDs_12_26_9: This is the Signal Line (the 's' stands for signal).
        macd = ta.macd(df2['close'])
        # print('this is MACD: \n  ',macd)
        new_df['MACD'] = macd['MACD_12_26_9']
        new_df['MACDh'] = macd['MACDh_12_26_9']
        new_df['MACDs'] = macd['MACDs_12_26_9']

        ############################## 1. Trend-Following Indicators:
        ####Aroon Indicator: Identifies trend direction and strength. Consists of Aroon Up and Aroon Down lines.
        aroon_df = ta.aroon(df_High['high'], df_Low['low'], length=14)
        aroon_df = aroon_df.fillna(0).astype('int64')
        #######DataFrame with AROOND_14 (Down), AROONU_14 (Up), AROONOSC_14 (Oscillator) columns.
        ########columns in aroon_df AROOND_14  AROONU_14  AROONOSC_14
        # print('this is aroon_df: \n  ',aroon_df)
        new_df['AROONOSC_14'] = aroon_df['AROONOSC_14']
        new_df['AROOND_14'] = aroon_df['AROOND_14']
        new_df['AROONU_14'] = aroon_df['AROONU_14']

        aroon_df_7 = ta.aroon(df_High['high'], df_Low['low'], length=7)
        aroon_df_7 = aroon_df_7.fillna(0).astype('int64')
        #######DataFrame with AROOND_14 (Down), AROONU_14 (Up), AROONOSC_14 (Oscillator) columns.
        ########columns in aroon_df AROOND_14  AROONU_14  AROONOSC_14
        # print('this is aroon_df: \n  ',aroon_df_7)
        new_df['AROONOSC_7'] = aroon_df_7['AROONOSC_7']
        new_df['AROOND_7'] = aroon_df_7['AROOND_7']
        new_df['AROONU_7'] = aroon_df_7['AROONU_7']

        ############################## 2. Momentum Indicators:
        ##Williams %R (Larry Williams' %R): Similar to Stochastic Oscillator, measures
        WilliamsR = ta.willr(df_High['high'], df_Low['low'], df2['close'], length=14)
        WilliamsR = WilliamsR.fillna(0).astype('int64')
        # print('this is WilliamsR: \n  ',WilliamsR)
        new_df['WILLR_14'] = WilliamsR

        WilliamsR7 = ta.willr(df_High['high'], df_Low['low'], df2['close'], length=7)
        WilliamsR7 = WilliamsR7.fillna(0).astype('int64')
        # print('this is WilliamsR7: \n  ',WilliamsR7)
        new_df['WILLR_7'] = WilliamsR7

        #####CCI (Commodity Channel Index): Measures the variation of a security's price from its
        CCI_20 = ta.cci(df_High['high'], df_Low['low'], df2['close'], length=20)
        CCI_20 = CCI_20.fillna(0).astype('int64')
        # print('this is CCI_20: \n  ',CCI_20)
        new_df['CCI_20'] = CCI_20

        CCI_10 = ta.cci(df_High['high'], df_Low['low'], df2['close'], length=10)
        CCI_10 = CCI_10.fillna(0).astype('int64')
        # print('this is CCI_10: \n  ',CCI_10)
        new_df['CCI_10'] = CCI_10


        ###ROC (Rate of Change): Measures the percentage change in price between the current
        ROC_10= ta.roc(df2['close'], length=10)
        ROC_10 = ROC_10.fillna(0).astype('int64')
        # print('this is ROC_10: \n  ',ROC_10)
        new_df['ROC_10'] = ROC_10

        ROC_5= ta.roc(df2['close'], length=5)
        ROC_5 = ROC_5.fillna(0).astype('int64')
        # print('this is ROC_5: \n  ',ROC_5)
        new_df['ROC_5'] = ROC_5

        ############################## 3. Volatility Indicators:
        # ATR (Average True Range): Measures market volatility:
        ATR_14 = ta.atr(df_High['high'], df_Low['low'], df2['close'], length=14)
        ATR_14 = ATR_14.fillna(0).astype('int64')
        # print('this is ATR_14: \n  ',ATR_14)
        new_df['ATR_14'] = ATR_14

        ATR_7 = ta.atr(df_High['high'], df_Low['low'], df2['close'], length=7)
        ATR_7 = ATR_7.fillna(0).astype('int64')
        # print('this is ATR_7: \n  ',ATR_7)
        new_df['ATR_7'] = ATR_7

        ############################## 4. Volume-Based Indicators:
        # add Money felow index:
        mfi = ta.mfi(df_High['high'], df_Low['low'], df2['close'], df_Volume['volume'], length=14).fillna(0).astype('int64')
        ####convert float64 to int64
        # mfi = mfi.fillna(0).astype('int64')
        # print('this is mfi: \n  ',mfi)
        new_df['MFI_14'] = mfi

        mfi_7 = ta.mfi(df_High['high'], df_Low['low'], df2['close'], df_Volume['volume'], length=7).fillna(0).astype('int64')
        ####convert float64 to int64
        # mfi = mfi.fillna(0).astype('int64')
        # print('this is mfi: \n  ',mfi)
        new_df['MFI_7'] = mfi_7


        # add OBV (On-Balance Volume): Uses volume flow to predict changes in stock price.:
        OBV = ta.obv(df2['close'], df_Volume['volume'])
        ####convert float64 to int64
        OBV = OBV.fillna(0).astype('int64')
        # print('this is OBV: \n  ', OBV)
        new_df['OBV'] = OBV

    else:
    # This block will run if there isn't enough data #added by Gemini and Majid 20/07/2025
        print(f"Insufficient data for {ticker_symbol}: need > {cfg.DATA_POINT_LIM} rows, but found {len(df2)}. Skipping.")
        return None
    ###############################################################################


    # Create a DataFrame
    new_df1 = pd.DataFrame(new_df)
    # print('new_df after adding SMA_10,EMA_20,RSI,MACD:  \n',new_df1)
    # Drop rows with NaN values (due to indicator calculation at the start)
    df.dropna(inplace=True)


    #####concatenate df(with integer index) and combined_df. Majid 01/06/2025
    combined_df=pd.concat([df, new_df1], axis=1)
    # print('this is combined_df      : \n  ',combined_df)


    # Drop rows with NaN values (due to indicator calculation at the start)
    # This line is moved AFTER technical indicators are added
    combined_df.dropna(inplace=True)

    df=combined_df

    # print(list(df.columns))
    ####[('Date', ''), ('Close', 'IMAB'), ('High', 'IMAB'), ('Low', 'IMAB'), ('Open', 'IMAB'), ('Volume', 'IMAB'), 'SMA_10', 'EMA_20', 'RSI', 'MACD', 'MACDh']
    ######renaming columns to match the features name
    # Rename column 'A' to 'Alpha' and 'C' to 'Gamma'
    # df_renamed = df.rename(columns={'A': 'Alpha', 'C': 'Gamma'})
    # df.rename() returns a new DataFrame by default.
    # To modify the original DataFrame in place, use inplace=True:
    # df.rename(columns={'A': 'Alpha', 'C': 'Gamma'}, inplace=True)
    # df.rename(columns={'Date': 'Date', 'Open': 'Open','High': 'High','Low': 'Low','Volume': 'Volume','Close': 'Close'}, inplace=True)
    return df

    # print('this is renamed columns combined_df      : \n  ',combined_df)



###these functions check the overall market situation

###note: since you do not need to run below function (is_market_bullish) in every loop,
###use it in config.py to fetch and calculate data once
def is_market_bullish(data_client = StockHistoricalDataClient):
    """
    Checks if the overall market is in a bullish regime.
    Uses SPY's position relative to its 50-day moving average as a proxy.

    Args:
        data_client: The Alpaca StockHistoricalDataClient object.

    Returns:
        bool: True if the market is considered bullish, otherwise False.
    """
    try:
        # Fetch the last 51 days of data for SPY to calculate a 50-day SMA
        spy_bars = data_client.get_stock_bars({
            "symbol_or_symbols": "SPY",
            "timeframe": TimeFrame.Day,
            "limit": 51
        })['SPY'].df

        if spy_bars.empty or len(spy_bars) < 50:
            print("Warning: Could not get enough SPY data for market filter. Defaulting to True.")
            return True # Fail safe to allow trades

        spy_bars['sma_50'] = spy_bars['close'].rolling(window=50).mean()
        last_close = spy_bars['close'].iloc[-1]
        last_sma = spy_bars['sma_50'].iloc[-1]

        if last_close > last_sma:
            print("Market Regime: Bullish (SPY > 50-day SMA)")
            return True
        else:
            print("Market Regime: Bearish (SPY < 50-day SMA)")
            return False

    except Exception as e:
        print(f"Could not determine market regime: {e}. Defaulting to True.")
        return True # Fail safe to allow trades

def has_volume_confirmation(df, volume_period = 20 ):
    """
    Checks if the most recent trading volume is above its moving average.

    Args:
        df (pd.DataFrame): The stock's DataFrame, must contain a 'volume' column.
        volume_period (int): The lookback period for the volume moving average.

    Returns:
        bool: True if the latest volume is above its average, otherwise False.
    """
    if df.empty or len(df) < volume_period:
        return False # Not enough data to confirm

    df['volume_sma'] = df['volume'].rolling(window=volume_period).mean()
    last_volume = df['volume'].iloc[-1]
    avg_volume = df['volume_sma'].iloc[-1]

    return last_volume > avg_volume , df['volume_sma']



