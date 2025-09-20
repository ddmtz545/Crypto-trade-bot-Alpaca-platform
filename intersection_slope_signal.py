import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta as ta
from indicators_compute_DFlabels_adjust import has_volume_confirmation
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler
from crypto_bulish_indicator import is_crypto_market_bullish
###Gemini suggestion 29/07/2025


def find_sharp_turning_points(df, price_column, new_peak_col, new_trough_col,
    distance=15,###it seems 15 works better than 20 for crypto
    prominence=0.04,
    height = None,  # #0.04 is better than 0.03 Peak must stand out by at least 15% of the total price range
    width= None  # Peak must be less than 20 data points wide
    ):
    """
    Finds sharp peaks and troughs in a normalized price column of a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        price_column (str): The name of the normalized price column to analyze.
        new_peak_col (str): The name for the new boolean column for peaks.
        new_trough_col (str): The name for the new boolean column for troughs.
        prominence (float): The required prominence of peaks. Normalized scale (0-1).
        width (tuple): The required width range of peaks. A smaller max width finds sharper peaks.

    Returns:
        pd.DataFrame: The DataFrame with added boolean columns for peaks and troughs.
    """
    # Find sharp peaks (local maxima)
    # A smaller max width ensures we find sharper, less-rounded peaks.
    peak_indices, _ = find_peaks(df[price_column],height=height, distance=distance, prominence=prominence)#, width=width)

    # Find sharp troughs by inverting the series
    trough_indices, _ = find_peaks(-df[price_column],height=height, distance=distance, prominence=prominence)#, width=width)

    # Initialize new columns with False
    df[new_peak_col] = False
    df[new_trough_col] = False

    # Use .loc with the original index to set the turning point locations to True
    # This correctly handles non-sequential or datetime indices
    df.loc[df.index[peak_indices], new_peak_col] = True
    df.loc[df.index[trough_indices], new_trough_col] = True

    return df

def analyze_turning_point_stock_prices(analysis_df):
    """
    Normalizes price data and finds turning points for both real and predicted prices.

    Args:
        analysis_df (pd.DataFrame): DataFrame with 'Real_Price' and 'Predicted_Price' columns.

    Returns:
        pd.DataFrame: The analyzed DataFrame with normalization and turning point columns.
    """
    # --- 1. Normalize Both Price Columns ---
    # Normalization is crucial so that prominence and other parameters are relative
    # and work consistently across stocks with different price ranges.
    scaler = MinMaxScaler()
    analysis_df[['Real_Price_Norm', 'Predicted_Price_Norm']] = scaler.fit_transform(
        analysis_df[['Real_Price', 'Predicted_Price']]
        # analysis_df[['Real_Price', 'Smoothed_Predicted_Price']]
    )
    # print('Normalized prices; ',analysis_df[['Real_Price_Norm', 'Predicted_Price_Norm']])
    # --- 2. Find Turning Points for Both Normalized Prices ---
    # Find turning points for Predicted Price

##---------------------------Buy Entry------------------------------------------
    ##this finds troughs for ''buy points'' seperately with different parameters values
    analysis_df = find_sharp_turning_points(
        df=analysis_df,
        price_column='Predicted_Price_Norm',
        new_peak_col='is_peak_Predicted_Price',
        new_trough_col='is_trough_Predicted_Price',
        distance=15,###it seems 15 works better than 20 for crypto
        prominence=0.1,
        height = None
    )

    ##finding wider peaks and traughs
    # trough_indices, properties = find_peaks(-price_data, prominence=100, width=10)
    analysis_df = find_sharp_turning_points(
        df=analysis_df,
        price_column='Predicted_Price_Norm',
        new_peak_col='is_peak_very_wide_Predicted_Price',
        new_trough_col='is_trough_very_wide_Predicted_Price',
        distance=26,###it seems 15 works better than 20 for crypto
        prominence=0.5,
        height = None,
        width=20
    )


##----------------------------Sell Entry----------------------------------------
    ##this finds peaks for '''sell points'' seperately with different parameters values
    analysis_df = find_sharp_turning_points(
        df=analysis_df,
        price_column='Predicted_Price_Norm',
        new_peak_col='is_sell_peak_Predicted_Price',
        new_trough_col='is_sell_trough_Predicted_Price',
        distance=20,###it seems 15 works better than 20 for crypto
        prominence=0.11,
        height = 0.25
    )

    ##finding wider peaks and traughs
    analysis_df = find_sharp_turning_points(
        df=analysis_df,
        price_column='Predicted_Price_Norm',
        new_peak_col='is_sell_peak_very_wide_Predicted_Price',
        new_trough_col='is_sell_trough_very_wide_Predicted_Price',
        distance=26,###it seems 15 works better than 20 for crypto
        prominence=0.5,
        height = None,
        width=20
    )


    # print('Predicted_Price_Norm',analysis_df)
    # Find turning points for Real Price
    analysis_df = find_sharp_turning_points(
        df=analysis_df,
        price_column='Real_Price_Norm',
        new_peak_col='is_peak_Real_Price',
        new_trough_col='is_trough_Real_Price',
    )
    # print('Real_Price_Norm',analysis_df)
    return analysis_df

###--------------------end of turning point functions--------------------------
def calculate_slope(series, window=3):
    """Calculates the slope of a pandas Series using a rolling linear regression."""
    # This is a simplified slope calculation. For more accuracy, a library
    # like scipy.stats.linregress could be used on rolling windows.
    return series.diff(periods=1).rolling(window=window).mean()

#this function is to calculate the price gap slope
def calculate_slope_polyfit(series):
    data = abs(series).round(3)
    print(data)
    x_values = np.arange(len(data))
    # Find the line of best fit. polyfit returns [slope, intercept]
    # We only care about the slope, which is the first element.
    slope, intercept = np.polyfit(x_values, data, 1)
    # A positive slope means the overall trend is increasing
    return slope

def calculate_normalized_error(actual_prices: pd.Series,
                               predicted_prices: pd.Series,
                               method: str = 'percent',
                               full_df: pd.DataFrame = None,
                               atr_period: int = 14) -> pd.Series:
    """
    Calculates the normalized difference (error) between actual and predicted prices.

    This allows for meaningful comparison of prediction errors across different stocks.

    Args:
        actual_prices (pd.Series): A pandas Series of the actual stock prices.
        predicted_prices (pd.Series): A pandas Series of the predicted stock prices.
        method (str, optional): The normalization method to use.
                                'percent' -> Percentage Error (default).
                                'atr' -> Normalize by Average True Range (volatility).
        full_df (pd.DataFrame, optional): The full OHLC DataFrame, required only for the 'atr' method.
                                          Must contain 'High', 'Low', and 'Close' columns.
        atr_period (int, optional): The lookback period for calculating ATR. Defaults to 14.

    Returns:
        pd.Series: A pandas Series containing the normalized error.
    """
    # Ensure inputs are aligned by index
    actual_prices, predicted_prices = actual_prices.align(predicted_prices, join='inner')

    # Calculate the raw price difference (the error)
    error = actual_prices - predicted_prices

    if method == 'percent':
        # Calculate error as a percentage of the actual price
        # Multiply by 100 to get a percentage value (e.g., 1.5 for 1.5%)
        normalized_error = (error / actual_prices) * 100
        return normalized_error.fillna(0) # Handle potential division by zero

    elif method == 'atr':
        if full_df is None:
            raise ValueError("The 'full_df' DataFrame is required for the 'atr' method.")

        # Calculate True Range
        high_low = full_df['High'] - full_df['Low']
        high_prev_close = abs(full_df['High'] - full_df['Close'].shift())
        low_prev_close = abs(full_df['Low'] - full_df['Close'].shift())

        tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)

        # Calculate Average True Range (ATR)
        atr = tr.ewm(span=atr_period, adjust=False).mean()

        # Normalize the error by the ATR
        # This tells you how large the error is relative to the stock's typical daily movement
        normalized_error = error / atr
        return normalized_error.fillna(0) # Handle potential division by zero

    else:
        raise ValueError("Invalid method. Choose either 'percent' or 'atr'.")


##----------------------------signal generators---------------------------------
##-----------------------------------------------------------------------------
##-----------------------------------------------------------------------------

def analyze_trends_and_generate_signals(
    Date,
    Real_Price,
    Predicted_Price,
    n_days=300,
    slope_period=3,
    strong_buy_angle=30,
    strong_sell_angle=-30,
    prediction_lag=-1,  # Changed from a hardcoded negative shift
    smoothing_window=10 # Parameter for optional smoothing, it is not good for live trading
):
    """
    Analyzes stock data to generate trading signals using vectorized operations.

    Args:
        Date, Real_Price, Predicted_Price: Input data series.
        n_days (int): The number of recent days to analyze.
        prediction_lag (int): The number of days the prediction was made in the past.
                              A positive value simulates a realistic backtest (no look-ahead bias).
        smoothing_window (int or None): Window for smoothing real prices. Set to None to disable.
        slope_period (int): The window to calculate the rolling slope.
        strong_buy_angle (float): The angle in degrees to qualify a "Strong Buy".
        strong_sell_angle (float): The angle in degrees to qualify a "Strong Sell".

    Returns:
        pd.DataFrame: DataFrame with analysis columns.
    """
    # --- 1. Data Preparation ---
    df = pd.DataFrame({'Date': Date, 'Real_Price': Real_Price, 'Predicted_Price': Predicted_Price})



    df['No_Smooth_Real_Price'] = df['Real_Price']
    # Optional smoothing to reduce noise
    ###note:**************smoothing should be zero for live trading*************
    if smoothing_window:
        df['Smoothed_Real_Price'] = df['Real_Price'].rolling(window=smoothing_window).mean()
        df['Smoothed_Predicted_Price'] = df['Predicted_Price'].rolling(window=smoothing_window).mean()

        # df['Real_Price'] = df['Real_Price'].rolling(window=smoothing_window).mean()
        # df['Predicted_Price'] = df['Predicted_Price'].rolling(window=smoothing_window).mean()

    # Apply lag to simulate a realistic prediction scenario (avoids look-ahead bias)
    df['Predicted_Price'] = df['Predicted_Price'].shift(prediction_lag)
    # df['Real_Price'] = df['Real_Price'].shift(prediction_lag)

    # Handle NaNs created by rolling/shifting
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Use the last n_days for analysis
    last_n_days_df = df.tail(n_days).copy()

    # --- 2. Core Signal Generation (Vectorized) ---
    # --- 1. Calculate Difference and Find Intersections ---
    last_n_days_df['Difference'] = last_n_days_df['Real_Price'] - last_n_days_df['Predicted_Price']
    last_n_days_df['Normalized_Difference'] = calculate_normalized_error(last_n_days_df['Real_Price'],
                                   last_n_days_df['Predicted_Price'],
                                   method = 'percent',
                                   full_df = None,
                                   atr_period = 14
                                   )
    # Find where the sign of the difference changes (an intersection)
    last_n_days_df['Intersection'] = np.sign(last_n_days_df['Difference']).diff().ne(0)

    # Set initial signal based on the difference at intersection points
    signal_at_intersection = last_n_days_df['Difference'].apply(lambda x: 'Buy' if x > 0 else 'Sell')
    last_n_days_df['Signal'] = signal_at_intersection.where(last_n_days_df['Intersection'])

    # Propagate the last signal forward
    last_n_days_df['Signal'].ffill(inplace=True)
    last_n_days_df['Signal'].fillna('Hold', inplace=True) # Fill any initial NaNs with 'Hold'

    # --- 3. Trend Strength Analysis (Vectorized) ---
    last_n_days_df['Slope'] = last_n_days_df['Difference'].diff().rolling(window=slope_period).mean()
    last_n_days_df['Trend_Angle'] = np.degrees(np.arctan(last_n_days_df['Slope']))
    last_n_days_df.fillna(0, inplace=True) # Fill NaNs from rolling calculations

    # --- 4. Upgrade Signals to "Strong" (Vectorized) ---
    is_buy = last_n_days_df['Signal'] == 'Buy'
    is_strong_buy = last_n_days_df['Trend_Angle'] > strong_buy_angle
    last_n_days_df['Signal'] = np.where(is_buy & is_strong_buy, 'Strong Buy', last_n_days_df['Signal'])

    is_sell = last_n_days_df['Signal'] == 'Sell'
    is_strong_sell = last_n_days_df['Trend_Angle'] < strong_sell_angle
    last_n_days_df['Signal'] = np.where(is_sell & is_strong_sell, 'Strong Sell', last_n_days_df['Signal'])

    return last_n_days_df



##############################################################################
#this functions finds and filters uptrends ot downtrends
##----------------------------------------------------------------------------

###this function adds another layer for filtering results and add columns to find_best_trend_signal datafram output
##default values = confirmation_period = 3, momentum_threshold=0.01
def generate_advanced_signals(df, confirmation_period = 3, momentum_threshold=0.01):
    ##momentum_threshold has no effect when you use Moving average
    ##default confirmation_period=3
    """
    Analyzes price prediction data to find high-probability entry points by
    incorporating trend and momentum confirmation.

    Args:
        df (pd.DataFrame): DataFrame with 'Real_Price' and 'Predicted_Price' columns.
        confirmation_period (int): Consecutive periods a condition must hold true.
        momentum_threshold (float): Minimum slope to confirm trend strength.

    Returns:
        pd.DataFrame: The original DataFrame with new analysis and 'Advanced_Signal' columns.
    """
    analysis_df = df.copy()
##-----------------------------------------------------
##--------------------------------------------------------
    ##use one of these 3 methods to find the trend
    # --- 1. Calculate Slopes and Concurrent Trend ---
    analysis_df['Real_Slope'] = calculate_slope(analysis_df['Real_Price'])
    analysis_df['Predicted_Slope'] = calculate_slope(analysis_df['Predicted_Price'])
    analysis_df.fillna(0, inplace=True)

    analysis_df['Concurrent_Uptrend'] = (analysis_df['Real_Slope'] > momentum_threshold) & (analysis_df['Predicted_Slope'] > momentum_threshold)
    analysis_df['Concurrent_Downtrend'] = (analysis_df['Real_Slope'] < -momentum_threshold) & (analysis_df['Predicted_Slope'] < -momentum_threshold)
    ##1.calculate slope works very well, moving average does not work correctly Majid 25/08/2025
    # --- 11. 1. Moving Average (MA) Crossover for finding turning point---

    # # # Define the periods for your short and long-term moving averages
    ##------------------buy signal---------------------------------
    short_window =5 #6 #9
    long_window =15 #12 #21

    # Apply lag to avoid interfer with find peaks while verifing the trends
    lag_gor_mva = 5
    shifted_Real_Price = df['Real_Price'].shift(lag_gor_mva)
    shifted_Predicted_Price = df['Predicted_Price'].shift(lag_gor_mva)
    # Calculate the Simple Moving Averages (SMAs) on the 'Real_Price'
    analysis_df['RP_Short_SMA'] = shifted_Real_Price.rolling(window=short_window).mean()
    analysis_df['RP_Long_SMA'] = shifted_Real_Price.rolling(window=long_window).mean()
    analysis_df['PP_Short_SMA'] = shifted_Predicted_Price.rolling(window=short_window).mean()
    analysis_df['PP_Long_SMA'] =  shifted_Predicted_Price.rolling(window=long_window).mean()
    # Define the trend based on which SMA is higher
    analysis_df['Turning_Uptrend'] = ((analysis_df['PP_Short_SMA'] > analysis_df['PP_Long_SMA']))#&
                                      # (analysis_df['RP_Short_SMA'] > analysis_df['RP_Long_SMA']))
    # analysis_df['Turning_Downtrend'] = ((analysis_df['PP_Short_SMA'] < analysis_df['PP_Long_SMA'])&
    #                                     (analysis_df['RP_Short_SMA'] < analysis_df['RP_Long_SMA']))

    ##------------------sell signal---------------------------------
    short_window =5 #9
    long_window = 15 #21

    # Apply lag to avoid interfer with find peaks while verifing the trends
    lag_gor_mva = 0
    shifted_Real_Price = df['Real_Price'].shift(lag_gor_mva)
    shifted_Predicted_Price = df['Predicted_Price'].shift(lag_gor_mva)
    # Calculate the Simple Moving Averages (SMAs) on the 'Real_Price'
    analysis_df['RP_Short_SMA'] = shifted_Real_Price.rolling(window=short_window).mean()
    analysis_df['RP_Long_SMA'] = shifted_Real_Price.rolling(window=long_window).mean()
    analysis_df['PP_Short_SMA'] = shifted_Predicted_Price.rolling(window=short_window).mean()
    analysis_df['PP_Long_SMA'] =  shifted_Predicted_Price.rolling(window=long_window).mean()
    # Define the trend based on which SMA is higher
    # analysis_df['Turning_Uptrend'] = ((analysis_df['PP_Short_SMA'] > analysis_df['PP_Long_SMA'])&
    #                                   (analysis_df['RP_Short_SMA'] > analysis_df['RP_Long_SMA']))
    analysis_df['Turning_Downtrend'] = ((analysis_df['PP_Short_SMA'] < analysis_df['PP_Long_SMA']))#&
                                        # (analysis_df['RP_Short_SMA'] < analysis_df['RP_Long_SMA']))



    ##------------------another bulish market signal, price and volume combination---------------------------------
    short_window =50#50#96 #6 #9###96 is 8 hours for 5m time frame
    long_window =130#125#288 #12 #21##288 is 24 hours for 5m time frame

    # Apply lag to avoid interfer with find peaks while verifing the trends
    lag_gor_mva = 0
    Bulish_Real_Price = df['Real_Price'].shift(lag_gor_mva)
    # Bulish_Real_Price = Bulish_Real_Price.rolling(window=3).mean()
    # Bulish_volume = df['volume'].shift(lag_gor_mva)
    # Calculate the Simple Moving Averages (SMAs) on the 'Real_Price'
    analysis_df['Bulish_RP_Short_SMA'] = Bulish_Real_Price.rolling(window=short_window).mean()
    analysis_df['Bulish_RP_Long_SMA'] = Bulish_Real_Price.rolling(window=long_window).mean()
    # analysis_df['Bulish_volume_Short_SMA'] = Bulish_volume.rolling(window=short_window).mean()
    # analysis_df['Bulish_volume_Long_SMA'] =  Bulish_volume.rolling(window=long_window).mean()
    # Define the trend based on which SMA is higher

    analysis_df['Bulish_Turning_Uptrend'] = ((analysis_df['Bulish_RP_Short_SMA'] > analysis_df['Bulish_RP_Long_SMA']))
    # (analysis_df['Bulish_volume_Short_SMA'] > analysis_df['Bulish_volume_Long_SMA']) &

    analysis_df['Bulish_Turning_Downtrend'] = ((analysis_df['Bulish_RP_Short_SMA'] < analysis_df['Bulish_RP_Long_SMA']))
    # (analysis_df['Bulish_volume_Short_SMA'] < analysis_df['Bulish_volume_Long_SMA']) &)



### #---12   scipy for finding turning Points
    # --- 2. Identify Turning Points ---
    # The find_peaks function from scipy is excellent for this.
    # We can adjust parameters like 'distance' and 'prominence' to filter out minor fluctuations.
    # 'distance': Minimum horizontal distance (in number of samples) between neighboring peaks.
    # 'prominence': The vertical distance of a peak to its surrounding baseline.
    analysis_df = analyze_turning_point_stock_prices(analysis_df)


    ##----------------




#########################################################
##---------------------------------------------------------
    # --- 2. Check for Crossover State ---
    # analysis_df['Predicted_Is_Above'] = analysis_df['Predicted_Price'] > analysis_df['Real_Price']
    # analysis_df['Predicted_Is_Below'] = analysis_df['Predicted_Price'] < analysis_df['Real_Price']

###replaced above single value comparison with rolling short_window 25/08/2025 by Majid
    analysis_df['Predicted_Is_Above'] = analysis_df['Predicted_Price'].rolling(window=confirmation_period).mean() > analysis_df['Real_Price'].rolling(window=confirmation_period).mean()
    analysis_df['Predicted_Is_Below'] = analysis_df['Predicted_Price'].rolling(window=confirmation_period).mean() < analysis_df['Real_Price'].rolling(window=confirmation_period).mean()


    # --- 3. Generate Signals with Confirmation ---
    analysis_df['Advanced_Signal'] = 'Hold'
##n simple terms, it's like saying, "Only give me a 'buy' signal if the light
#has been green for the last 3 days in a row." This helps to filter out noise and
#avoid acting on temporary spikes.

    is_buy_confirmed = analysis_df['Predicted_Is_Above'].rolling(window=confirmation_period).sum() == confirmation_period
    is_sell_confirmed = analysis_df['Predicted_Is_Below'].rolling(window=confirmation_period).sum() == confirmation_period

    # Generate BUY signals only if both conditions are met
    buy_conditions = (is_buy_confirmed) & (analysis_df['Concurrent_Uptrend'])
    # buy_conditions = (analysis_df['Concurrent_Uptrend'])
    analysis_df.loc[buy_conditions, 'Advanced_Signal'] = 'Buy'

    # Generate SELL signals only if both conditions are met
    sell_conditions = (is_sell_confirmed) & (analysis_df['Concurrent_Downtrend'])
    # sell_conditions = (analysis_df['Concurrent_Downtrend'])
    analysis_df.loc[sell_conditions, 'Advanced_Signal'] = 'Sell'

    return analysis_df



#----------------------------signal detection rules------------------------------------
#------------------------------------------------------------------------------------------------


##rule 3   note: results for rule 1 and rule 3 are the same, no difference
##this is rule 1 plus added uptrend and downtrend conditions
##this works for stocks that follow prediction under real price graph
def check_trade_confirmation_rule3(df,df_mva,volume_period,lookback_days):
    """
    Checks for a confirmed trade signal based on the last 3 rows of the DataFrame.
    df_mva is the main df to calculate volume moving average

    A signal is confirmed if:
    1. There is at least one 'Intersection' value of True in the last 3 rows.
    2. The 'Signal' in the last n(lookback_days) rows is consistent (all the same) and is not 'Hold'.
    3. The 'Predicted_Is_Above' in the last n rows is consistent.
    4. The 'Predicted_Is_Below' in the last n rows is consistent.

    Args:
        df (pd.DataFrame): The DataFrame containing signal data. Must have columns
                           'Intersection', 'Signal', 'Predicted_Is_Above',
                           and 'Predicted_Is_Below'.

    Returns:
        str: 'Buy', 'Sell', or 'Hold' based on the confirmation logic.
    """
    # --- 1. Initial Checks ---
    # Ensure the DataFrame has enough rows to check
    if len(df) < lookback_days:
        print("DataFrame has fewer than 3 rows, cannot perform confirmation.")
        return "Hold"

    # --- 2. Extract Recent Data ---
    # Get the last 3 rows for checking signal consistency
    last_n_rows = df.tail(lookback_days)

    # --- 3. Define Confirmation Conditions ---

    # Condition A: Check if there has been an intersection in the last 3 rows.
    # The .any() method returns True if at least one value is True.
    is_new_intersection = last_n_rows['Intersection'].any()

    # Condition B: Check if the last 3 signals are all the same
    # The .nunique() method counts the number of unique values.
    # If it's 1, all values are the same.
    signals = last_n_rows['Signal']
    is_signal_consistent = signals.nunique() == 1

    # Condition C: Check if the consistent signal is a 'Buy' or 'Sell'
    # We get the single unique signal from the last three rows.
    last_signal = signals.iloc[0] if is_signal_consistent else None
    is_signal_actionable = last_signal in ['Buy', 'Sell']

    # Condition D: Check for consistency in 'Predicted_Is_Above'
    is_above_consistent = last_n_rows['Predicted_Is_Above'].nunique() == 1

    # Condition E: Check for consistency in 'Predicted_Is_Below'
    is_below_consistent = last_n_rows['Predicted_Is_Below'].nunique() == 1

    # comndition F: Check for consistency in 'Concurrent_Uptrend'
    # is_uptrend_consistent2 = (last_n_rows['Concurrent_Uptrend'] == True).any()
    is_uptrend_consistent = last_n_rows['Concurrent_Uptrend'].nunique() == 1

    # comndition G: Check for consistency in 'Concurrent_Downtrend'
    # is_downtrend_consistent2 = (last_n_rows['Concurrent_Downtrend'] == True).any()
    is_downtrend_consistent = last_n_rows['Concurrent_Downtrend'].nunique() == 1

    #diff_threshold= 0.001
    diff_threshold= 0.01
    #the price gap between real and prediction price is above a threshod
    is_gap_consistent = (abs(last_n_rows['Normalized_Difference']) > diff_threshold).any()

    # #below line is for finding sudo intersections
    # diff_threshold2= 0.01
    # is_gap_intersection_consistent = (abs(last_n_rows['Normalized_Difference']) < diff_threshold2).any()

    # This will return True if the absolute values are in ascending order, otherwise False.
    is_trending_upwards = abs(last_n_rows['Normalized_Difference']).round(3).is_monotonic_increasing

    #check slope of trending
    #this function is to calculate the price gap slope, is different from trend slope

    # slope = calculate_slope_polyfit(last_n_rows['Normalized_Difference'])
    # # A positive slope means the overall trend is increasing
    # print(slope)
    # is_trending_upwards = abs(slope) > 0 #slope is not angle


    ##check if the volume is above moving average
    is_volume_above_mva_consistent,volume_sma = has_volume_confirmation(df_mva,volume_period)



    # Check if all conditions are met
    if (
        is_new_intersection and
        is_above_consistent and
        is_below_consistent and
        is_signal_consistent and
        is_signal_actionable and
        is_uptrend_consistent and
        is_downtrend_consistent and
        is_volume_above_mva_consistent and
        is_gap_consistent and
        is_trending_upwards):

        confirmed_signal = last_signal
        # confirmed_signal = 'Buy'
        return confirmed_signal
    # elif(
    #     # is_new_intersection and
    #     # is_above_consistent and
    #     # is_below_consistent and
    #     is_signal_consistent and
    #     # is_signal_actionable and
    #     # is_uptrend_consistent and
    #     is_downtrend_consistent and
    #     is_downtrend_consistent2 and
    #     # is_volume_above_mva_consistent and
    #     # is_gap_consistent and
    #     is_trending_upwards):
    #     confirmed_signal = 'Sell'
    # #     # print(f"----> CONFIRMED SIGNAL: {confirmed_signal}")
    #     return confirmed_signal
    else:
        # print("----> No confirmed signal. Signal: Hold")
        return "Hold"

###below function is based on finding turning point and not intersections, finds trends very well
def check_trade_confirmation_rule4(df,df_mva,volume_period,lookback_days):
    """
    Checks for a confirmed trade signal based on the last 3 rows of the DataFrame.
    df_mva is the main df to calculate volume moving average

    A signal is confirmed if:
    1. There is at least one 'Intersection' value of True in the last 3 rows.
    2. The 'Signal' in the last n(lookback_days) rows is consistent (all the same) and is not 'Hold'.
    3. The 'Predicted_Is_Above' in the last n rows is consistent.
    4. The 'Predicted_Is_Below' in the last n rows is consistent.

    Args:
        df (pd.DataFrame): The DataFrame containing signal data. Must have columns
                           'Intersection', 'Signal', 'Predicted_Is_Above',
                           and 'Predicted_Is_Below'.

    Returns:
        str: 'Buy', 'Sell', or 'Hold' based on the confirmation logic.
    """
#------------------------------------

#------------------------------------

    # --- 1. Initial Checks ---
    # Ensure the DataFrame has enough rows to check
    # lookback_days = lookback_days + 1
    if len(df) < lookback_days:
        print("DataFrame has fewer than 3 rows, cannot perform confirmation.")
        return "Hold"

    # --- 2. Extract Recent Data ---
    # Get the last 3 rows for checking signal consistency
    last_n_rows = df.tail(lookback_days+1)

    # --- 3. Define Confirmation Conditions ---

    # Condition A: Check if there has been an intersection in the last 3 rows.
    # The .any() method returns True if at least one value is True.
    is_new_intersection = last_n_rows['Intersection'].any()

    # Condition B: Check if the last 3 signals are all the same
    # The .nunique() method counts the number of unique values.
    # If it's 1, all values are the same.
    signals = last_n_rows['Signal']
    is_signal_consistent = signals.nunique() == 1

    # Condition C: Check if the consistent signal is a 'Buy' or 'Sell'
    # We get the single unique signal from the last three rows.
    last_signal = signals.iloc[0] if is_signal_consistent else None
    is_signal_actionable = last_signal in ['Buy', 'Sell']

    # Condition D: Check for consistency in 'Predicted_Is_Above'
    is_above_consistent = last_n_rows['Predicted_Is_Above'].nunique() == 1

    # Condition E: Check for consistency in 'Predicted_Is_Below'
    is_below_consistent = last_n_rows['Predicted_Is_Below'].nunique() == 1

    # comndition F: Check for consistency in 'Concurrent_Uptrend'
    is_uptrend_consistent2 = (last_n_rows['Concurrent_Uptrend'] == True).any()
    is_uptrend_consistent = last_n_rows['Concurrent_Uptrend'].nunique() == 1

    # comndition G: Check for consistency in 'Concurrent_Downtrend'
    is_downtrend_consistent2 = (last_n_rows['Concurrent_Downtrend'] == True).any()
    is_downtrend_consistent = last_n_rows['Concurrent_Downtrend'].nunique() == 1

##------------------------------peaks and troughs-------------------------------

    # condition H: is a turning points
    is_peak_PP_consistent=(last_n_rows['is_peak_Predicted_Price'] == True).any()
    is_trough_PP_consistent=(last_n_rows['is_trough_Predicted_Price'] == True).any()

    ##is_sell_peak_Predicted_Price find peaks with different parameters than above troughs
    ##to find better sell points
    is_sell_peak_PP_consistent=(last_n_rows['is_sell_peak_Predicted_Price'] == True).any()
    is_sell_trough_PP_consistent=(last_n_rows['is_sell_trough_Predicted_Price'] == True).any()

    ##the only calculation for real price that has not been used
    is_peak_RP_consistent=(last_n_rows['is_peak_Real_Price'] == True).any()
    is_trough_RP_consistent=(last_n_rows['is_trough_Real_Price'] == True).any()

    ##---------------------very wide peaks and troughs
    ##to find better sell points
    is_peak_very_wide_PP_consistent=(last_n_rows['is_peak_very_wide_Predicted_Price'] == True).any()
    is_trough_very_wide_PP_consistent=(last_n_rows['is_trough_very_wide_Predicted_Price'] == True).any()

    is_sell_peak_very_wide_PP_consistent=(last_n_rows['is_sell_peak_very_wide_Predicted_Price'] == True).any()
    is_sell_trough_very_wide_PP_consistent=(last_n_rows['is_sell_trough_very_wide_Predicted_Price'] == True).any()


#-------------------------------------------------------------------------------

    #diff_threshold= 0.001
    diff_threshold= 0.01
    #the price gap between real and prediction price is above a threshod
    is_gap_consistent = (abs(last_n_rows['Normalized_Difference']) > diff_threshold).any()

    # #below line is for finding sudo intersections
    # diff_threshold2= 0.01
    # is_gap_intersection_consistent = (abs(last_n_rows['Normalized_Difference']) < diff_threshold2).any()

    # This will return True if the absolute values are in ascending order, otherwise False.
    is_trending_upwards = abs(last_n_rows['Normalized_Difference']).round(3).is_monotonic_increasing

    ##check turning points
    is_turning_uptrend = (last_n_rows['Turning_Uptrend'] == True).any()
    is_turning_downtrend = (last_n_rows['Turning_Downtrend'] == True).any()
    # is_turning_point = is_turning_uptrend and is_turning_downtrend ###is_peak and is_trough do the same

    ###another bulish market indicator combinations of volume and real price short and long mva
    ##this checks is last hours mva is above last 24 hours mva for both real price and volume
    ###---#good indocator for last (24hours) market overal move--------------------------------------
    is_crypto_market_bullish_uptrend_consistent2 = (last_n_rows['Bulish_Turning_Uptrend'] == True).any()
    is_crypto_market_bullish_uptrend_consistent = last_n_rows['Bulish_Turning_Uptrend'].nunique() == 1
    # condition I: Check for consistency in 'Concurrent_Downtrend'
    is_crypto_market_bullish_downtrend_consistent2 = (last_n_rows['Bulish_Turning_Downtrend'] == True).any()
    is_crypto_market_bullish_downtrend_consistent = last_n_rows['Bulish_Turning_Downtrend'].nunique() == 1



    #check slope of trending
    #this function is to calculate the price gap slope, is different from trend slope

    # slope = calculate_slope_polyfit(last_n_rows['Normalized_Difference'])
    # # A positive slope means the overall trend is increasing
    # print(slope)
    # is_trending_upwards = abs(slope) > 0 #slope is not angle


    ##check if the volume is above moving average
    is_volume_above_mva_consistent,volume_sma = has_volume_confirmation(df_mva,volume_period)


    # Check if all conditions are met
    if (
        # is_new_intersection and
        # is_above_consistent and     ##no effect
        # is_below_consistent and
        # is_signal_consistent and
        # is_signal_actionable and
        # is_uptrend_consistent and
        # is_uptrend_consistent2 and
        # is_turning_point and
        is_turning_uptrend and##
        is_trending_upwards and##
        # is_trough_RP_consistent and
        # is_downtrend_consistent and
        # is_volume_above_mva_consistent and
        is_crypto_market_bullish_uptrend_consistent2 and ##good indocator for last (24hours) market overal move
        # is_gap_consistent and
        is_trough_PP_consistent):

        # confirmed_signal = last_signal
        confirmed_signal = 'Buy'
        return confirmed_signal,False

    elif(is_trough_very_wide_PP_consistent):
        confirmed_signal = 'Buy'
        return confirmed_signal,'wide_buy'

    elif(
        # is_new_intersection and
        # is_above_consistent and
        # is_below_consistent and  ##no effect
        # is_signal_consistent and
        # is_signal_actionable and
        # is_uptrend_consistent and
        # is_downtrend_consistent and
        # is_downtrend_consistent2 and
        # is_turning_point and ###is_peak and is_trough do the same
        is_turning_downtrend and
        is_trending_upwards and
        # is_peak_RP_consistent and
        # is_volume_above_mva_consistent and  ##volume removes all buy and sells for crypto
        # is_gap_consistent and
        is_sell_peak_PP_consistent) :
        confirmed_signal = 'Sell'
    #     # print(f"----> CONFIRMED SIGNAL: {confirmed_signal}")
        return confirmed_signal, False

    elif(is_sell_peak_very_wide_PP_consistent):
        confirmed_signal = 'Sell'
        return confirmed_signal,'wide_sell'

    else:
        # print("----> No confirmed signal. Signal: Hold")
        return "Ho",None

#----------------------------------------------------------------------------

# --- 3. Wrapper Function to Generate Signals Over Time ---
##this function helps to plot the signal generator results
def generate_rule4_signals_over_time(full_df, volume_df, volume_period, lookback_days):
    """
    Iterates through the DataFrame and applies the confirmation rule at each step.
    """
    ##volume_df is actually the df in the main.py
    confirmed_signals = []
    very_wide_bool_value = []
    for i in range(lookback_days, len(full_df)):
        historical_slice = full_df.iloc[:i+1]
        signal,wide_bool_value = check_trade_confirmation_rule4(historical_slice, volume_df, volume_period, lookback_days)
        confirmed_signals.append(signal)
        very_wide_bool_value.append(wide_bool_value)
    padding = ['Hold'] * lookback_days
    padding2 = [False] * lookback_days
    return padding + confirmed_signals,padding2 + very_wide_bool_value





###----------------------final signal-------------------------------------------
##------------------------------------------------------------------------------
##------------------------------------------------------------------------------


##---------------for signals turning point method-------------------------------
#--------------------
def find_best_trend_signal_advanced(
    singnals_last_n_days_df,
    ticker,
    strong_sell_signals,
    tickers_strong_sell_signals,
    strong_buy_signals,
    tickers_strong_buy_signals,
    df,
    rule_volume_period,
    lookback_days = 1
    ):
    '''
    This function first checks check_trade_confirmation_rule4 signal, if it is not 'Buy' or 'Sell'
    it checks lookback_days+1 data points and if finds any 'Buy' or 'Sell'; it will signal it output
    "A significant challenge for live trading is the inherent signal lag caused by any analysis that
    relies on a lookback window.This delay is directly proportional to the number of data points in the calculation."
    majid 31/08/2025
    '''
    ##this converts the ticker name to alpaca ticker format to al
    if "_" in ticker:
        # If it is, replace it with '_'
        ticker = ticker.replace("_","/" )
        print(f"Character '/' found. New name: {ticker}")
    #------------------------------------
    ###first check if the market is BULISH_CONFIRMED
    market_status = is_crypto_market_bullish()
    is_crypto_market_bullish_consistent = market_status['is_bullish']
    print("--- Crypto Market Bullish Analysis (Using alpaca-py) ---")
    if "error" in market_status:
        print(f"Error: {market_status['error']}")
    else:
        if is_crypto_market_bullish_consistent:
            print(f"🚀Real Overall Bullish Assessment: {market_status['is_bullish']}🚀")
            print(f"🚀Real Confidence Score: {market_status['bullish_score']}🚀")
        else:
            print(f"🐻Real Overall Bullish Assessment: {market_status['is_bullish']}🐻")
            print(f"🐻Real Confidence Score: {market_status['bullish_score']}🐻")
        print("\n--- Detailed Analysis ---")
        print("\nTechnical Analysis (Golden Cross):")
        for symbol, data in market_status['details']['technical_analysis'].items():
            print(f"  {symbol}: {data}")

        print("\nNews Sentiment Analysis:")
        for symbol, data in market_status['details']['sentiment_analysis'].items():
            print(f"  {symbol}: {data}")
    print("\n------------------------------------")
    #-------------------------------------------------------
    is_crypto_market_bullish_consistent = True ###cheating
    #---------------------------------------------------------
    
    ##rule 1
    confirmed_signal,very_wide_bool_value = check_trade_confirmation_rule4(singnals_last_n_days_df,df,rule_volume_period,lookback_days)
    print('This is confirmed_signal:',confirmed_signal)
    print('This is very_wide_bool_value confirmed_signal:',very_wide_bool_value)

    ##rule 2
    ##this checks last two values if the last value has no buy or sell signals
    ##it triggers the signal with one data point latency,
    ##it is the best accuracy for live automatic trading for now
    confirmed_signal_lookback_days,very_wide_bool_value_lookback_days = generate_rule4_signals_over_time(singnals_last_n_days_df,df,rule_volume_period,lookback_days)
    print('This is confirmed_signal_lookback_days:',confirmed_signal_lookback_days)
    print('This is very_wide_bool_value_lookback_days:',very_wide_bool_value_lookback_days)
##----------------------------lookback days offset-----------------------------------
    ##two below if clauses find very wide peaks and traughs which have wide lags
    vwbvld = very_wide_bool_value_lookback_days[-(lookback_days+25):]
    csld = confirmed_signal_lookback_days[-(lookback_days+25):]

    # Loop through the list once to check for either signal
    for value in vwbvld:
        if value == 'wide_buy':
            confirmed_signal = 'Buy'
            print("🌕 'Buy' wide turning point found. 🌕")
            break  # Exit the loop once the first signal is found
        elif value == 'wide_sell':
            confirmed_signal = 'Sell'
            print("🌕 'Sell' wide turning point found. 🌕")
            break  # Exit the loop once the first signal is found


    
##-----------------------------------------------------------------------------------
    confirmed_signal_tail = confirmed_signal_lookback_days[-(lookback_days+4):]
    print('This is confirmed signal tail:',confirmed_signal_tail)
##-----------------------------------------------------------------------------------
    # The check:
    # Assume this is your variable with two values
    actionable_triggers = ['Buy', 'Sell']
    # 1. Create a new list with only the actionable signals
    actionable_signals = [s for s in confirmed_signal_tail if s in actionable_triggers]
    print('actionable_signals:',actionable_signals)
    # 2. Check if the new list is not empty and get the signal
    actual_signal='Hold'
    if actionable_signals:
        # If the list is not empty, it means we found at least one signal.
        # We take the first one found.
        actual_signal = actionable_signals[-1]
        # print('actual_signal:',actual_signal)

    ##rule 4
    # confirmed_signal = check_trade_confirmation_rule4(singnals_last_n_days_df)

    if confirmed_signal == 'Sell' and is_crypto_market_bullish_consistent:
        strong_sell_signals.append(confirmed_signal)
        tickers_strong_sell_signals.append(ticker)
        print(f"✅🐳🦋 Sell signal found and stored: Rule1 ('{confirmed_signal}') and Rule2 ('{actual_signal}').")
        print(f"✅🐳🦋 crypto market bullish indicator is: ('{is_crypto_market_bullish_consistent}').")


    elif confirmed_signal == 'Buy' and is_crypto_market_bullish_consistent:
        strong_buy_signals.append(confirmed_signal)
        tickers_strong_buy_signals.append(ticker)
        print(f"✅🐳🦋 Buy signal found and stored: Rule1 ('{confirmed_signal}') and Rule2 ('{actual_signal}').")
        print(f"✅🐳🦋 crypto market bullish indicator is: ('{is_crypto_market_bullish_consistent}').")


    elif actual_signal == 'Sell' and is_crypto_market_bullish_consistent:
        strong_sell_signals.append('Sell')
        tickers_strong_sell_signals.append(ticker)
        print(f"✅🦦🐢 Sell signal found and stored: Rule1 ('{confirmed_signal}') and Rule2 ('{actual_signal}').")
        print(f"✅🦦🐢 crypto market bullish indicator is: ('{is_crypto_market_bullish_consistent}').")


    elif actual_signal == 'Buy' and is_crypto_market_bullish_consistent:
        strong_buy_signals.append('Buy')
        tickers_strong_buy_signals.append(ticker)
        print(f"✅🦦🐢 Buy signal found and stored: Rule1 ('{confirmed_signal}') and Rule2 ('{actual_signal}').")
        print(f"✅🦦🐢 crypto market bullish indicator is: ('{is_crypto_market_bullish_consistent}').")


    else:
        print(f"❌ No strong signal found. Last value is Rule1 ('{confirmed_signal}') and Rule2 ('{actual_signal}').")
        print(f"❌  crypto market bullish indicator is: ('{is_crypto_market_bullish_consistent}').")
    return strong_sell_signals,tickers_strong_sell_signals,strong_buy_signals,tickers_strong_buy_signals
