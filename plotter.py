import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
import numpy as np

from indicators_compute_DFlabels_adjust import has_volume_confirmation
import intersection_slope_signal as ints
import config as cfg

def find_closest_timestamp(df, target_timestamp_str):
    """
    Finds the row in a DataFrame with the timestamp closest to a target timestamp.

    Args:
        df (pd.DataFrame): The DataFrame to search, which must have a 'Date' column
                           of datetime objects and a standard integer index.
        target_timestamp_str (str): The target timestamp in a string format
                                    (e.g., "2025-07-24 14:30:00+00:00").

    Returns:
        tuple: A tuple containing (integer_index, row_data) for the closest match,
               or (None, None) if the DataFrame is empty.
    """
    if df.empty:
        return None, None

    # Convert the target string to a pandas Timestamp object
    target_ts = pd.to_datetime(target_timestamp_str)

    # Calculate the absolute time difference for each row
    time_diff = (df['Date'] - target_ts).abs()

    # Find the integer index of the row with the smallest time difference
    closest_integer_index = time_diff.idxmin()

    # Get the data for that row using the found index
    closest_row_data = df.loc[closest_integer_index]

    return closest_integer_index, closest_row_data



def get_market_open_time(data_frame,hour,minute):
    # 1. finding market opening time index
    # 1. Get today's date and create the target time.
    today_date = datetime.date.today() # Gets today's date, e.g., 2025-07-24
    yesterday_date = today_date - datetime.timedelta(days=1)

    target_time = datetime.time(hour, minute) # Defines the time 13:30

    # --- 2. Find Timestamp for Today ---
    naive_datetime_today = datetime.datetime.combine(today_date, target_time)
    aware_datetime_utc_today = naive_datetime_today.replace(tzinfo=datetime.timezone.utc)
    target_str_today = aware_datetime_utc_today.isoformat()
    today_idx, today_row = find_closest_timestamp(data_frame, target_str_today)


    # --- 3. Find Timestamp for Yesterday ---
    naive_datetime_yesterday = datetime.datetime.combine(yesterday_date, target_time)
    aware_datetime_utc_yesterday = naive_datetime_yesterday.replace(tzinfo=datetime.timezone.utc)
    target_str_yesterday = aware_datetime_utc_yesterday.isoformat()
    yesterday_idx, yesterday_row = find_closest_timestamp(data_frame, target_str_yesterday)


    return today_idx, today_row, yesterday_idx, yesterday_row


##this function plots real price and model predicted price and saves the figure
##for daily time frame
def plot_price_prediction_day_tf(
    test_dates,
    y_test_actual,
    predictions,
    ticker_symbol,
    SELECTED_feature_SET,
    time_interval,
    train_size_ratio,
    start_date,
    FOLDER_PLOTS_SAVED,
    ENABLE_SAVE_PREDICTION_FIGURE,
    df,
    VOLUME_PERIOD_MVA_DAILY
    ):

    # 1. Get the current date and time
    now = datetime.datetime.now()
    # 1. Format it into a string suitable for a filename (YYYY-MM-DD_HH-MM-SS)
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    ##market overal signals
    # 3. Run the function to get the result and the SMA series
    volume_confirmed , volume_sma = has_volume_confirmation(df, VOLUME_PERIOD_MVA_DAILY)
    bulish_confirmed = cfg.BULISH_CONFIRMED


    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_test_actual, label='Actual Prices', color='blue',marker='|',markersize=15)
    plt.plot(test_dates, predictions, label='Predicted Prices', color='red', linestyle='--',marker='x',markersize=5)
    plt.title(f'{ticker_symbol}_{timestamp}_Stock Price_{SELECTED_feature_SET}_{time_interval}_Prediction_tsr{train_size_ratio}_{start_date.date()} \n Volume Confirmation: {volume_confirmed},  Bulish Market: {bulish_confirmed},  (Test Set)')
    ##plots volume sma
    # plt.plot(test_dates, volume_sma[-len(test_dates):], color='orange', linestyle='-.', label='20-Period Avg. Volume')
    # x_ticks = np.arange(test_dates[0], test_dates[-1], 10)
    # # print(x_ticks)
    # plt.xticks(x_ticks)

    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)

    # 3. Create your full filename and save the plot
    # It's good practice to add a file extension like .png
    filename = f'{ticker_symbol}_{timestamp}_Stock_Price_{SELECTED_feature_SET}_{time_interval}_Prediction_tsr{train_size_ratio}_{start_date.date()}.png'
    # Get the directory of the current script (main_script.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the model file
    # Joins: my_project_directory + models + my_keras_model.keras
    plot_path_name = os.path.join(script_dir,FOLDER_PLOTS_SAVED,filename)

    if ENABLE_SAVE_PREDICTION_FIGURE:
        plt.savefig(plot_path_name)
    ###plotting the predicitons
    # plt.pause(1)
    # 4. Close the figure automatically
    plt.close()


##this function plots real price and model predicted price and saves the figure
##for minutes time frame
def plot_price_prediction_minute_tf(
    test_dates,
    y_test_actual,
    predictions,
    ticker_symbol,
    SELECTED_feature_SET,
    time_interval,
    train_size_ratio,
    start_date,
    FOLDER_PLOTS_SAVED,
    ENABLE_SAVE_PREDICTION_FIGURE,
    TIME_CONV_RATIO,
    df,
    VOLUME_PERIOD_MVA_MINUTE
    ):

    # 1. finding market opening time index for today and yesterday to plot
    market_open_idx, market_open_row,yesterday_idx, yesterday_row = get_market_open_time(df,13,30)
    market_open_idx = market_open_idx/ TIME_CONV_RATIO
    yesterday_idx = yesterday_idx/ TIME_CONV_RATIO

    # # Visualize the predictions vs actual prices
    # # Adjusting dates for chronological test set
    # print(df)
    # train_end_date_idx = len(df) - len(y_test_actual)
    # test_dates = df.index[train_end_date_idx:]

    # 1. Get the current date and time
    now = datetime.datetime.now()
    # 1. Format it into a string suitable for a filename (YYYY-MM-DD_HH-MM-SS)
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    test_hours = np.array(test_dates) / TIME_CONV_RATIO ###converting test dates from minutes to hours

    ##market overal signals
    # 3. Run the function to get the result and the SMA series
    volume_confirmed , volume_sma = has_volume_confirmation(df,VOLUME_PERIOD_MVA_MINUTE)
    bulish_confirmed = cfg.BULISH_CONFIRMED

    plt.figure(figsize=(14, 7))
    plt.plot([yesterday_idx]*len(y_test_actual),y_test_actual)
    plt.plot([market_open_idx]*len(y_test_actual),y_test_actual)
    plt.plot(test_hours, y_test_actual, label='Actual Prices', color='blue', linestyle=':',marker='|',markersize=15)
    plt.plot(test_hours, predictions, label='Predicted Prices', color='red', linestyle='--',marker='x',markersize=5)
    ##plots volume sma
    # plt.plot(test_hours, volume_sma[-len(test_hours):], color='orange', linestyle='-.', label='20-Period Avg. Volume')
    #adjust x-axix tickers
    # Define the specific ticks we want to see

    # x_ticks = np.arange(test_hours[0], test_hours[-1], 30)
    # # print(x_ticks)
    # plt.xticks(x_ticks)

    plt.title(f'{ticker_symbol}_{timestamp}_Stock Price_{SELECTED_feature_SET}_{time_interval}_Prediction_tsr{train_size_ratio}_{start_date.date()} \n Volume Confirmation: {volume_confirmed},  Bulish Market: {bulish_confirmed}  (Test Set)')
    plt.xlabel('Hour')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)

    # 3. Create your full filename and save the plot
    # It's good practice to add a file extension like .png
    filename = f'{ticker_symbol}_{timestamp}_Stock_Price_{SELECTED_feature_SET}_{time_interval}_Prediction_tsr{train_size_ratio}_{start_date.date()}.png'
    # Get the directory of the current script (main_script.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the model file
    # Joins: my_project_directory + models + my_keras_model.keras
    plot_path_name = os.path.join(script_dir,FOLDER_PLOTS_SAVED,filename)

    if ENABLE_SAVE_PREDICTION_FIGURE:
        plt.savefig(plot_path_name)
    ###plotting the predicitons
    # plt.pause(1)
    # 4. Close the figure automatically
    plt.close()


def simple_series_plot(
    ticker_symbol,
    SELECTED_feature_SET,
    time_interval,
    train_size_ratio,
    start_date,
    FOLDER_PLOTS_SAVED,
    ENABLE_SAVE_PREDICTION_FIGURE,
    singnals_last_n_days_df,
    df,
    ENTRY_EXIT_SIMPLE_SERIES_PLOT,
    LOOKBACK_DAYS = 1,
    VOLUME_PERIOD = 20
    ):

    # C. Generate the final Rule 3 signals over the entire time series
    # LOOKBACK_DAYS = 1
    # VOLUME_PERIOD = 20

    singnals_last_n_days_df['Rule3_Signal'],_ = ints.generate_rule4_signals_over_time(singnals_last_n_days_df, df, VOLUME_PERIOD, LOOKBACK_DAYS)
    singnals_last_n_days_df.fillna(method='ffill', inplace=True)
    # print('This is Rule4_Signal:',singnals_last_n_days_df['Rule3_Signal'])
    # D. Plot the final results
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(singnals_last_n_days_df.index, singnals_last_n_days_df['Real_Price'], label='Real Price',marker='|',markersize=15, color='blue', lw=2, alpha=0.9)
    ax.plot(singnals_last_n_days_df.index, singnals_last_n_days_df['Predicted_Price'], label='Predicted Price',marker='x',markersize=5, color='orange', linestyle='--', lw=2)
    ax.plot(singnals_last_n_days_df.index, singnals_last_n_days_df['No_Smooth_Real_Price'], label='Real Price not smoothed',marker='d',markersize=4, color='m', lw=1, alpha=0.5)


    #------plotting find_peaks
    
    buy_signals = singnals_last_n_days_df[singnals_last_n_days_df['Rule3_Signal'] == 'Buy']
    sell_signals = singnals_last_n_days_df[singnals_last_n_days_df['Rule3_Signal'] == 'Sell']


    ax.scatter(buy_signals.index, buy_signals['Predicted_Price'] * 0.998, label='Confirmed Buy Signal (Rule 3)', marker='^', color='green', s=200, zorder=10, edgecolors='k')
    ax.scatter(sell_signals.index, sell_signals['Predicted_Price'] * 1.002, label='Confirmed Sell Signal (Rule 3)', marker='v', color='red', s=200, zorder=10, edgecolors='k')
    
#-----------
    hour_8over24_sma = (singnals_last_n_days_df['Bulish_RP_Short_SMA'].mean()>singnals_last_n_days_df['Bulish_RP_Long_SMA'].mean())
    last_n_rows = singnals_last_n_days_df.tail(2)
    is_crypto_market_bullish_uptrend_consistent2 = (last_n_rows['Bulish_Turning_Uptrend'] == True).any()

    ax.set_title(f'{ticker_symbol} Time Series, Markrt is Bulish(8/24 hours): {hour_8over24_sma} or {is_crypto_market_bullish_uptrend_consistent2}; Confirmed Signals (Rule 4)', fontsize=18)
    # ax.set_title(f'{ticker_symbol} Time Series with Confirmed Signals (Rule 3)', fontsize=18)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend()
    plt.tight_layout()

    # 1. Get the current date and time
    now = datetime.datetime.now()
    # 1. Format it into a string suitable for a filename (YYYY-MM-DD_HH-MM-SS)
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    #3. Create your full filename and save the plot
    #It's good practice to add a file extension like .png
    filename = f'{ticker_symbol}_{timestamp}_rule3_Price_{SELECTED_feature_SET}_{time_interval}_Prediction_tsr{train_size_ratio}_{start_date.date()}.png'
    # Get the directory of the current script (main_script.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the model file
    # Joins: my_project_directory + models + my_keras_model.keras
    plot_path_name = os.path.join(script_dir,FOLDER_PLOTS_SAVED,filename)

    if ENTRY_EXIT_SIMPLE_SERIES_PLOT:
        plt.savefig(plot_path_name)
    ##plotting the predicitons

    # plt.show()
    # ###plotting the predicitons
    # plt.pause(1)
    # 4. Close the figure automatically
    plt.close()
