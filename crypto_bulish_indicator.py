import os
import pandas as pd
from textblob import TextBlob
from datetime import datetime, timedelta

# Imports for the modern alpaca-py SDK
from alpaca.data import TimeFrame
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, NewsRequest
from alpaca.data.historical.news import NewsClient
# It's recommended to set your API keys as environment variables for security
# You can get these from your Alpaca dashboard.
# --- Import Your Project's Config ---
import config as cfg


# --- Your API Credentials ---
# Make sure to set these as environment variables for security
API_KEY = os.environ.get('APCA_API_KEY_ID', cfg.ALPACA_API_KEY_ID)
API_SECRET = os.environ.get('APCA_API_SECRET_KEY', cfg.ALPACA_SECRET_KEY)
# Use paper=True for paper trading, paper=False for live trading
# PAPER = False #True
PAPER = cfg.PAPER



def is_crypto_market_bullish(api_key=API_KEY, api_secret=API_SECRET):
    """
    Analyzes the crypto market to determine if it's bullish using the modern alpaca-py SDK.

    This function uses a multi-factor approach:
    1.  Technical Analysis: Checks for a "golden cross" (50-day moving average
        crossing above the 200-day moving average) for Bitcoin (BTC/USD) and
        Ethereum (ETH/USD). A golden cross is a strong long-term bullish indicator.
    2.  Sentiment Analysis: Gathers recent news for BTC and ETH and analyzes the
        sentiment of the headlines. Positive news sentiment can be a leading
        indicator of bullish price action.

    Args:
        api_key (str): Your Alpaca API key.
        api_secret (str): Your Alpaca API secret key.

    Returns:
        dict: A dictionary containing the overall bullish assessment,
              along with the detailed results from the technical and
              sentiment analyses.

              The script requires a Confidence Score of at least 75% (or 3/4) to consider the market bullish.
    """
    if not api_key or not api_secret:
        return {
            "is_bullish": False,
            "error": "API key and secret must be provided or set as environment variables."
        }

    # Initialize the modern data clients
    crypto_client = CryptoHistoricalDataClient(api_key, api_secret)
    news_client = NewsClient(api_key, api_secret)

    # --- Parameters ---
    major_cryptos = ['BTC/USD', 'ETH/USD']
    short_window = 50
    long_window = 200
    news_limit = 50
    sentiment_threshold = 0.1  # Polarity score above which we consider sentiment "positive"

    # We need at least 200 days of data for the moving averages
    start_date = (datetime.now() - timedelta(days=300)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    analysis_results = {
        "technical_analysis": {},
        "sentiment_analysis": {},
    }

    bullish_score = 0
    total_possible_score = len(major_cryptos) * 2 # One point for TA, one for sentiment per crypto

    # --- 1. Technical Analysis ---
    for symbol in major_cryptos:
        try:
            # Construct the request for historical bar data
            request_params = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            # Retrieve historical daily data (bars)
            bars = crypto_client.get_crypto_bars(request_params).df

            if len(bars) < long_window:
                analysis_results["technical_analysis"][symbol] = f"Not enough data to calculate {long_window}-day MA."
                continue

            # Calculate moving averages
            bars['sma_short'] = bars['close'].rolling(window=short_window).mean()
            bars['sma_long'] = bars['close'].rolling(window=long_window).mean()

            # Check for golden cross
            latest_sma_short = bars['sma_short'].iloc[-1]
            latest_sma_long = bars['sma_long'].iloc[-1]

            is_golden_cross = latest_sma_short > latest_sma_long

            analysis_results["technical_analysis"][symbol] = {
                f"sma_{short_window}": round(latest_sma_short, 2),
                f"sma_{long_window}": round(latest_sma_long, 2),
                "golden_cross_active": is_golden_cross
            }
            if is_golden_cross:
                bullish_score += 1

        except Exception as e:
            analysis_results["technical_analysis"][symbol] = f"An error occurred: {e}"

    # --- 2. Sentiment Analysis ---
    for symbol in major_cryptos:
        try:
            symbol_name = symbol.split('/')[0]
            # Construct the request for news articles
            news_request_params = NewsRequest(
                symbols=symbol_name,
                limit=news_limit
            )
            # Retrieve news articles. Returns a dict-like NewsSet object.
            news_set = news_client.get_news(news_request_params)

            # FIX: Using a more robust try/except block to access news data.
            articles_for_symbol = []
            try:
                articles_for_symbol = news_set[symbol_name]
            except KeyError:
                # This handles the case where no news is returned for the symbol.
                pass

            if not articles_for_symbol:
                analysis_results["sentiment_analysis"][symbol] = "No news found."
                continue

            symbol_polarity = 0
            headlines = [item.headline for item in articles_for_symbol if item.headline]

            for headline in headlines:
                analysis = TextBlob(headline)
                symbol_polarity += analysis.sentiment.polarity

            average_polarity = symbol_polarity / len(headlines) if headlines else 0

            analysis_results["sentiment_analysis"][symbol] = {
                "average_polarity": round(average_polarity, 4),
                "is_positive": average_polarity > sentiment_threshold
            }
            if average_polarity > sentiment_threshold:
                bullish_score += 1

        except Exception as e:
            analysis_results["sentiment_analysis"][symbol] = f"An error occurred: {e}"

    # --- Final Assessment ---
    final_assessment = (bullish_score / total_possible_score) >= 0.75 if total_possible_score > 0 else False

    return {
        "is_bullish": final_assessment,
        "bullish_score": f"{bullish_score}/{total_possible_score}",
        "details": analysis_results
    }


if __name__ == '__main__':
    # To run this, you must have your Alpaca API keys set as environment variables:
    # On Mac/Linux:
    # export APCA_API_KEY_ID='YOUR_KEY_ID'
    # export APCA_API_SECRET_KEY='YOUR_SECRET_KEY'
    #
    # On Windows:
    # set APCA_API_KEY_ID='YOUR_KEY_ID'
    # set APCA_API_SECRET_KEY='YOUR_SECRET_KEY'
    #
    # You might also need to install the required libraries:
    # pip install alpaca-py pandas textblob
    #
    # And download the TextBlob corpora:
    # python -m textblob.download_corpora

    market_status = is_crypto_market_bullish()

    # is_crypto_market_bullish_consistant = market_status['is_bullish']
    # print(type(is_crypto_market_bullish_consistant))

    print("--- Crypto Market Bullish Analysis (Using alpaca-py) ---")
    if "error" in market_status:
        print(f"Error: {market_status['error']}")
    else:
        print(f"Overall Bullish Assessment: {market_status['is_bullish']}")
        print(f"Confidence Score: {market_status['bullish_score']}")
        print("\n--- Detailed Analysis ---")
        print("\nTechnical Analysis (Golden Cross):")
        for symbol, data in market_status['details']['technical_analysis'].items():
            print(f"  {symbol}: {data}")

        print("\nNews Sentiment Analysis:")
        for symbol, data in market_status['details']['sentiment_analysis'].items():
            print(f"  {symbol}: {data}")
    print("\n------------------------------------")
