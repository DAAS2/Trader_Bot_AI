import numpy as np
from ta import volatility, momentum, trend, volume, others
import yfinance as yf
import pandas as pd
import talib as ta_lib
from tradingview_ta import TA_Handler, Interval, Exchange
import google.generativeai as genai
import json
import os # To get environment variables

# --- Gemini API Configuration ---

genai.configure(api_key=API_KEY)


gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")


# Global variables (used for historical context in market regime)
historical_adx_state = None
historical_candlestick_patterns = None
reasons = [] # This list will accumulate all reasons for the decision.


def _safe_float(value, default_val=0.0):
    """Safely converts a value to float, handling N/A or non-numeric strings."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default_val

def calculate_pivot_points(high_prev, low_prev, close_prev):
    """
    Calculates Pivot Point (PP) and its associated Support (S) and Resistance (R) levels.
    Uses the previous day's (or last complete candle's) High, Low, and Close.
    """
    if any(np.isnan([high_prev, low_prev, close_prev])):
        return {'PP': np.nan, 'R1': np.nan, 'R2': np.nan, 'S1': np.nan, 'S2': np.nan}

    pp = (high_prev + low_prev + close_prev) / 3
    r1 = (pp * 2) - low_prev
    s1 = (pp * 2) - high_prev
    r2 = pp + (high_prev - low_prev)
    s2 = pp - (high_prev - low_prev)

    return {'PP': pp, 'R1': r1, 'R2': r2, 'S1': s1, 'S2': s2}


def get_candlestick_patterns(open_prices, high_prices, low_prices, close_prices):
    """
    Calculates various TA-Lib candlestick patterns and returns their total detection counts.
    For the latest candle, it returns the specific pattern value.
    """
    patterns_raw = {
        "hammer": ta_lib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices),
        "engulfing": ta_lib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices),
        "doji": ta_lib.CDLDOJI(open_prices, high_prices, low_prices, close_prices),
        "two_crows": ta_lib.CDL2CROWS(open_prices, high_prices, low_prices, close_prices),
        "breakaway": ta_lib.CDLBREAKAWAY(open_prices, high_prices, low_prices, close_prices),
        "high_wave": ta_lib.CDLHIGHWAVE(open_prices, high_prices, low_prices, close_prices),
        "three_black_crows": ta_lib.CDL3BLACKCROWS(open_prices, high_prices, low_prices, close_prices),
        "three_inside": ta_lib.CDL3INSIDE(open_prices, high_prices, low_prices, close_prices),
        "three_line": ta_lib.CDL3LINESTRIKE(open_prices, high_prices, low_prices, close_prices),
        "three_outside": ta_lib.CDL3OUTSIDE(open_prices, high_prices, low_prices, close_prices),
        "advance_block": ta_lib.CDLADVANCEBLOCK(open_prices, high_prices, low_prices, close_prices),
        "counterattack": ta_lib.CDLCOUNTERATTACK(open_prices, high_prices, low_prices, close_prices),
        "gap_side_white": ta_lib.CDLGAPSIDESIDEWHITE(open_prices, high_prices, low_prices, close_prices),
        "kicking_bull": ta_lib.CDLKICKINGBYLENGTH(open_prices, high_prices, low_prices, close_prices),
        "long_line": ta_lib.CDLLONGLINE(open_prices, high_prices, low_prices, close_prices),
        "piercing": ta_lib.CDLPIERCING(open_prices, high_prices, low_prices, close_prices),
        "short_line": ta_lib.CDLSHORTLINE(open_prices, high_prices, low_prices, close_prices),
        "seperating_lines": ta_lib.CDLSEPARATINGLINES(open_prices, high_prices, low_prices, close_prices),
        "spinning_top": ta_lib.CDLSPINNINGTOP(open_prices, high_prices, low_prices, close_prices)
    }

    # Count non-zero occurrences for each pattern for historical context
    patterns_counts = {name: np.sum(pattern != 0) for name, pattern in patterns_raw.items()}

    # Get the value for the latest candle for decision making
    latest_patterns_values = {name: pattern.iloc[-1].item() if len(pattern) > 0 else 0 for name, pattern in patterns_raw.items()}

    return patterns_counts, latest_patterns_values

def get_historical_data(ticker, interval, yfinance_period):
    global historical_adx_state, historical_candlestick_patterns

    print(f"Fetching historical data for {ticker} with interval {interval} (period={yfinance_period}) from Yahoo Finance...")
    data = yf.download(ticker, interval=interval, period=yfinance_period, auto_adjust=True)
    data.columns = [col[0].lower() for col in data.columns] # Ensure column names are lowercase
    data_cleaned = data.dropna()

    if data_cleaned.empty or len(data_cleaned) < 20: # Ensure enough data for indicators (e.g., ADX needs 14-20 periods)
        print("Not enough historical data retrieved from Yahoo Finance for indicator calculations.")
        historical_adx_state = "INSUFFICIENT DATA"
        historical_candlestick_patterns = {}
        return pd.Series([]), {}, "INSUFFICIENT DATA", \
               pd.Series([]), pd.Series([]), pd.Series([]), pd.Series([]), pd.Series([])

    # Ensure data types are float64 for TA-Lib and `ta`
    open_prices_series = pd.Series(data_cleaned["open"].values.astype(np.float64))
    close_prices_series = pd.Series(data_cleaned["close"].values.astype(np.float64))
    high_prices_series = pd.Series(data_cleaned["high"].values.astype(np.float64))
    low_prices_series = pd.Series(data_cleaned["low"].values.astype(np.float64))
    volume_series = pd.Series(data_cleaned["volume"].values.astype(np.float64)) # Volume as float64

    # Get historical candlestick patterns (counts over the period and latest pattern value)
    historical_candlestick_patterns_counts, _ = get_candlestick_patterns(
        open_prices_series, high_prices_series, low_prices_series, close_prices_series
    )
    historical_candlestick_patterns = historical_candlestick_patterns_counts # Update global

    # Get historical ADX using `ta` library
    historical_adx = trend.ADXIndicator(
        high_prices_series, low_prices_series, close_prices_series, window=14, fillna=True
    ).adx()

    # Determine historical market conditions based on ADX
    if len(historical_adx) > 0 and not historical_adx.isnull().all():
        # Drop NaNs for percentage calculation, ADX can have initial NaNs
        adx_valid = historical_adx.dropna()
        if not adx_valid.empty:
            if (adx_valid < 25).sum() / len(adx_valid) > 0.7:
                historical_adx_state = "RANGING"
            elif (adx_valid > 40).sum() / len(adx_valid) > 0.5:
                historical_adx_state = "TRENDING"
            else:
                historical_adx_state = "MIXED/UNCLEAR"
        else:
            historical_adx_state = "INSUFFICIENT DATA (ADX NaN)"
    else:
        historical_adx_state = "INSUFFICIENT DATA (ADX calculation failed)"

    return historical_adx, historical_candlestick_patterns_counts, historical_adx_state, \
           open_prices_series, high_prices_series, low_prices_series, close_prices_series, volume_series


def get_indicators(open_prices, high_prices, low_prices, close_prices, volume):
    """
    Calculates a comprehensive set of technical indicators using the `ta` library.
    Returns a dictionary where each key is an indicator name and the value is its Pandas Series.
    """
    indicators = {}

    # Momentum Indicators
    indicators['rsi'] = momentum.RSIIndicator(close_prices, window=14, fillna=True).rsi()
    indicators['stoch_k'] = momentum.StochasticOscillator(high_prices, low_prices, close_prices, window=14, smooth_window=3, fillna=True).stoch()
    indicators['stoch_d'] = momentum.StochasticOscillator(high_prices, low_prices, close_prices, window=14, smooth_window=3, fillna=True).stoch_signal()
    indicators['stoch_rsi'] = momentum.StochRSIIndicator(close_prices, window=14, smooth1=3, smooth2=3, fillna=True).stochrsi()
    indicators['cci'] = trend.CCIIndicator(high_prices, low_prices, close_prices, window=20, fillna=True).cci()

    # Trend Indicators
    indicators['aroon_up'] = trend.AroonIndicator(close_prices, low_prices, window=14, fillna=True).aroon_up()
    indicators['aroon_down'] = trend.AroonIndicator(close_prices, low_prices, window=14, fillna=True).aroon_down()
    indicators['adx'] = trend.ADXIndicator(high_prices, low_prices, close_prices, window=14, fillna=True).adx()
    indicators['neg_adx'] = trend.ADXIndicator(high_prices, low_prices, close_prices, window=14, fillna=True).adx_neg()
    indicators['pos_adx'] = trend.ADXIndicator(high_prices, low_prices, close_prices, window=14, fillna=True).adx_pos()
    indicators['dpo'] = trend.DPOIndicator(close_prices, window=20, fillna=True).dpo()
    indicators['ichimoku_a'] = trend.IchimokuIndicator(high_prices, low_prices, window1=9, window2=26, window3=52, fillna=True).ichimoku_a()
    indicators['ichimoku_b'] = trend.IchimokuIndicator(high_prices, low_prices, window1=9, window2=26, window3=52, fillna=True).ichimoku_b()
    indicators['macd'] = trend.MACD(close_prices, window_slow=26, window_fast=12, window_sign=9, fillna=True).macd()
    indicators['macd_signal'] = trend.MACD(close_prices, window_slow=26, window_fast=12, window_sign=9, fillna=True).macd_signal()
    indicators['macd_diff'] = trend.MACD(close_prices, window_slow=26, window_fast=12, window_sign=9, fillna=True).macd_diff()
    indicators['sma'] = trend.SMAIndicator(close_prices, window=20, fillna=True).sma_indicator()
    indicators['mass_index'] = trend.MassIndex(high_prices, low_prices, window_fast=9, fillna=True).mass_index()
    indicators['psar'] = trend.PSARIndicator(high_prices, low_prices, close_prices, step=0.02, max_step=0.2, fillna=True).psar()
    indicators['psar_down'] = trend.PSARIndicator(high_prices, low_prices, close_prices, step=0.02, max_step=0.2, fillna=True).psar_down()
    indicators['psar_up'] = trend.PSARIndicator(high_prices, low_prices, close_prices, step=0.02, max_step=0.2, fillna=True).psar_up()
    indicators['stc'] = trend.STCIndicator(close_prices, window_slow=50, window_fast=23, fillna=True).stc()

    vortex_obj = trend.VortexIndicator(high_prices, low_prices, close_prices, window=14, fillna=True)
    indicators['vortex_plus'] = vortex_obj.vortex_indicator_pos()
    indicators['vortex_minus'] = vortex_obj.vortex_indicator_neg()

    # Volatility Indicators
    indicators['atr'] = volatility.AverageTrueRange(high_prices, low_prices, close_prices, window=14, fillna=True).average_true_range()
    # TA-Lib ATR takes numpy arrays. Ensure consistency for this specific one.
    indicators['atr_ta_lib'] = ta_lib.ATR(high_prices.values, low_prices.values, close_prices.values, timeperiod=14)
    bollinger_obj = volatility.BollingerBands(close_prices, window=20, window_dev=2, fillna=True)
    indicators['bollinger_hband'] = bollinger_obj.bollinger_hband()
    indicators['bollinger_lband'] = bollinger_obj.bollinger_lband()
    keltner_obj = volatility.KeltnerChannel(high_prices, low_prices, close_prices, window=20, window_atr=10, fillna=True)
    indicators['keltner_channel_hband'] = keltner_obj.keltner_channel_hband_indicator()
    indicators['keltner_channel_lband'] = keltner_obj.keltner_channel_lband_indicator()
    indicators['ulcer_index'] = volatility.UlcerIndex(close_prices, fillna=True).ulcer_index()

    return indicators


def get_current_data(ticker_yf, interval_yf, exchange, screener, ta_symbol_tv):
    """
    Fetches current day's data using yfinance, calculates live indicators and patterns,
    determines the current market regime, and calculates pivot points.
    Returns a dictionary containing all processed data and analysis.
    """
    global reasons # Declare global to modify the `reasons` list

    # Fetch enough data to ensure at least 2 complete candles for pivot points and indicators.
    print(f"Fetching current data for {ticker_yf} with interval {interval_yf} (period=1d) from Yahoo Finance...")
    data = yf.download(ticker_yf, interval=interval_yf, period="1d", auto_adjust=True)
    data.columns = [col[0].lower() for col in data.columns]
    data_cleaned = data.dropna()

    # Check if data is empty or has less than 2 rows for pivot point calculation
    if data_cleaned.empty or len(data_cleaned) < 2:
        print(f"Not enough data retrieved for {ticker_yf} with interval {interval_yf}. Need at least 2 candles for pivot points and indicators.")
        return None # Indicate failure to retrieve data

    # Extract prices and volume as Pandas Series for `ta` library
    open_prices_series = pd.Series(data_cleaned["open"].values.astype(np.float64))
    close_prices_series = pd.Series(data_cleaned["close"].values.astype(np.float64))
    high_prices_series = pd.Series(data_cleaned["high"].values.astype(np.float64))
    low_prices_series = pd.Series(data_cleaned["low"].values.astype(np.float64))
    volume_series = pd.Series(data_cleaned["volume"].values.astype(np.float64))

    # Get live indicators (full Series for all indicators)
    live_indicators = get_indicators(open_prices_series, high_prices_series, low_prices_series, close_prices_series, volume_series)

    # Get latest candlestick patterns (specific pattern value for the last candle). Use .item() to ensure scalar.
    # Ensure to use .values[-1] or .iloc[-1].item() for scalar values
    latest_hammer = ta_lib.CDLHAMMER(open_prices_series, high_prices_series, low_prices_series, close_prices_series).values[-1]
    latest_engulfing = ta_lib.CDLENGULFING(open_prices_series, high_prices_series, low_prices_series, close_prices_series).values[-1]
    latest_doji = ta_lib.CDLDOJI(open_prices_series, high_prices_series, low_prices_series, close_prices_series).values[-1]

    # Get latest RSI, ADX, MACD, ATR from live_indicators
    latest_adx = live_indicators['adx'].iloc[-1]
    latest_rsi = live_indicators['rsi'].iloc[-1]
    latest_macd = live_indicators['macd'].iloc[-1]
    latest_atr = live_indicators['atr'].iloc[-1]

    # Display TA Library and YFinance data
    print("\n---------TA Library and YFinance Data---------\n")
    print(f"TA-Lib Data for {ticker_yf} ({interval_yf}):")
    print(f"Open: {open_prices_series.iloc[-1]:.5f}, Close: {close_prices_series.iloc[-1]:.5f}, High: {high_prices_series.iloc[-1]:.5f}, Low: {low_prices_series.iloc[-1]:.5f}, Volume: {volume_series.iloc[-1]}")
    print(f"Latest Hammer Pattern: {'Bullish' if latest_hammer > 0 else ('Bearish' if latest_hammer < 0 else 'None')}")
    print(f"Latest Engulfing Pattern: {'Bullish' if latest_engulfing > 0 else ('Bearish' if latest_engulfing < 0 else 'None')}")
    print(f"Latest Doji Pattern: {'Detected' if latest_doji != 0 else 'None'}")
    print(f"Latest RSI: {latest_rsi:.2f}, ADX: {latest_adx:.2f}, MACD: {latest_macd:.5f}, ATR: {latest_atr:.5f}")

    # Get data from TradingView
    try:
        handler = TA_Handler(
            symbol=ta_symbol_tv,
            screener=screener,
            exchange=exchange,
            interval=interval_yf # Use the same interval for TradingView for consistency
        )
        analysis = handler.get_analysis()

        # Current candle info from TradingView
        ta_open_prices = analysis.indicators["open"]
        ta_close_prices = analysis.indicators["close"]
        ta_high_prices = analysis.indicators["high"]
        ta_low_prices = analysis.indicators["low"]
        ta_volume = analysis.indicators["volume"]

        # Get latest RSI, ADX, MACD from TradingView
        ta_adx = analysis.indicators["ADX"]
        ta_rsi = analysis.indicators["RSI"]
        ta_macd = analysis.indicators["MACD.macd"]
        ta_atr = analysis.indicators.get("ATR", None) # ATR may not be available in TradingView

        ta_oscillators = analysis.oscillators
        ta_summary = analysis.summary

        # Display TradingView data
        print("\n---------TradingView Data---------\n")
        print(f"TradingView Data for {ta_symbol_tv} ({exchange}, {interval_yf}):")
        print(f"Open: {ta_open_prices:.5f}, Close: {ta_close_prices:.5f}, High: {ta_high_prices:.5f}, Low: {ta_low_prices:.5f}, Volume: {ta_volume}")
        print(f"RSI: {ta_rsi:.2f}, ADX: {ta_adx:.2f}, MACD: {ta_macd:.5f}, ATR: {ta_atr if ta_atr is not None else 'N/A'}")
        print("Summarised Recommendation (Oscillators):", ta_oscillators)
        print("Overall Technical Summary:", ta_summary)

    except Exception as e:
        print(f"\nError fetching TradingView data: {e}")
        # Set default/fallback values if TradingView data fails
        analysis = None # Set analysis to None if fetching fails
        ta_adx = np.nan
        ta_rsi = np.nan
        ta_macd = np.nan
        ta_atr = np.nan
        ta_oscillators = {"buy": 0, "sell": 0, "neutral": 0, "summary": "N/A"}
        ta_summary = {"BUY": 0, "SELL": 0, "NEUTRAL": 0, "RECOMMENDATION": "N/A"}

    # --- Market Regime Identification ---
    # Retrieve historical ADX from get_historical_data
    hist_adx_series, hist_candlesticks, hist_adx_state_str, \
        _, _, _, _, _ = \
        get_historical_data(ticker_yf, interval_yf, "1mo") # Use a longer period for historical context

    adx_trending_threshold = 25
    adx_strong_trend_threshold = 40

    reasons.clear() # Clear global reasons before adding new ones for market regime

    current_market_regime = "MIXED/UNCLEAR"

    # Determine market regime based on ADX values
    if not hist_adx_series.empty and len(hist_adx_series) > 0 and not np.isnan(hist_adx_series.iloc[-1]):
        ta_lib_historical_adx_value = hist_adx_series.iloc[-1]

        ta_lib_ranging = ta_lib_historical_adx_value < adx_trending_threshold
        trading_view_ranging = ta_adx < adx_trending_threshold if not np.isnan(ta_adx) else False

        ta_lib_trending = ta_lib_historical_adx_value > adx_strong_trend_threshold
        trading_view_trending = ta_adx > adx_strong_trend_threshold if not np.isnan(ta_adx) else False

        if ta_lib_ranging and trading_view_ranging:
            current_market_regime = "RANGING"
            reasons.append("Both TA Lib & TradingView ADX confirm RANGING market.")
        elif ta_lib_trending and trading_view_trending:
            current_market_regime = "TRENDING"
            reasons.append("Both TA Lib & TradingView ADX confirm TRENDING market.")
        elif ta_lib_ranging:
            current_market_regime = "RANGING (TA Lib Confirmed)"
            reasons.append("TA Lib ADX suggests RANGING market.")
        elif trading_view_ranging:
            current_market_regime = "RANGING (TradingView Confirmed)"
            reasons.append("TradingView ADX suggests RANGING market.")
        elif ta_lib_trending:
            current_market_regime = "TRENDING (TA Lib Confirmed)"
            reasons.append("TA Lib ADX suggests TRENDING market.")
        elif trading_view_trending:
            current_market_regime = "TRENDING (TradingView Confirmed)"
            reasons.append("TradingView ADX suggests TRENDING market.")
        else:
            reasons.append("Market direction is MIXED/UNCLEAR based on current ADX values.")
    else:
        reasons.append("Insufficient historical data to determine market regime.")
        current_market_regime = "INSUFFICIENT DATA"


    print(f"Current Market Regime: {current_market_regime}")
    print(f"Reasons for Market Regime: {', '.join(reasons)}")

    # --- Calculate Pivot Points ---
    # Use the HLC of the *previous* complete candle for pivot point calculation
    if len(data_cleaned) >= 2:
        prev_high = data_cleaned["high"].iloc[-2]
        prev_low = data_cleaned["low"].iloc[-2]
        prev_close = data_cleaned["close"].iloc[-2]
        pivot_points = calculate_pivot_points(prev_high, prev_low, prev_close)
    else:
        pivot_points = {'PP': np.nan, 'R1': np.nan, 'R2': np.nan, 'S1': np.nan, 'S2': np.nan}
        reasons.append("Not enough data for Pivot Point calculation.")

    return {
        "open_prices_series": open_prices_series,
        "high_prices_series": high_prices_series,
        "low_prices_series": low_prices_series,
        "close_prices_series": close_prices_series,
        "volume_series": volume_series,
        "live_indicators": live_indicators,
        "latest_hammer": latest_hammer,
        "latest_engulfing": latest_engulfing,
        "latest_doji": latest_doji,
        "tradingview_analysis": analysis, # Full analysis object from TradingView
        "current_market_regime": current_market_regime,
        "reasons": reasons, # Current reasons list (for market regime)
        "pivot_points": pivot_points # Add pivot points to the returned dictionary
    }

def send_to_gemini(prompt_text):
    """
    Sends the prompt to Gemini API with specific generation configuration.
    """
    print("\nSending data to Gemini for decision...")
    try:
        # Use a specific generation configuration for more deterministic output
        generation_config = {
            "temperature": 0.1,  # Lower temperature for less randomness
            "top_p": 0.1,        # Lower top_p for more focused output
        }
        response = gemini_model.generate_content(prompt_text, generation_config=generation_config)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None

def parse_gemini_response(json_response_string):
    """
    Parses the JSON response string from Gemini into a Python dictionary.
    """
    try:
        # Gemini might sometimes include markdown code block syntax. Remove it.
        cleaned_response = json_response_string.replace("```json\n", "").replace("\n```", "")
        parsed_data = json.loads(cleaned_response)
        return parsed_data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Gemini: {e}")
        print(f"Raw response from Gemini (start): {json_response_string[:500]}...") # Print a portion of the problematic string
        print(f"Raw response from Gemini (end): ...{json_response_string[-500:]}")
        return None


def main():
    """
    Main function to orchestrate data retrieval, indicator calculation,
    market regime identification, and then send all data to Gemini for decision making.
    """
    global reasons  # Ensure main can clear reasons if needed for multiple runs

    # Clear reasons from previous runs if any
    reasons.clear()

    # Define ticker and interval (these would typically come from configuration or arguments)
    # Using YFinance ticker for data fetching
    ticker_yf = "EURAUD=X"
    # Using TradingView symbol for TA_Handler
    ta_symbol_tv = "EURAUD"
    # Interval for both YFinance and TradingView TA
    interval_common = "5m"
    exchange = "FX_IDC"  # Example for TradingView
    screener = "forex"   # Example for TradingView
    yf_period_historical = "1mo" # Period for historical data for market regime

    # Retrieve current data and initial market regime analysis
    current_data_info = get_current_data(ticker_yf, interval_common, exchange, screener, ta_symbol_tv)

    if current_data_info is None:
        print("Failed to retrieve current data. Cannot make a trading decision.")
        return

    # --- Prepare Data for Gemini ---
    # Extract relevant latest values from current_data_info for the prompt
    latest_close_price = current_data_info["close_prices_series"].iloc[-1]
    latest_open_price = current_data_info["open_prices_series"].iloc[-1]
    latest_high_price = current_data_info["high_prices_series"].iloc[-1]
    latest_low_price = current_data_info["low_prices_series"].iloc[-1]
    
    # Get ATR value for calculations later
    latest_atr_ta_lib = _safe_float(current_data_info['live_indicators']['atr_ta_lib'][-1])
    
    print("The data sent to GEMINI for latest Candlestick ", latest_open_price, latest_close_price, latest_high_price, latest_low_price)
    # A helper function to safely get the latest value from a Series
    def get_latest_indicator_value(indicator_series, default_val=np.nan):
        # Handle cases where the series might be empty or contain NaN
        if not isinstance(indicator_series, pd.Series) or indicator_series.empty:
            return default_val
        last_val = indicator_series.iloc[-1]
        return last_val if not pd.isna(last_val) else default_val
    
    # Populate a dictionary with all data points for Gemini
    technical_data_for_gemini = {
        "ticker": ta_symbol_tv, # Use TradingView symbol for prompt
        "interval": interval_common, # Use common interval for prompt
        "current_open": f"{latest_open_price:.5f}",
        "current_high": f"{latest_high_price:.5f}",
        "current_low": f"{latest_low_price:.5f}",
        "current_close": f"{latest_close_price:.5f}",
        "current_volume": f"{current_data_info['volume_series'].iloc[-1]}",

        "market_regime": current_data_info["current_market_regime"],
        "market_regime_reasons": current_data_info["reasons"],  # Reasons for market regime from previous step

        # --- Live Indicators from TA Lib ---
        "rsi_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['rsi']):.2f}",
        "adx_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['adx']):.2f}",
        "macd_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['macd']):.5f}",
        "atr_ta_lib": f"{latest_atr_ta_lib:.5f}", # Use the actual float value here
        "stoch_k_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['stoch_k']):.2f}",
        "stoch_d_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['stoch_d']):.2f}",
        "cci_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['cci']):.2f}",
        "macd_diff_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['macd_diff']):.5f}",
        "aroon_up_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['aroon_up']):.2f}",
        "aroon_down_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['aroon_down']):.2f}",
        "pos_adx_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['pos_adx']):.2f}",
        "neg_adx_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['neg_adx']):.2f}",
        "sma_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['sma']):.5f}",
        "psar_up_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['psar_up'], 0.0):.5f}",  # PSAR can be 0 if no signal
        "psar_down_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['psar_down'], 0.0):.5f}",
        "ichimoku_a_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['ichimoku_a']):.5f}",
        "dpo_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['dpo']):.5f}",
        "stc_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['stc']):.2f}",
        "vortex_plus_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['vortex_plus']):.5f}",
        "vortex_minus_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['vortex_minus']):.5f}",
        "mass_index_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['mass_index']):.5f}",
        "bollinger_hband_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['bollinger_hband']):.5f}",
        "bollinger_lband_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['bollinger_lband']):.5f}",
        "keltner_hband_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['keltner_channel_hband']):.5f}",
        "keltner_lband_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['keltner_channel_lband']):.5f}",
        "ulcer_index_ta_lib": f"{get_latest_indicator_value(current_data_info['live_indicators']['ulcer_index']):.5f}",

        # --- Candlestick Patterns (from get_current_data) ---
        "hammer_pattern": "Bullish" if current_data_info['latest_hammer'] > 0 else ("Bearish" if current_data_info['latest_hammer'] < 0 else "None"),
        "engulfing_pattern": "Bullish" if current_data_info['latest_engulfing'] > 0 else ("Bearish" if current_data_info['latest_engulfing'] < 0 else "None"),
        "doji_pattern": "Detected" if current_data_info['latest_doji'] != 0 else "None",

        # --- TradingView Analysis (if available) ---
        "tv_rsi": f"{current_data_info['tradingview_analysis'].indicators.get('RSI', np.nan):.2f}" if current_data_info['tradingview_analysis'] else 'N/A',
        "tv_adx": f"{current_data_info['tradingview_analysis'].indicators.get('ADX', np.nan):.2f}" if current_data_info['tradingview_analysis'] else 'N/A',
        "tv_macd": f"{current_data_info['tradingview_analysis'].indicators.get('MACD.macd', np.nan):.5f}" if current_data_info['tradingview_analysis'] else 'N/A',
        "tv_atr": f"{current_data_info['tradingview_analysis'].indicators.get('ATR', np.nan):.5f}" if current_data_info['tradingview_analysis'] and current_data_info['tradingview_analysis'].indicators.get('ATR') is not None else 'N/A',
        "tv_oscillators_summary": current_data_info['tradingview_analysis'].oscillators.get('summary', 'N/A') if current_data_info['tradingview_analysis'] else 'N/A',
        "tv_overall_recommendation": current_data_info['tradingview_analysis'].summary.get('RECOMMENDATION', 'N/A') if current_data_info['tradingview_analysis'] else 'N/A',
        "tv_buy_signals_oscillators": current_data_info['tradingview_analysis'].oscillators.get('buy', 0) if current_data_info['tradingview_analysis'] else 0,
        "tv_sell_signals_oscillators": current_data_info['tradingview_analysis'].oscillators.get('sell', 0) if current_data_info['tradingview_analysis'] else 0,
        "tv_neutral_signals_oscillators": current_data_info['tradingview_analysis'].oscillators.get('neutral', 0) if current_data_info['tradingview_analysis'] else 0,
        "tv_buy_signals_summary": current_data_info['tradingview_analysis'].summary.get('BUY', 0) if current_data_info['tradingview_analysis'] else 0,
        "tv_sell_signals_summary": current_data_info['tradingview_analysis'].summary.get('SELL', 0) if current_data_info['tradingview_analysis'] else 0,
        "tv_neutral_signals_summary": current_data_info['tradingview_analysis'].summary.get('NEUTRAL', 0) if current_data_info['tradingview_analysis'] else 0,

        # --- Pivot Points ---
        "pivot_point": f"{current_data_info['pivot_points']['PP']:.5f}" if not np.isnan(current_data_info['pivot_points']['PP']) else 'N/A',
        "resistance_1": f"{current_data_info['pivot_points']['R1']:.5f}" if not np.isnan(current_data_info['pivot_points']['R1']) else 'N/A',
        "resistance_2": f"{current_data_info['pivot_points']['R2']:.5f}" if not np.isnan(current_data_info['pivot_points']['R2']) else 'N/A',
        "support_1": f"{current_data_info['pivot_points']['S1']:.5f}" if not np.isnan(current_data_info['pivot_points']['S1']) else 'N/A',
        "support_2": f"{current_data_info['pivot_points']['S2']:.5f}" if not np.isnan(current_data_info['pivot_points']['S2']) else 'N/A',
    }

    # Construct the prompt with all the data for Gemini
    prompt = f"""
Analyze the following market technical analysis data for {technical_data_for_gemini['ticker']} on the {technical_data_for_gemini['interval']} interval.
Provide a trading decision (BUY, SELL, or HOLD) along with detailed reasons and a summary of signal counts. Also include the market regime and pivot points.

Current Candle Data:
- Open: {technical_data_for_gemini['current_open']}
- High: {technical_data_for_gemini['current_high']}
- Low: {technical_data_for_gemini['current_low']}
- Close: {technical_data_for_gemini['current_close']}
- Volume: {technical_data_for_gemini['current_volume']}

Market Regime: {technical_data_for_gemini['market_regime']}
Reasons for Market Regime: {', '.join(technical_data_for_gemini['market_regime_reasons'])}

TA Lib Indicators:
- RSI: {technical_data_for_gemini['rsi_ta_lib']}
- ADX: {technical_data_for_gemini['adx_ta_lib']}
- MACD: {technical_data_for_gemini['macd_ta_lib']}
- ATR: {technical_data_for_gemini['atr_ta_lib']}
- Stochastic K: {technical_data_for_gemini['stoch_k_ta_lib']}
- Stochastic D: {technical_data_for_gemini['stoch_d_ta_lib']}
- CCI: {technical_data_for_gemini['cci_ta_lib']}
- MACD Diff: {technical_data_for_gemini['macd_diff_ta_lib']}
- Aroon Up: {technical_data_for_gemini['aroon_up_ta_lib']}
- Aroon Down: {technical_data_for_gemini['aroon_down_ta_lib']}
- Positive ADX: {technical_data_for_gemini['pos_adx_ta_lib']}
- Negative ADX: {technical_data_for_gemini['neg_adx_ta_lib']}
- SMA: {technical_data_for_gemini['sma_ta_lib']}
- PSAR Up: {technical_data_for_gemini['psar_up_ta_lib']}
- PSAR Down: {technical_data_for_gemini['psar_down_ta_lib']}
- Ichimoku A: {technical_data_for_gemini['ichimoku_a_ta_lib']}
- DPO: {technical_data_for_gemini['dpo_ta_lib']}
- STC: {technical_data_for_gemini['stc_ta_lib']}
- Vortex Plus: {technical_data_for_gemini['vortex_plus_ta_lib']}
- Vortex Minus: {technical_data_for_gemini['vortex_minus_ta_lib']}
- Mass Index: {technical_data_for_gemini['mass_index_ta_lib']}
- Bollinger HBand: {technical_data_for_gemini['bollinger_hband_ta_lib']}
- Bollinger LBand: {technical_data_for_gemini['bollinger_lband_ta_lib']}
- Keltner HBand: {technical_data_for_gemini['keltner_hband_ta_lib']}
- Keltner LBand: {technical_data_for_gemini['keltner_lband_ta_lib']}
- Ulcer Index: {technical_data_for_gemini['ulcer_index_ta_lib']}

Candlestick Patterns:
- Hammer: {technical_data_for_gemini['hammer_pattern']}
- Engulfing: {technical_data_for_gemini['engulfing_pattern']}
- Doji: {technical_data_for_gemini['doji_pattern']}

TradingView Analysis Summary:
- RSI (TV): {technical_data_for_gemini['tv_rsi']}
- ADX (TV): {technical_data_for_gemini['tv_adx']}
- MACD (TV): {technical_data_for_gemini['tv_macd']}
- ATR (TV): {technical_data_for_gemini['tv_atr']}
- Oscillators Summary (TV): {technical_data_for_gemini['tv_oscillators_summary']} (Buy: {technical_data_for_gemini['tv_buy_signals_oscillators']}, Sell: {technical_data_for_gemini['tv_sell_signals_oscillators']}, Neutral: {technical_data_for_gemini['tv_neutral_signals_oscillators']})
- Overall Recommendation (TV): {technical_data_for_gemini['tv_overall_recommendation']} (Buy: {technical_data_for_gemini['tv_buy_signals_summary']}, Sell: {technical_data_for_gemini['tv_sell_signals_summary']}, Neutral: {technical_data_for_gemini['tv_neutral_signals_summary']})

Calculated Pivot Points:
- Pivot Point (PP): {technical_data_for_gemini['pivot_point']}
- Resistance 1 (R1): {technical_data_for_gemini['resistance_1']}
- Resistance 2 (R2): {technical_data_for_gemini['resistance_2']}
- Support 1 (S1): {technical_data_for_gemini['support_1']}
- Support 2 (S2): {technical_data_for_gemini['support_2']}

Based on the above data, analyze the market conditions and provide a trading decision (BUY, SELL, or HOLD).
Summarize the signals.

Please provide your analysis and trading decision in the following exact JSON format:

```json
{{
  "Current Market Regime": "",
  "Reasons for Market Regime": [],
  "--- Final Trading Decision ---": {{
    "Decision": "",
    "Reasons": [],
    "Total Buy Signals": null,
    "Total Sell Signals": null,
    "Total Strong Buy Signals": null,
    "Total Strong Sell Signals": null
  }},
  "--- Calculated Pivot Points ---": {{
    "Pivot Point (PP)": null,
    "Resistance 1 (R1)": null,
    "Resistance 2 (R2)": null,
    "Support 1 (S1)": null,
    "Support 2 (S2)": null
  }}
}}"""

    # --- Send prompt to Gemini and process response ---
    print("\nSending prompt to Gemini for analysis...")
    gemini_response_text = send_to_gemini(prompt)

    if gemini_response_text:
        parsed_gemini_data = parse_gemini_response(gemini_response_text)

        if parsed_gemini_data:
            decision_section = parsed_gemini_data.get('--- Final Trading Decision ---', {})
            decision = decision_section.get('Decision', 'HOLD') # Default to HOLD if no decision

            estimated_entry_price = None
            estimated_stop_loss = None
            estimated_take_profit = None

            # Calculate Entry, Stop Loss, and Take Profit based on decision and ATR
            if decision == "BUY":
                estimated_entry_price = latest_close_price
                estimated_stop_loss = latest_close_price - (1 * latest_atr_ta_lib)
                estimated_take_profit = latest_close_price + (1.5 * latest_atr_ta_lib)
            elif decision == "SELL":
                estimated_entry_price = latest_close_price
                estimated_stop_loss = latest_close_price + (1.5 * latest_atr_ta_lib)
                estimated_take_profit = latest_close_price - (1 * latest_atr_ta_lib)
            else: # HOLD or other
                estimated_entry_price = latest_close_price # Still show current price
                estimated_stop_loss = "N/A"
                estimated_take_profit = "N/A"

            # Inject calculated values back into the parsed data
            decision_section['Estimated Entry Price'] = f"{estimated_entry_price:.5f}" if isinstance(estimated_entry_price, float) else estimated_entry_price
            decision_section['Estimated Stop Loss'] = f"{estimated_stop_loss:.5f}" if isinstance(estimated_stop_loss, float) else estimated_stop_loss
            decision_section['Estimated Take Profit'] = f"{estimated_take_profit:.5f}" if isinstance(estimated_take_profit, float) else estimated_take_profit

            # Print the structured output
            print("\n--- Gemini's Trading Decision ---")
            print(f"Current Market Regime: {parsed_gemini_data.get('Current Market Regime', 'N/A')}")
            print("Reasons for Market Regime:")
            for reason in parsed_gemini_data.get('Reasons for Market Regime', []):
                print(f"- {reason}")

            print("\n--- Final Trading Decision ---")
            print(f"Decision: {decision}")
            print("Reasons:")
            for reason in decision_section.get('Reasons', []):
                print(f"- {reason}")
            print(f"Estimated Entry Price: {decision_section.get('Estimated Entry Price', 'N/A')}")
            print(f"Estimated Stop Loss: {decision_section.get('Estimated Stop Loss', 'N/A')}")
            print(f"Estimated Take Profit: {decision_section.get('Estimated Take Profit', 'N/A')}")
            print(f"Total Buy Signals: {decision_section.get('Total Buy Signals', 'N/A')}")
            print(f"Total Sell Signals: {decision_section.get('Total Sell Signals', 'N/A')}")
            print(f"Total Strong Buy Signals: {decision_section.get('Total Strong Buy Signals', 'N/A')}")
            print(f"Total Strong Sell Signals: {decision_section.get('Total Strong Sell Signals', 'N/A')}")

            pivot_points_section = parsed_gemini_data.get('--- Calculated Pivot Points ---', {})
            print("\n--- Calculated Pivot Points ---")
            print(f"Pivot Point (PP): {pivot_points_section.get('Pivot Point (PP)', 'N/A')}")
            print(f"Resistance 1 (R1): {pivot_points_section.get('Resistance 1 (R1)', 'N/A')}")
            print(f"Resistance 2 (R2): {pivot_points_section.get('Resistance 2 (R2)', 'N/A')}")
            print(f"Support 1 (S1): {pivot_points_section.get('Support 1 (S1)', 'N/A')}")
            print(f"Support 2 (S2): {pivot_points_section.get('Support 2 (S2)', 'N/A')}")
        else:
            print("Failed to parse Gemini's response.")
    else:
        print("Gemini did not return a response.")

if __name__ == "__main__":
    main()
