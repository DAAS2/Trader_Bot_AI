import numpy as np
from ta import volatility, momentum, trend, volume, others
import yfinance as yf
import pandas as pd
import talib as ta_lib 
from tradingview_ta import TA_Handler, Interval, Exchange

# --- Configuration Variables ---
ticker = "USDJPY=X"
ta_symbol = "USDJPY"
screener = "forex"
exchange = "FX_IDC"
interval = "5m"
yfiananace_period = "1mo" # Period for historical data for TA-Lib calculations

# --- Global Variables for Decision and Reasoning ---
# These are initialized globally and updated by functions.
decision = None
reasons = [] # This list will accumulate all reasons for the decision.
estimated_entry = None
estimated_stop_loss = None
estimated_take_profit = None

# Global variables to store historical ADX state and candlestick patterns
# These are updated in get_historical_data and used by other functions.
historical_adx_state = None
historical_candlestick_patterns = None


def get_candlestick_patterns(open_prices, high_prices, low_prices, close_prices):
    """
    Calculates various TA-Lib candlestick patterns and returns their total detection counts.
    Note: This function returns counts over the entire series.
    For the latest candle's pattern value (100, -100, or 0),
    access the last element of the individual TA-Lib pattern output.
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

    # Count non-zero occurrences for each pattern
    patterns_counts = {name: np.sum(pattern != 0) for name, pattern in patterns_raw.items()}
    return patterns_counts

def get_historical_data(ticker, interval, yfiananace_period):
    """
    Fetches historical data and calculates historical ADX state and candlestick patterns.
    Updates global variables historical_adx_state and historical_candlestick_patterns.
    Returns the full historical ADX Series, historical candlestick pattern counts, and ADX state string.
    """
    global historical_adx_state, historical_candlestick_patterns
    
    data = yf.download(ticker, interval=interval, period=yfiananace_period, auto_adjust=True)
    data.columns = [col[0].lower() for col in data.columns] # Ensure column names are lowercase
    data_cleaned = data.dropna()
    
    # Convert to numpy arrays for TA-Lib compatibility first, then to Pandas Series
    # This ensures consistency in data types for TA-Lib.
    open_prices_np = data_cleaned["open"].values.astype(np.float64)
    close_prices_np = data_cleaned["close"].values.astype(np.float64)
    high_prices_np = data_cleaned["high"].values.astype(np.float64)
    low_prices_np = data_cleaned["low"].values.astype(np.float64)
    volume_np = data_cleaned["volume"].values.astype(np.float64)
    
    # Get historical candlestick patterns (counts over the period)
    historical_candlestick_patterns = get_candlestick_patterns(open_prices_np, high_prices_np, low_prices_np, close_prices_np)
    
    # Convert to pandas series for `ta` library (which expects Series)
    open_prices_series = pd.Series(open_prices_np)
    close_prices_series = pd.Series(close_prices_np)
    high_prices_series = pd.Series(high_prices_np)
    low_prices_series = pd.Series(low_prices_np)
    volume_series = pd.Series(volume_np)
    
    # Get historical ADX using `ta` library
    historical_adx = trend.ADXIndicator(high_prices_series, low_prices_series, close_prices_series, window=14, fillna=True).adx()
    
    # Determine historical market conditions based on ADX
    # Check if there's enough data for meaningful ADX calculation
    if len(historical_adx) > 0:
        if (historical_adx < 25).sum() / len(historical_adx) > 0.7:
            historical_adx_state = "RANGING"
        elif (historical_adx > 40).sum() / len(historical_adx) > 0.5:
            historical_adx_state  = "TRENDING"
        else:
            historical_adx_state  = "MIXED/UNCLEAR"
    else:
        historical_adx_state = "INSUFFICIENT DATA"

    return historical_adx, historical_candlestick_patterns, historical_adx_state, \
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
    indicators['atr_ta_lib'] = ta_lib.ATR(high_prices.values, low_prices.values, close_prices.values, timeperiod=14) # TA-Lib expects numpy arrays
    bollinger_obj = volatility.BollingerBands(close_prices, window=20, window_dev=2, fillna=True)
    indicators['bollinger_hband'] = bollinger_obj.bollinger_hband()
    indicators['bollinger_lband'] = bollinger_obj.bollinger_lband()
    keltner_obj = volatility.KeltnerChannel(high_prices, low_prices, close_prices, window=20, window_atr=10, fillna=True)
    indicators['keltner_channel_hband'] = keltner_obj.keltner_channel_hband_indicator()
    indicators['keltner_channel_lband'] = keltner_obj.keltner_channel_lband_indicator()
    indicators['ulcer_index'] = volatility.UlcerIndex(close_prices, fillna=True).ulcer_index()

    return indicators

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


def make_trading_decision(
    current_market_regime,
    live_indicators, # Contains full Series for indicators
    open_prices_series, # Full Series
    high_prices_series, # Full Series
    low_prices_series,  # Full Series
    close_prices_series, # Full Series
    pivot_points # Dictionary of calculated pivot points
):
    """
    Makes a trading decision (BUY, SELL, HOLD) based on market regime,
    various technical indicators, candlestick patterns, and pivot points.
    """
    global reasons, decision, estimated_entry, estimated_stop_loss, estimated_take_profit

    # --- Initialize Trade Signals ---
    decision = "HOLD"
    estimated_entry = None
    estimated_stop_loss = None
    estimated_take_profit = None

    buy_signals_count = 0
    sell_signals_count = 0
    strong_buy_signals_count = 0
    strong_sell_signals_count = 0

    # Ensure there are enough data points for previous values (at least 2)
    if len(close_prices_series) < 2:
        reasons.append("Not enough historical data for full indicator analysis (less than 2 data points).")
        return {
            "decision": "HOLD",
            "reasons": reasons,
            "estimated_entry": None,
            "estimated_stop_loss": None,
            "estimated_take_profit": None,
            "buy_signals": 0,
            "sell_signals": 0,
            "strong_buy_signals": 0,
            "strong_sell_signals": 0
        }

    # --- Extract Latest and Previous Values from Series ---
    latest_open_price = open_prices_series.iloc[-1]
    latest_close_price = close_prices_series.iloc[-1]
    previous_close_price = close_prices_series.iloc[-2]

    latest_high_price = high_prices_series.iloc[-1]
    latest_low_price = low_prices_series.iloc[-1]
    
    # Momentum Indicators
    latest_rsi = live_indicators['rsi'].iloc[-1]
    previous_rsi = live_indicators['rsi'].iloc[-2] if len(live_indicators['rsi']) >= 2 else np.nan

    latest_stoch_k = live_indicators['stoch_k'].iloc[-1]
    previous_stoch_k = live_indicators['stoch_k'].iloc[-2] if len(live_indicators['stoch_k']) >= 2 else np.nan

    latest_stoch_d = live_indicators['stoch_d'].iloc[-1]
    previous_stoch_d = live_indicators['stoch_d'].iloc[-2] if len(live_indicators['stoch_d']) >= 2 else np.nan

    latest_macd_diff = live_indicators['macd_diff'].iloc[-1]
    previous_macd_diff = live_indicators['macd_diff'].iloc[-2] if len(live_indicators['macd_diff']) >= 2 else np.nan
    
    latest_cci = live_indicators['cci'].iloc[-1]
    previous_cci = live_indicators['cci'].iloc[-2] if len(live_indicators['cci']) >= 2 else np.nan

    # Trend Indicators
    latest_adx = live_indicators['adx'].iloc[-1]
    latest_pos_adx = live_indicators['pos_adx'].iloc[-1]
    latest_neg_adx = live_indicators['neg_adx'].iloc[-1]

    latest_aroon_up = live_indicators['aroon_up'].iloc[-1]
    latest_aroon_down = live_indicators['aroon_down'].iloc[-1]

    latest_sma = live_indicators['sma'].iloc[-1]
    previous_sma = live_indicators['sma'].iloc[-2] if len(live_indicators['sma']) >= 2 else np.nan

    # PSAR values are directly used for current trend
    latest_psar_up = live_indicators['psar_up'].iloc[-1]
    latest_psar_down = live_indicators['psar_down'].iloc[-1]

    latest_ichimoku_a = live_indicators['ichimoku_a'].iloc[-1]

    latest_dpo = live_indicators['dpo'].iloc[-1]
    previous_dpo = live_indicators['dpo'].iloc[-2] if len(live_indicators['dpo']) >= 2 else np.nan

    latest_stc = live_indicators['stc'].iloc[-1]
    previous_stc = live_indicators['stc'].iloc[-2] if len(live_indicators['stc']) >= 2 else np.nan

    latest_vortex_plus = live_indicators['vortex_plus'].iloc[-1]
    previous_vortex_plus = live_indicators['vortex_plus'].iloc[-2] if len(live_indicators['vortex_plus']) >= 2 else np.nan

    latest_vortex_minus = live_indicators['vortex_minus'].iloc[-1]
    previous_vortex_minus = live_indicators['vortex_minus'].iloc[-2] if len(live_indicators['vortex_minus']) >= 2 else np.nan

    latest_mass_index = live_indicators['mass_index'].iloc[-1]
    previous_mass_index = live_indicators['mass_index'].iloc[-2] if len(live_indicators['mass_index']) >= 2 else np.nan


    # Volatility Indicators
    latest_atr = live_indicators['atr'].iloc[-1]
    latest_bollinger_hband = live_indicators['bollinger_hband'].iloc[-1]
    latest_bollinger_lband = live_indicators['bollinger_lband'].iloc[-1]
    latest_keltner_channel_hband = live_indicators['keltner_channel_hband'].iloc[-1]
    latest_keltner_channel_lband = live_indicators['keltner_channel_lband'].iloc[-1]

    # Candlestick Patterns (re-calculate for the latest candle's specific pattern value)
    latest_hammer_val = ta_lib.CDLHAMMER(open_prices_series, high_prices_series, low_prices_series, close_prices_series).iloc[-1]
    latest_engulfing_val = ta_lib.CDLENGULFING(open_prices_series, high_prices_series, low_prices_series, close_prices_series).iloc[-1]
    latest_doji_val = ta_lib.CDLDOJI(open_prices_series, high_prices_series, low_prices_series, close_prices_series).iloc[-1]


    # --- Decision-Making Logic (Based on Provided Notes) ---

    # --- A. Momentum Indicators ---

    # RSI
    if current_market_regime == "RANGING":
        if latest_rsi < 30:
            reasons.append("RSI oversold (Ranging market)")
            buy_signals_count += 1
        elif latest_rsi > 70:
            reasons.append("RSI overbought (Ranging market)")
            sell_signals_count += 1

    # Stochastic Oscillator
    if current_market_regime == "RANGING":
        if not np.isnan(previous_stoch_k) and not np.isnan(previous_stoch_d):
            if latest_stoch_k < 20 and latest_stoch_k > latest_stoch_d and previous_stoch_k <= previous_stoch_d:
                reasons.append("Stochastic K crossing above D in oversold (Ranging market)")
                buy_signals_count += 1
            elif latest_stoch_k > 80 and latest_stoch_k < latest_stoch_d and previous_stoch_k >= previous_stoch_d:
                reasons.append("Stochastic K crossing below D in overbought (Ranging market)")
                sell_signals_count += 1

    # MACD (using macd_diff/histogram)
    if current_market_regime == "TRENDING":
        if not np.isnan(previous_macd_diff):
            if latest_macd_diff > 0 and previous_macd_diff <= 0:
                reasons.append("MACD bullish crossover (Trending market)")
                buy_signals_count += 1
                strong_buy_signals_count += 1
            elif latest_macd_diff < 0 and previous_macd_diff >= 0:
                reasons.append("MACD bearish crossover (Trending market)")
                sell_signals_count += 1
                strong_sell_signals_count += 1

    # CCI
    if current_market_regime == "RANGING":
        if latest_cci < -100:
            reasons.append("CCI oversold (Ranging market)")
            buy_signals_count += 1
        elif latest_cci > 100:
            reasons.append("CCI overbought (Ranging market)")
            sell_signals_count += 1


    # --- B. Trend Indicators ---

    # ADX, Pos_ADX, Neg_ADX
    if latest_adx > 25:
        if latest_pos_adx > latest_neg_adx:
            reasons.append("ADX confirms uptrend")
            buy_signals_count += 1
            if latest_adx > 40:
                reasons.append("ADX indicates strong uptrend")
                strong_buy_signals_count += 1
        elif latest_neg_adx > latest_pos_adx:
            reasons.append("ADX confirms downtrend")
            sell_signals_count += 1
            if latest_adx > 40:
                reasons.append("ADX indicates strong downtrend")
                strong_sell_signals_count += 1

    # Aroon Indicator
    if latest_aroon_up > latest_aroon_down and latest_aroon_up > 80:
        reasons.append("Aroon indicates emerging uptrend")
        buy_signals_count += 1
    elif latest_aroon_down > latest_aroon_up and latest_aroon_down > 80:
        reasons.append("Aroon indicates emerging downtrend")
        sell_signals_count += 1

    # SMA
    if not np.isnan(previous_sma):
        if latest_close_price > latest_sma and previous_close_price <= previous_sma:
            reasons.append("Price crosses above SMA (Bullish)")
            buy_signals_count += 1
        elif latest_close_price < latest_sma and previous_close_price >= previous_sma:
            reasons.append("Price crosses below SMA (Bearish)")
            sell_signals_count += 1

    # PSAR
    if latest_psar_up > 0: # PSAR dot is below price (uptrend signal)
        reasons.append("PSAR confirms uptrend (dots below price)")
        buy_signals_count += 1
        strong_buy_signals_count += 1
    elif latest_psar_down > 0: # PSAR dot is above price (downtrend signal)
        reasons.append("PSAR confirms downtrend (dots above price)")
        sell_signals_count += 1
        strong_sell_signals_count += 1

    # Ichimoku (Simplified interpretation of Leading Span A)
    if latest_close_price > latest_ichimoku_a:
        reasons.append("Price above Ichimoku Leading Span A (Bullish)")
        buy_signals_count += 1
    elif latest_close_price < latest_ichimoku_a:
        reasons.append("Price below Ichimoku Leading Span A (Bearish)")
        sell_signals_count += 1

    # DPO
    if not np.isnan(previous_dpo):
        if latest_dpo > 0 and previous_dpo <= 0:
            reasons.append("DPO turning positive (Bullish cycle)")
            buy_signals_count += 1
        elif latest_dpo < 0 and previous_dpo >= 0:
            reasons.append("DPO turning negative (Bearish cycle)")
            sell_signals_count += 1

    # STC (Schaff Trend Cycle)
    if not np.isnan(previous_stc):
        if latest_stc > 25 and previous_stc <= 25:
            reasons.append("STC crosses above 25 (Buy signal)")
            buy_signals_count += 1
        elif latest_stc < 75 and previous_stc >= 75:
            reasons.append("STC crosses below 75 (Sell signal)")
            sell_signals_count += 1
        
        if latest_stc <= 5 and previous_stc > 5 and current_market_regime != "TRENDING":
            reasons.append("STC strong buy signal (from oversold)")
            strong_buy_signals_count += 1
        elif latest_stc >= 95 and previous_stc < 95 and current_market_regime != "TRENDING":
            reasons.append("STC strong sell signal (from overbought)")
            strong_sell_signals_count += 1

    # Vortex Indicator
    if not np.isnan(previous_vortex_plus) and not np.isnan(previous_vortex_minus):
        if latest_vortex_plus > latest_vortex_minus and previous_vortex_plus <= previous_vortex_minus:
            reasons.append("Vortex Indicator bullish crossover")
            buy_signals_count += 1
        elif latest_vortex_minus > latest_vortex_plus and previous_vortex_minus <= previous_vortex_plus:
            reasons.append("Vortex Indicator bearish crossover")
            sell_signals_count += 1

    # Mass Index (Simplified - usually needs more complex logic for confirmation)
    if not np.isnan(latest_mass_index) and not np.isnan(previous_mass_index):
        if previous_mass_index > 27 and latest_mass_index < 26.5:
            reasons.append("Mass Index indicating potential trend reversal (widening range followed by contraction).")

    # --- C. Volatility Indicators ---

    # Bollinger Bands
    if current_market_regime == "RANGING":
        if latest_close_price <= latest_bollinger_lband:
            reasons.append("Price at or below Bollinger Lower Band (Ranging market)")
            buy_signals_count += 1
        elif latest_close_price >= latest_bollinger_hband:
            reasons.append("Price at or above Bollinger Upper Band (Ranging market)")
            sell_signals_count += 1

    # Keltner Channels
    if current_market_regime == "RANGING":
        if latest_close_price <= latest_keltner_channel_lband:
            reasons.append("Price at or below Keltner Lower Channel (Ranging market)")
            buy_signals_count += 1
        elif latest_close_price >= latest_keltner_channel_hband:
            reasons.append("Price at or above Keltner Upper Channel (Ranging market)")
            sell_signals_count += 1


    # --- D. Candlestick Patterns ---
    
    # Hammer
    if latest_hammer_val > 0: # Bullish Hammer
        reasons.append("Bullish Hammer candlestick detected")
        buy_signals_count += 1
        if current_market_regime == "RANGING": # Stronger in ranging/reversal context
            strong_buy_signals_count += 1
    elif latest_hammer_val < 0: # Bearish Hammer / Hanging Man
        reasons.append("Bearish Hammer/Hanging Man candlestick detected")
        sell_signals_count += 1
        if current_market_regime == "RANGING":
            strong_sell_signals_count += 1

    # Engulfing
    if latest_engulfing_val > 0: # Bullish Engulfing
        reasons.append("Bullish Engulfing candlestick detected")
        buy_signals_count += 1
        strong_buy_signals_count += 1
    elif latest_engulfing_val < 0: # Bearish Engulfing
        reasons.append("Bearish Engulfing candlestick detected")
        sell_signals_count += 1
        strong_sell_signals_count += 1

    # Doji
    if latest_doji_val != 0: # Doji (indecision)
        reasons.append("Doji candlestick (indecision) detected")
        # Further logic needed if you want to use Doji as a signal based on context


    # --- E. Support and Resistance (Pivot Points) ---
    # Define a "nearness" threshold based on ATR
    near_threshold = latest_atr * 0.5 if not np.isnan(latest_atr) else 0.0005 # Fallback if ATR is NaN

    # Check if pivot points are valid before using them
    if not np.isnan(pivot_points['PP']):
        if abs(latest_close_price - pivot_points['S1']) < near_threshold and latest_close_price > previous_close_price:
            reasons.append(f"Price bouncing off S1 support ({pivot_points['S1']:.5f})")
            buy_signals_count += 1
        elif abs(latest_close_price - pivot_points['S2']) < near_threshold and latest_close_price > previous_close_price:
            reasons.append(f"Price bouncing off S2 support ({pivot_points['S2']:.5f})")
            buy_signals_count += 1
        
        if abs(latest_close_price - pivot_points['R1']) < near_threshold and latest_close_price < previous_close_price:
            reasons.append(f"Price rejecting R1 resistance ({pivot_points['R1']:.5f})")
            sell_signals_count += 1
        elif abs(latest_close_price - pivot_points['R2']) < near_threshold and latest_close_price < previous_close_price:
            reasons.append(f"Price rejecting R2 resistance ({pivot_points['R2']:.5f})")
            sell_signals_count += 1
    else:
        reasons.append("Pivot points not available for S/R analysis.")


    # --- Step 4 (Cont.): Consolidation of Signals ---
    if strong_buy_signals_count > strong_sell_signals_count and buy_signals_count > 1:
        decision = "BUY"
        reasons.append("Overall strong bullish signals.")
    elif strong_sell_signals_count > strong_buy_signals_count and sell_signals_count > 1:
        decision = "SELL"
        reasons.append("Overall strong bearish signals.")
    elif buy_signals_count > sell_signals_count and buy_signals_count >= 2:
        decision = "BUY"
        reasons.append("Predominance of bullish signals.")
    elif sell_signals_count > buy_signals_count and sell_signals_count >= 2:
        decision = "SELL"
        reasons.append("Predominance of bearish signals.")
    else:
        decision = "HOLD"
        reasons.append("No clear strong signals or mixed signals.")

    # --- Step 5: Risk Management (Estimation) ---
    if decision == "BUY":
        estimated_entry = latest_close_price
        # Ensure latest_atr is not NaN or zero before calculation
        if not np.isnan(latest_atr) and latest_atr > 0:
            estimated_stop_loss = latest_close_price - (1 * latest_atr)
            estimated_take_profit = latest_close_price + (1.5 * latest_atr) # Example 1:2 Risk-Reward
            reasons.append(f"Calculated Stop-Loss ({estimated_stop_loss:.5f}) and Take-Profit ({estimated_take_profit:.5f}) based on ATR.")
        else:
            reasons.append("ATR not available or zero for SL/TP calculation.")

    elif decision == "SELL":
        estimated_entry = latest_close_price
        if not np.isnan(latest_atr) and latest_atr > 0:
            estimated_stop_loss = latest_close_price + (1 * latest_atr)
            estimated_take_profit = latest_close_price - (1.5 * latest_atr) # Example 1:2 Risk-Reward
            reasons.append(f"Calculated Stop-Loss ({estimated_stop_loss:.5f}) and Take-Profit ({estimated_take_profit:.5f}) based on ATR.")
        else:
            reasons.append("ATR not available or zero for SL/TP calculation.")

    return {
        "decision": decision,
        "reasons": reasons,
        "estimated_entry": estimated_entry,
        "estimated_stop_loss": estimated_stop_loss,
        "estimated_take_profit": estimated_take_profit,
        "buy_signals": buy_signals_count,
        "sell_signals": sell_signals_count,
        "strong_buy_signals": strong_buy_signals_count,
        "strong_sell_signals": strong_sell_signals_count
    }


def get_current_data(ticker, interval, exchange, screener, ta_symbol):
    """
    Fetches current day's data, calculates live indicators and patterns,
    determines the current market regime, and calculates pivot points.
    Returns a dictionary containing all processed data and analysis.
    """
    global reasons # Declare global to modify the `reasons` list

    # Fetch data for at least the last two candles to get previous day's HLC for pivot points
    # Changed period to "2d" to ensure we have previous day's data for pivot points.
    data = yf.download(ticker, interval=interval, period="2d", auto_adjust=True)
    data.columns = [col[0].lower() for col in data.columns] # Ensure column names are lowercase
    data_cleaned = data.dropna()
    
    # Check if data_cleaned is empty or has less than 2 rows for pivot point calculation
    if data_cleaned.empty or len(data_cleaned) < 2:
        print(f"Not enough data retrieved for {ticker} with interval {interval} and period '2d'. Need at least 2 candles for pivot points.")
        return None # Indicate failure to retrieve data

    # Extract prices and volume as Pandas Series for `ta` library
    open_prices_series = pd.Series(data_cleaned["open"].values.astype(np.float64))
    close_prices_series = pd.Series(data_cleaned["close"].values.astype(np.float64))
    high_prices_series = pd.Series(data_cleaned["high"].values.astype(np.float64))
    low_prices_series = pd.Series(data_cleaned["low"].values.astype(np.float64))
    volume_series = pd.Series(data_cleaned["volume"].values.astype(np.float64))
    
    # Get live indicators (full Series for all indicators)
    live_indicators = get_indicators(open_prices_series, high_prices_series, low_prices_series, close_prices_series, volume_series)
    
    # Get latest candlestick patterns (specific pattern value for the last candle)
    latest_hammer = ta_lib.CDLHAMMER(open_prices_series, high_prices_series, low_prices_series, close_prices_series).iloc[-1]
    latest_engulfing = ta_lib.CDLENGULFING(open_prices_series, high_prices_series, low_prices_series, close_prices_series).iloc[-1]
    latest_doji = ta_lib.CDLDOJI(open_prices_series, high_prices_series, low_prices_series, close_prices_series).iloc[-1]
    
    # Get latest RSI, ADX, MACD, ATR from live_indicators
    latest_adx = live_indicators['adx'].iloc[-1]
    latest_rsi = live_indicators['rsi'].iloc[-1]
    latest_macd = live_indicators['macd'].iloc[-1]
    latest_atr = live_indicators['atr'].iloc[-1]
    
    # Display TA Library and YFinance data
    print("\n---------TA Library and YFinance Data---------\n")
    print(f"TA-Lib Data for {ticker} ({exchange}, {interval}):")
    print(f"Open: {open_prices_series.iloc[-1]:.5f}, Close: {close_prices_series.iloc[-1]:.5f}, High: {high_prices_series.iloc[-1]:.5f}, Low: {low_prices_series.iloc[-1]:.5f}, Volume: {volume_series.iloc[-1]}")
    print(f"Latest Hammer Pattern: {'Bullish' if latest_hammer > 0 else ('Bearish' if latest_hammer < 0 else 'None')}")
    print(f"Latest Engulfing Pattern: {'Bullish' if latest_engulfing > 0 else ('Bearish' if latest_engulfing < 0 else 'None')}")
    print(f"Latest Doji Pattern: {'Detected' if latest_doji != 0 else 'None'}")
    print(f"Latest RSI: {latest_rsi:.2f}, ADX: {latest_adx:.2f}, MACD: {latest_macd:.5f}, ATR: {latest_atr:.5f}")
    
    # Get data from TradingView
    try:
        handler = TA_Handler(
            symbol=ta_symbol,
            screener=screener,
            exchange=exchange,
            interval=interval
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
        print(f"TradingView Data for {ticker} ({exchange}, {interval}):")
        print(f"Open: {ta_open_prices:.5f}, Close: {ta_close_prices:.5f}, High: {ta_high_prices:.5f}, Low: {ta_low_prices:.5f}, Volume: {ta_volume}")
        print(f"RSI: {ta_rsi:.2f}, ADX: {ta_adx:.2f}, MACD: {ta_macd:.5f}, ATR: {ta_atr if ta_atr is not None else 'N/A'}")
        print("Summarised Recommendation (Oscillators):", ta_oscillators)
        print("Overall Technical Summary:", ta_summary)

    except Exception as e:
        print(f"\nError fetching TradingView data: {e}")
        # Set default/fallback values if TradingView data fails
        ta_adx = np.nan
        ta_rsi = np.nan
        ta_macd = np.nan
        ta_atr = np.nan
        ta_oscillators = {"buy": 0, "sell": 0, "neutral": 0, "summary": "N/A"}
        ta_summary = {"BUY": 0, "SELL": 0, "NEUTRAL": 0, "RECOMMENDATION": "N/A"}
    
    # --- Market Regime Identification ---
    # Retrieve historical ADX from get_historical_data
    # Note: This call also updates global historical_adx_state and historical_candlestick_patterns
    hist_adx_series, hist_candlesticks, hist_adx_state_str, \
        _, _, _, _, _ = \
        get_historical_data(ticker, interval, yfiananace_period) # Unpack only what's needed for market regime here

    adx_trending_threshold = 25
    adx_strong_trend_threshold = 40

    current_market_regime = "MIXED/UNCLEAR"
    reasons.clear() # Clear global reasons before adding new ones for market regime

    # Check if historical_adx_series has enough data points
    if not hist_adx_series.empty and len(hist_adx_series) > 0:
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

def main():
    """
    Main function to orchestrate data retrieval, indicator calculation,
    market regime identification, and trading decision making.
    """
    global reasons # Ensure main can clear reasons if needed for multiple runs

    # Clear reasons from previous runs if any
    reasons.clear() 
    
    # Retrieve current data and initial market regime analysis
    current_data_info = get_current_data(ticker, interval, exchange, screener, ta_symbol)
    
    if current_data_info is None:
        print("Failed to retrieve current data. Cannot make a trading decision.")
        return

    # Make the trading decision based on all collected data
    final_decision_info = make_trading_decision(
        current_data_info["current_market_regime"],
        current_data_info["live_indicators"],
        current_data_info["open_prices_series"],
        current_data_info["high_prices_series"],
        current_data_info["low_prices_series"],
        current_data_info["close_prices_series"],
        current_data_info["pivot_points"] # Pass pivot points here
    )

    print("\n--- Final Trading Decision ---")
    print(f"Decision: {final_decision_info['decision']}")
    print("Reasons:")
    for reason in final_decision_info['reasons']:
        print(f"- {reason}")
    print(f"Estimated Entry Price: {final_decision_info['estimated_entry']:.5f}" if final_decision_info['estimated_entry'] is not None else "Estimated Entry Price: N/A")
    print(f"Estimated Stop Loss: {final_decision_info['estimated_stop_loss']:.5f}" if final_decision_info['estimated_stop_loss'] is not None else "Estimated Stop Loss: N/A")
    print(f"Estimated Take Profit: {final_decision_info['estimated_take_profit']:.5f}" if final_decision_info['estimated_take_profit'] is not None else "Estimated Take Profit: N/A")
    print(f"Total Buy Signals: {final_decision_info['buy_signals']}")
    print(f"Total Sell Signals: {final_decision_info['sell_signals']}")
    print(f"Total Strong Buy Signals: {final_decision_info['strong_buy_signals']}")
    print(f"Total Strong Sell Signals: {final_decision_info['strong_sell_signals']}")

    # Display calculated Pivot Points
    print("\n--- Calculated Pivot Points ---")
    if not np.isnan(current_data_info['pivot_points']['PP']):
        print(f"Pivot Point (PP): {current_data_info['pivot_points']['PP']:.5f}")
        print(f"Resistance 1 (R1): {current_data_info['pivot_points']['R1']:.5f}")
        print(f"Resistance 2 (R2): {current_data_info['pivot_points']['R2']:.5f}")
        print(f"Support 1 (S1): {current_data_info['pivot_points']['S1']:.5f}")
        print(f"Support 2 (S2): {current_data_info['pivot_points']['S2']:.5f}")
    else:
        print("Pivot points could not be calculated due to insufficient data.")


if __name__ == "__main__":
    main()
