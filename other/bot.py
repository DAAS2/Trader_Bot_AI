import numpy
from ta import volatility, momentum, trend, volume, others
import yfinance as yf
import numpy as np
import pandas as pd
import talib as ta_lib 
from tradingview_ta import TA_Handler, Interval, Exchange

ticker = "AUDCAD=X"
ta_symbol = "AUDCAD"
screener = "forex"
exchange = "FX_IDC"
interval = "5m"
yfiananace_period = "1mo"

# initialise decision and reasoning variables
decision = None
reasons = []
estimated_entry = None
estimated_stop_loss = None
estimated_take_profit = None



def get_candlestick_patterns(open_prices, high_prices, low_prices, close_prices):
    """
    Calculates various TA-Lib candlestick patterns and returns their total detection counts.
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
    
    global historical_adx_state, historical_candlestick_patterns
    
    data = yf.download(ticker, interval=interval, period=yfiananace_period, auto_adjust=True)
    data.columns = [col[0].lower() for col in data.columns]
    data_cleaned = data.dropna()
    
    # make panda series
    open_prices = data_cleaned["open"].values.astype(np.float64)
    close_prices = data_cleaned["close"].values.astype(np.float64)
    high_prices = data_cleaned["high"].values.astype(np.float64)
    low_prices = data_cleaned["low"].values.astype(np.float64)
    volume = data_cleaned["volume"].values.astype(np.float64)
    
    # get historical candlestick patterns 
    historical_candlestick_patterns = get_candlestick_patterns(open_prices, high_prices, low_prices, close_prices)
    
    # convert to pandas series
    open_prices = pd.Series(open_prices)
    close_prices = pd.Series(close_prices)
    high_prices = pd.Series(high_prices)
    low_prices = pd.Series(low_prices)
    volume = pd.Series(volume)
    
    # get historical ADX
    historical_adx = trend.ADXIndicator(high_prices, low_prices, close_prices, window=14, fillna=True).adx()
    
    # Determine market conditions
    if (historical_adx < 25).sum() / len(historical_adx) > 0.7:
        historical_adx_state = "RANGING"
    elif (historical_adx > 40).sum() / len(historical_adx) > 0.5:
        historical_adx_state  = "TRENDING"
    else:
        historical_adx_state  = "MIXED/UNCLEAR"
    
    return historical_adx, historical_candlestick_patterns, historical_adx_state

def get_indicators(open_prices, high_prices, low_prices, close_prices, volume):
    indicators = {}

    # Momentum Indicators
    indicators['rsi'] = momentum.RSIIndicator(close_prices, window=14, fillna=True).rsi()
    indicators['stoch_k'] = momentum.StochasticOscillator(high_prices, low_prices, close_prices, window=14, smooth_window=3, fillna=True).stoch()
    indicators['stoch_d'] = momentum.StochasticOscillator(high_prices, low_prices, close_prices, window=14, smooth_window=3, fillna=True).stoch_signal()
    indicators['stoch_rsi'] = momentum.StochRSIIndicator(close_prices, window=14, smooth1=3, smooth2=3, fillna=True).stochrsi()

    # Trend Indicators
    indicators['aroon_up'] = trend.AroonIndicator(close_prices, low_prices, window=14, fillna=True).aroon_up()
    indicators['aroon_down'] = trend.AroonIndicator(close_prices, low_prices, window=14, fillna=True).aroon_down()
    indicators['adx'] = trend.ADXIndicator(high_prices, low_prices, close_prices, window=14, fillna=True).adx()
    indicators['neg_adx'] = trend.ADXIndicator(high_prices, low_prices, close_prices, window=14, fillna=True).adx_neg()
    indicators['pos_adx'] = trend.ADXIndicator(high_prices, low_prices, close_prices, window=14, fillna=True).adx_pos()
    indicators['dpo'] = trend.DPOIndicator(close_prices, window=20, fillna=True).dpo()
    indicators['ichimoku_a'] = trend.IchimokuIndicator(high_prices, low_prices, window1=9, window2=26, window3=52, fillna=True).ichimoku_a()
    indicators['ichimoku_b'] = trend.IchimokuIndicator(high_prices, low_prices, window1=9, window2=26, window3=52, fillna=True).ichimoku_b()
    indicators['cci'] = trend.CCIIndicator(high_prices, low_prices, close_prices, window=20, fillna=True).cci()
    indicators['macd'] = trend.MACD(close_prices, window_slow=26, window_fast=12, window_sign=9, fillna=True).macd()
    indicators['macd_signal'] = trend.MACD(close_prices, window_slow=26, window_fast=12, window_sign=9, fillna=True).macd_signal() # Added macd_signal for completeness
    indicators['macd_diff'] = trend.MACD(close_prices, window_slow=26, window_fast=12, window_sign=9, fillna=True).macd_diff()
    indicators['sma'] = trend.SMAIndicator(close_prices, window=20, fillna=True).sma_indicator()
    indicators['mass_index'] = trend.MassIndex(high_prices, low_prices, window_fast=9, fillna=True).mass_index()
    indicators['psar'] = trend.PSARIndicator(high_prices, low_prices, close_prices, step=0.02, max_step=0.2, fillna=True).psar()
    indicators['psar_down'] = trend.PSARIndicator(high_prices, low_prices, close_prices, step=0.02, max_step=0.2, fillna=True).psar_down()
    indicators['psar_up'] = trend.PSARIndicator(high_prices, low_prices, close_prices, step=0.02, max_step=0.2, fillna=True).psar_up()
    indicators['stc'] = trend.STCIndicator(close_prices, window_slow=50, window_fast=23, fillna=True).stc()
    
    # Vortext indicators
    vortex_obj = trend.VortexIndicator(high_prices, low_prices, close_prices, window=14, fillna=True)
    indicators['vortex_plus'] = vortex_obj.vortex_indicator_pos()
    indicators['vortex_minus'] = vortex_obj.vortex_indicator_neg()


    # Volatility Indicators
    indicators['atr'] = volatility.AverageTrueRange(high_prices, low_prices, close_prices, window=14, fillna=True).average_true_range()
    indicators['atr_ta_lib'] = ta_lib.ATR(high_prices, low_prices, close_prices, timeperiod=14) 
    bollinger_obj = volatility.BollingerBands(close_prices, window=20, window_dev=2, fillna=True)
    indicators['bollinger_hband'] = bollinger_obj.bollinger_hband()
    indicators['bollinger_lband'] = bollinger_obj.bollinger_lband()
    keltner_obj = volatility.KeltnerChannel(high_prices, low_prices, close_prices, window=20, window_atr=10, fillna=True)
    indicators['keltner_channel_hband'] = keltner_obj.keltner_channel_hband_indicator()
    indicators['keltner_channel_lband'] = keltner_obj.keltner_channel_lband_indicator()
    indicators['ulcer_index'] = volatility.UlcerIndex(close_prices, fillna=True).ulcer_index()

    return indicators


def get_current_data(ticker, interval, exchange, screener, ta_symbol):
    data = yf.download(ticker, interval=interval, period="1d", auto_adjust=True)
    data.columns = [col[0].lower() for col in data.columns]
    data_cleaned = data.dropna()
    
    # extract latest open, close, high, low prices and volume 
    open_prices = pd.Series(data_cleaned["open"].values.astype(np.float64))
    close_prices = pd.Series(data_cleaned["close"].values.astype(np.float64))
    high_prices = pd.Series(data_cleaned["high"].values.astype(np.float64))
    low_prices = pd.Series(data_cleaned["low"].values.astype(np.float64))
    volume = pd.Series(data_cleaned["volume"].values.astype(np.float64))
    
    
    # get live candlestick patterns and indicators
    live_indicators = get_indicators(open_prices, high_prices, low_prices, close_prices, volume)
    live_candlestick_patterns = get_candlestick_patterns(open_prices, high_prices, low_prices, close_prices)
    
    # get latest RSI, ADX, MACD, ATR and Candlestick patterns
    latest_adx = live_indicators['adx'].iloc[-1]
    latest_rsi = live_indicators['rsi'].iloc[-1]
    latest_macd = live_indicators['macd'].iloc[-1]
    latest_atr = live_indicators['atr'].iloc[-1]
    
    # get latest candlestick patterns
    latest_hammer = ta_lib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices).values[-1]


    latest_engulfing = ta_lib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices).values[-1]


    latest_doji = ta_lib.CDLDOJI(open_prices, high_prices, low_prices, close_prices).values[-1]
    
    # display ta library and yfinance data
    print("\n---------TA Library and YFinance Data---------\n")
    print(f"TA-Lib Data for {ticker} ({exchange}, {interval}):")
    print(f"Open: {open_prices.iloc[-1]}, Close: {close_prices.iloc[-1]}, High: {high_prices.iloc[-1]}, Low: {low_prices.iloc[-1]}, Volume: {volume.iloc[-1]}")
    print(f"Latest Hammer Pattern: {'Bullish' if latest_hammer > 0 else ('Bearish' if latest_hammer < 0 else 'None')}")
    print(f"Latest Engulfing Pattern: {'Bullish' if latest_engulfing > 0 else ('Bearish' if latest_engulfing < 0 else 'None')}")
    print(f"Latest Doji Pattern: {'Detected' if latest_doji != 0 else 'None'}")
    
    # display latest RSI, ADX, MACD, ATR values
    print(f"Latest RSI: {latest_rsi}, ADX: {latest_adx}, MACD: {latest_macd}, ATR: {latest_atr}")
    
    # get data from tradingview 
    handler = TA_Handler(
        symbol=ta_symbol,
        screener=screener,
        exchange=exchange,
        interval=interval
    )
    
    
    analysis = handler.get_analysis()
    
    # Current candle info
    ta_open_prices = analysis.indicators["open"]
    ta_close_prices = analysis.indicators["close"]
    ta_high_prices = analysis.indicators["high"]
    ta_low_prices = analysis.indicators["low"]
    ta_volume = analysis.indicators["volume"]
    
    # get latest RSI, ADX, MACD
    ta_adx = analysis.indicators["ADX"]
    ta_rsi = analysis.indicators["RSI"]
    ta_macd = analysis.indicators["MACD.macd"]
    ta_atr = analysis.indicators.get("ATR", None)  # ATR may not be available in TradingView
    
    # all indicators from tradingview
    ta_indicators = analysis.indicators
    
    # get recommendation and overall decision
    ta_oscillators = analysis.oscillators
    ta_summary = analysis.summary
    
    # display Tradingview data
    print("\n---------TradingView Data---------\n")
    print(f"TradingView Data for {ticker} ({exchange}, {interval}):")
    print(f"Open: {ta_open_prices}, Close: {ta_close_prices}, High: {ta_high_prices}, Low: {ta_low_prices}, Volume: {ta_volume}")
    print(f"RSI: {ta_rsi}, ADX: {ta_adx}, MACD: {ta_macd}, ATR: {ta_atr}")
    print("Summarised Recommendation", ta_oscillators)
    print("Overall Technical Summary:", ta_summary)
    
    hist_adx_series, hist_candlesticks, hist_adx_state_str = get_historical_data(ticker, interval, yfiananace_period)

    adx_trending = 25
    adx_strong_trend = 40

    # ACCESS THE LAST ELEMENT OF THE SERIES:
    ta_lib_historical_adx_value = hist_adx_series.iloc[-1]

    ta_ranging = ta_lib_historical_adx_value < adx_trending
    trading_view_ranging = ta_adx < adx_trending

    ta_trending = ta_lib_historical_adx_value > adx_strong_trend
    trading_view_trending = ta_adx > adx_strong_trend

    current_market_regime = "MIXED/UNCLEAR" 

    if ta_ranging and trading_view_ranging:
        current_market_regime = "RANGING"
        reasons.append("Both TA Lib & TradingView ADX confirm RANGING market.")
        
    elif ta_trending and trading_view_trending:
        current_market_regime = "TRENDING"
        reasons.append("Both TA Lib & TradingView ADX confirm TRENDING market.")
        
    elif ta_ranging: # If TA Lib is ranging and TradingView isn't confirming a trend
        current_market_regime = "RANGING (TA Lib Confirmed)"
        reasons.append("TA Lib ADX suggests RANGING market.")
        
    elif trading_view_ranging: # If TradingView is ranging and TA Lib isn't confirming a trend
        current_market_regime = "RANGING (TradingView Confirmed)"
        reasons.append("TradingView ADX suggests RANGING market.")
        
    elif ta_trending: # If TA Lib is trending and TradingView isn't confirming a range
        current_market_regime = "TRENDING (TA Lib Confirmed)"
        reasons.append("TA Lib ADX suggests TRENDING market.")
        
    elif trading_view_trending: # If TradingView is trending and TA Lib isn't confirming a range
        current_market_regime = "TRENDING (TradingView Confirmed)"
        reasons.append("TradingView ADX suggests TRENDING market.")
        
    else:
        reasons.append("Market direction is MIXED/UNCLEAR based on current ADX values.")


    print(f"Current Market Regime: {current_market_regime}")
    print(f"Reasons for Market Regime: {', '.join(reasons)}")
    
    # Decision making based on indicators
    
    
    
    
def main():
    # Retrieve historical data
    
    Current_data_info = get_current_data(ticker, interval, exchange, screener, ta_symbol)
    
    print(Current_data_info)
    
if __name__ == "__main__":
    main()
    
