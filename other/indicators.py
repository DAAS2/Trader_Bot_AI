import numpy
from ta import volatility, momentum, trend, volume, others
import yfinance as yf
import numpy as np
import pandas as pd
import talib as ta_lib 

ticker = "GBPJPY=X"

# retrieve data from Yahoo Finance
data = yf.download(ticker, interval="5m", period="1d", auto_adjust=True)

# fix column names to lowercase
data.columns = [col[0].lower() for col in data.columns]

# clear data from any Nan Values
data_cleaned = data.dropna()

# extract open, close, high, low prices from tuple as numpy arrays
open_prices = data_cleaned["open"].values.astype(np.float64)
close_prices = data_cleaned["close"].values.astype(np.float64)
high_prices = data_cleaned["high"].values.astype(np.float64)
low_prices = data_cleaned["low"].values.astype(np.float64)
volume = data_cleaned["volume"].values.astype(np.float64)

# convert to pd.series
open_prices = pd.Series(open_prices)
close_prices = pd.Series(close_prices)
high_prices = pd.Series(high_prices)
low_prices = pd.Series(low_prices)
volume = pd.Series(volume)

# CALCULATE MOMENTUM & TREND INDICATORS

# Measures the speed and change of price movements, identifying overbought or oversold conditions.
rsi = momentum.RSIIndicator(close_prices, window=14, fillna=True).rsi()

# Measures the strength of an uptrend and how long it has lasted.
aroon_up = trend.AroonIndicator(close_prices, low_prices, window=14, fillna=True).aroon_up()
# Measures the strength of a downtrend and how long it has lasted.
aroon_down = trend.AroonIndicator(close_prices, low_prices, window=14, fillna=True).aroon_down()

# Quantifies the strength of a trend, regardless of its direction.
adx = trend.ADXIndicator(high_prices, low_prices, close_prices, window=14, fillna=True).adx()

# Indicates the strength of downward price movement.
neg_adx = trend.ADXIndicator(high_prices, low_prices, close_prices, window=14, fillna=True).adx_neg()

# Indicates the strength of upward price movement.
pos_adx = trend.ADXIndicator(high_prices, low_prices, close_prices, window=14, fillna=True).adx_pos()

# Removes trend from price to identify cycles and overbought/oversold levels.
dpo = trend.DPOIndicator(close_prices, window=20, fillna=True).dpo()

# Part of the Ichimoku Cloud, providing a future resistance/support level.
ichimoku = trend.IchimokuIndicator(high_prices, low_prices, window1=9, window2=26, window3=52, fillna=True).ichimoku_a()

# Measures the deviation of price from its statistical average, identifying overbought/oversold conditions.
cci = trend.CCIIndicator(high_prices, low_prices, close_prices, window=20, fillna=True).cci()

# Shows the relationship between two moving averages of a security’s price to identify momentum and trend changes.
macd = trend.MACD(close_prices, window_slow=26, window_fast=12, window_sign=9, fillna=True).macd()

# Calculates the average price over a specific number of periods to smooth price data and identify trend.
sma = trend.SMAIndicator(close_prices, window=20, fillna=True).sma_indicator()

# Initializes the Stochastic Oscillator, which compares a closing price to its price range over time.
stoch = momentum.StochasticOscillator(high_prices, low_prices, close_prices, window=14, smooth_window=3, fillna=True)

# Represents the Stochastic %K line, showing the current price's position relative to its recent high-low range.
stoch_k = stoch.stoch()

# Represents the Stochastic %D line (a moving average of %K), used to signal crossovers with %K.
stoch_d = stoch.stoch_signal()

# Applies the Stochastic formula to RSI values, creating a more sensitive overbought/oversold indicator.
stoch_rsi = momentum.StochRSIIndicator(close_prices, window=14, smooth1=3, smooth2=3, fillna=True).stochrsi()

# Identifies potential trend reversals based on the narrowing and widening of price ranges.
mass_index = trend.MassIndex(high_prices, low_prices, window_fast=9, fillna=True).mass_index()

# Identifies potential reversal points and sets trailing stop-loss levels.
psar = trend.PSARIndicator(high_prices, low_prices, close_prices, step=0.02, max_step=0.2, fillna=True).psar()

# Indicates bearish PSAR points for downward trend tracking.
psar_down = trend.PSARIndicator(high_prices, low_prices, close_prices, step=0.02, max_step=0.2, fillna=True).psar_down()

# Indicates bullish PSAR points for upward trend tracking.
psar_up = trend.PSARIndicator(high_prices, low_prices, close_prices, step=0.02, max_step=0.2, fillna=True).psar_up()

# A leading indicator used to identify trend direction and potential reversal points.
stc = trend.STCIndicator(close_prices, window_slow=50, window_fast=23, fillna=True).stc()

# Measures the strength of price movement in both positive and negative directions.
vortex = trend.VortexIndicator(high_prices, low_prices, close_prices, window=14, fillna=True)


# CALCULATE VOLATILITY INDICATORS

# Measures market volatility by calculating the average true range of price movement.
atr = volatility.AverageTrueRange(high_prices, low_prices, close_prices, window=14, fillna=True).average_true_range()

# Measures market volatility using TA-Lib's Average True Range calculation.
atr_ta_lib = ta_lib.ATR(high_prices, low_prices, close_prices, timeperiod=14)

# Initializes Bollinger Bands, which define upper and lower price boundaries around a moving average.
bollinger = volatility.BollingerBands(close_prices, window=20, window_dev=2, fillna=True)

# Represents the upper Bollinger Band, often indicating overbought conditions or resistance.
bollinger_hband = bollinger.bollinger_hband()

# Represents the lower Bollinger Band, often indicating oversold conditions or support.
bollinger_lband = bollinger.bollinger_lband()

# Initializes Keltner Channels, which are volatility-based envelopes around price, similar to Bollinger Bands.
keltner = volatility.KeltnerChannel(high_prices, low_prices, close_prices, window=20, window_atr=10, fillna=True)

# Represents the upper Keltner Channel band, often acting as dynamic resistance.
keltner_channel_hband = keltner.keltner_channel_hband_indicator()

# Represents the lower Keltner Channel band, often acting as dynamic support.
keltner_channel_lband = keltner.keltner_channel_lband_indicator()

# Measures the depth and duration of price drawdowns, indicating investment risk.
ulcer = volatility.UlcerIndex(close_prices, fillna=True).ulcer_index()


print("Technical Indicators:")
print("RSI:", rsi)
print("ADX:", adx)
print("CCI:", cci)
print("MACD:", macd)
print("SMA:", sma)
print("Stochastic %K:", stoch_k)
print("Stochastic %D:", stoch_d)
print("Stochastic RSI:", stoch_rsi)

print("Momentum Indicators:")
print("Bollinger Bands High:", bollinger_hband)
print("Bollinger Bands Low:", bollinger_lband)
print("Keltner Channel High:", keltner_channel_hband)
print("Keltner Channel Low:", keltner_channel_lband)
print("Ulcer Index:", ulcer)

print("ATR:", atr)
print("ATR (ta-lib):", atr_ta_lib)

print("Trend Indicators:")
print("DPO:", dpo)
print("Ichimoku A:", ichimoku)
print("Mass Index:", mass_index)
print("PSAR:", psar)
print("PSAR Down:", psar_down)
print("PSAR Up:", psar_up)
print("STC:", stc)
print("Vortex Indicator:", vortex)
print("Aroon Up:", aroon_up)
print("Aroon Down:", aroon_down)
print("Negative ADX:", neg_adx)
print("Positive ADX:", pos_adx)