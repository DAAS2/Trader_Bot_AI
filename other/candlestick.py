import pandas as pd
import yfinance as yf
import numpy as np
import talib as ta



ticker = "AUDCAD=X"

# retrieve data from Yahoo Finance
data = yf.download(ticker, interval="5m", period="1mo", auto_adjust=True)

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


# hammer pattern - 
hammer = ta.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)


engulfing = ta.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)


doji = ta.CDLDOJI(open_prices, high_prices, low_prices, close_prices)


two_crows = ta.CDL2CROWS(open_prices, high_prices, low_prices, close_prices)

breakaway = ta.CDLBREAKAWAY(open_prices, high_prices, low_prices, close_prices)

high_wave = ta.CDLHIGHWAVE(open_prices, high_prices, low_prices, close_prices)

three_black_crows = ta.CDL3BLACKCROWS(open_prices, high_prices, low_prices, close_prices)

three_inside = ta.CDL3INSIDE(open_prices, high_prices, low_prices, close_prices)

three_line = ta.CDL3LINESTRIKE(open_prices, high_prices, low_prices, close_prices)

three_outside = ta.CDL3OUTSIDE(open_prices, high_prices, low_prices, close_prices)

advance_block = ta.CDLADVANCEBLOCK(open_prices, high_prices, low_prices, close_prices)

counterattack = ta.CDLCOUNTERATTACK(open_prices, high_prices, low_prices, close_prices)

gap_side_white = ta.CDLGAPSIDESIDEWHITE(open_prices, high_prices, low_prices, close_prices)

kicking_bull = ta.CDLKICKINGBYLENGTH(open_prices, high_prices, low_prices, close_prices)

long_line = ta.CDLLONGLINE(open_prices, high_prices, low_prices, close_prices)

piercing_patter = ta.CDLPIERCING(open_prices, high_prices, low_prices, close_prices)

short_line = ta.CDLSHORTLINE(open_prices, high_prices, low_prices, close_prices)

seperating_lines = ta.CDLSEPARATINGLINES(open_prices, high_prices, low_prices, close_prices)

spinning_top = ta.CDLSPINNINGTOP(open_prices, high_prices, low_prices, close_prices)


adx = ta.ADX(high_prices, low_prices, close_prices, timeperiod=14)


print(f"\nTotal Hammer patterns detected: {np.sum(hammer != 0)}")
print(f"Total Engulfing patterns detected: {np.sum(engulfing != 0)}")
print(f"Total Doji patterns detected: {np.sum(doji != 0)}")
print(f"Total Two Crows patterns detected: {np.sum(two_crows != 0)}")
print(f"Total Breakaway patterns detected: {np.sum(breakaway != 0)}")
print(f"Total High Wave patterns detected: {np.sum(high_wave != 0)}")
print(f"Total Three Black Crows patterns detected: {np.sum(three_black_crows != 0)}")
print(f"Total Three Inside patterns detected: {np.sum(three_inside != 0)}")
print(f"Total Three Line patterns detected: {np.sum(three_line != 0)}")
print(f"Total Three Outside patterns detected: {np.sum(three_outside != 0)}")
print(f"Total Advance Block patterns detected: {np.sum(advance_block != 0)}")
print(f"Total Counterattack patterns detected: {np.sum(counterattack != 0)}")
print(f"Total Gap Side Side White patterns detected: {np.sum(gap_side_white != 0)}")
print(f"Total Kicking Bull patterns detected: {np.sum(kicking_bull != 0)}")
print(f"Total Long Line patterns detected: {np.sum(long_line != 0)}")
print(f"Total Piercing patterns detected: {np.sum(piercing_patter != 0)}")
print(f"Total Short Line patterns detected: {np.sum(short_line != 0)}")
print(f"Total Seperating Lines patterns detected: {np.sum(seperating_lines != 0)}")
print(f"Total Spinning Top patterns detected: {np.sum(spinning_top != 0)}")
print(f"Total ADX values calculated: {len(adx)}")
print(f"Total ADX values less than 25: {np.sum(adx < 25)}")
print(f"Total ADX values greater than 25: {np.sum(adx > 25)}")
print(f"Total ADX values greater than 50: {np.sum(adx > 50)}")
print(f"Total ADX values greater than 75: {np.sum(adx > 75)}")
print(f"Total ADX values greater than 100: {np.sum(adx > 100)}")


