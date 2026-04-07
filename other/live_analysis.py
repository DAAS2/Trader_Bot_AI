from tradingview_ta import TA_Handler, Interval, Exchange
import json
import ta as technical_analysis


handler = TA_Handler(
    symbol="AUDCAD",
    screener="forex",
    exchange="FX_IDC",
    interval="5m", 
)

symbol="AUDCAD",
screener="forex"
exchange="FX_IDC"
interval="5m"

print(f"Stock Info: {symbol} {exchange} {interval} ")

analysis = handler.get_analysis()
print("Price of stock when candle starts in interval")
print(analysis.indicators["open"])
print("Price of stock when candle ends in interval")
print(analysis.indicators["close"])

print("Highest price of stock in that candle interval")
print(analysis.indicators["high"])
print("Lowest price of stock in that candle interval")
print(analysis.indicators["low"])
print("Relative Strength Index represents the speed and price of a stock's price change")
print(analysis.indicators["RSI"])
print("ADX Value")
print(analysis.indicators["ADX"])
print("MACD Value")
print(analysis.indicators["MACD.macd"])
print("All indicators of Stock")
print(analysis.indicators)
print("Summarised recommendation of BUY, SELL OR NEUTRAL")
print(analysis.oscillators)

print("Overall technical recommendation of BUY, SELL OR NEUTRAL")
print(analysis.summary)

print("Volume of Stock in 5 minute interval")
print(analysis.indicators["volume"])

