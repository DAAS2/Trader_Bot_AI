📈 Trader Bot AI (OANDA + Gemini 2.0)
An automated, AI-driven Forex trading bot that leverages the OANDA API for live market data, comprehensive technical analysis libraries (TA-Lib, TradingView-TA), and Google's Gemini 2.0 Flash model to execute complex trading decisions.

Unlike standard algorithmic bots, this system utilizes a Retrieval-Augmented Generation (RAG) approach by reading established trading strategy PDFs (e.g., ICT, Fibonacci, Bollinger Bandit) and using them as the contextual foundation for Gemini's real-time buy/sell/hold decisions.

✨ Features
Live Market Data: Fetches real-time, 5-minute granular candlestick data using the OANDA v20 API.

Extensive Technical Analysis: Calculates over 20 indicators including RSI, MACD, ADX, Ichimoku Clouds, Keltner Channels, and Bollinger Bands using Python's ta library and TA-Lib.

Candlestick Pattern Recognition: Automatically detects formations like Hammers, Engulfing patterns, and Dojis.

Market Regime Detection: Analyzes historical ADX data to classify the current market state as "Trending" or "Ranging" to inform strategy selection.

TradingView Integration: Cross-references internal calculations with live TradingView oscillator and trend summaries.

AI-Powered Strategy Ingestion: Uses PyPDF2 to extract logic from established trading strategy PDFs (ICT, Fibonacci, etc.) and feeds them to Gemini.

Smart Risk Management: Calculates dynamic Entry, Stop Loss (SL), and Take Profit (TP) targets based on real-time Average True Range (ATR).

Automated Email Alerts: Sends immediate email notifications outlining Gemini's trading decisions, reasoning, and SL/TP levels.

🛠️ Prerequisites
Before running this bot, you will need:

Python 3.8+

An OANDA Live or Practice Account (for Account ID and Access Token)

A Google Gemini API Key (for Gemini 2.0 Flash)

A Gmail Account (with App Passwords enabled for sending email alerts)

C compiler tools (required to build TA-Lib for Python)

📦 Installation
Clone the repository:

Bash
git clone https://github.com/DAAS2/Trader_Bot_AI.git
cd Trader_Bot_AI
Install TA-Lib:
TA-Lib requires the underlying C library to be installed on your system before installing the Python wrapper.

Windows: Download the pre-compiled .whl file from Christoph Gohlke's repository and run pip install TA_Lib-*.whl.

macOS: brew install ta-lib

Linux: Download the source from the TA-Lib website, configure, make, and make install.

Install Python Dependencies:

Bash
pip install numpy pandas oandapyV20 tradingview-ta ta PyPDF2 google-generativeai pytz
⚙️ Configuration
Environment Variables:
You must set your OANDA credentials as environment variables to keep them secure. Do not hardcode them in the script.

Bash
export OANDA_ACCESS_TOKEN="your_oanda_token_here"
export OANDA_ACCOUNT_ID="your_oanda_account_id_here"
(On Windows, use set or configure them in your system environment variables).

API Keys & Email:

Update the API_KEY in extract_text_from_pdf.py and oanda_ai.py with your secure Gemini API key.

In send_email.py, update email_sender and email_password (use an App Password, not your standard account password).

PDF Knowledge Base:
Ensure you have a directory named PDF's/ in the root folder containing your trading strategy documents as referenced in return_trading_info().

🚀 Usage
Run the main script to start the bot. It will run in a continuous loop, analyzing the market every 5 minutes.

Bash
python oanda_ai.py
Example Output
Plaintext
Fetching current data for GBP_JPY with granularity M5 (count=100) from OANDA...

---------TA Library and OANDA Data---------
Open: 191.54000, Close: 191.56000, High: 191.58000, Low: 191.52000, Volume: 154
Latest RSI: 54.20, ADX: 22.10, MACD: 0.01500, ATR: 0.05000

--- Gemini's Trading Decision ---
Current Market Regime: RANGING
Reasons for Market Regime:
- Both TA Lib & TradingView ADX confirm RANGING market.

--- Final Trading Decision ---
Decision: BUY
Reasons:
- Price has bounced off S1 pivot point.
- ICT Strategy 2 indicates a liquidity sweep on the 5m timeframe.
- RSI is resetting while remaining above 50.
Estimated Entry Price: 191.56000
Estimated Stop Loss: 191.48500
Estimated Take Profit: 191.68500
⚠️ Disclaimer
This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. The authors and contributors assume no responsibility for your trading results. Always test automated trading algorithms on a demo account before deploying them with real capital.
