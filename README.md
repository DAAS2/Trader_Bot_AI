🤖 Trader_Bot_AI
An automated, AI-driven Forex trading bot that combines real-time market data from the OANDA API, comprehensive technical analysis (TA-Lib, ta, TradingView-TA), and Google's Gemini 2.0 Flash model to generate executing trading decisions.

By utilizing a Retrieval-Augmented Generation (RAG) approach, this bot reads established trading strategy PDFs (e.g., ICT, Fibonacci, Bollinger Bandit) and uses them as the contextual foundation for Gemini's real-time BUY, SELL, or HOLD decisions.

🚀 Key Features
Real-Time Data Pipeline: Fetches live 5-minute granular candlestick data via the OANDA v20 API.

Extensive Technical Analysis: Calculates over 20 indicators including RSI, MACD, ADX, Ichimoku Clouds, Keltner Channels, and Bollinger Bands.

Candlestick Pattern Recognition: Automatically detects formations like Hammers, Engulfing patterns, and Dojis using TA-Lib.

Market Regime Detection: Analyzes historical ADX to classify the current market state as "Trending" or "Ranging".

TradingView Integration: Cross-references internal Python calculations with live TradingView oscillator and trend summaries.

AI-Powered RAG Engine: Ingests PDF trading strategies (ICT, Fibonacci, etc.) via PyPDF2 to ground the LLM's logic in established trading theory.

Smart Risk Management: Dynamically calculates Entry, Stop Loss (SL), and Take Profit (TP) targets based on the real-time Average True Range (ATR).

Automated Email Alerts: Sends immediate email notifications outlining Gemini's decisions, reasoning, and SL/TP levels via SMTP.

🧰 Tech Stack
Core: Python 3

Market Data: oandapyV20, tradingview-ta

Technical Analysis: ta, TA-Lib, numpy, pandas

AI / LLM: google-generativeai (Gemini 2.0 Flash)

Document Parsing: PyPDF2

📋 Prerequisites
Before running the bot, ensure you have the following set up:

An OANDA Live or Practice Account (for Account ID and Access Token).

A Google Gemini API Key (Generated via Google AI Studio).

A Gmail Account with App Passwords enabled (for sending email alerts).

C Compiler Tools installed on your system (required to build TA-Lib).

🛠️ Installation
1. Clone the repository

Bash
git clone https://github.com/DAAS2/Trader_Bot_AI.git
cd Trader_Bot_AI
2. Install TA-Lib (C-Library)
TA-Lib requires the underlying C library to be installed before installing the Python wrapper.

Windows: Download the pre-compiled .whl file from Christoph Gohlke's repository and run pip install TA_Lib-*.whl.

macOS: brew install ta-lib

Linux: Download the source from the TA-Lib website, configure, make, and make install.

3. Install Python Dependencies

Bash
pip install -r requirements.txt
# Alternatively: pip install numpy pandas oandapyV20 tradingview-ta ta PyPDF2 google-generativeai pytz
⚙️ Configuration
1. Environment Variables To keep your credentials secure, set your OANDA tokens as environment variables:

Bash
export OANDA_ACCESS_TOKEN="your_oanda_token_here"
export OANDA_ACCOUNT_ID="your_oanda_account_id_here"
2. API Keys & Email * Update the API_KEY in extract_text_from_pdf.py and oanda_ai.py. (Never commit your actual API keys to GitHub).

In send_email.py, update the email_sender and email_password variables.

3. Strategy PDFs Ensure you have a directory named PDF's/ in your root folder containing the required trading strategy documents referenced in return_trading_info().

💻 Usage
Start the bot by running the main script. It will run in a continuous loop, analyzing the market every 5 minutes and pausing in between intervals.

Bash
python oanda_ai.py
Example Console Output
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

Waiting for 5 minutes before the next analysis...
⚠️ Disclaimer
This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. The authors and contributors assume no responsibility for your trading results. Always test automated trading algorithms on a paper-trading or demo account before deploying them with real capital.
