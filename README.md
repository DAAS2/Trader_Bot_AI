# 🤖 Trader_Bot_AI

An automated, AI-driven Forex trading bot that combines real-time market data from the OANDA API, advanced technical analysis, and Google's Gemini 2.0 Flash model to generate and execute trading decisions.

This project leverages a **Retrieval-Augmented Generation (RAG)** approach by ingesting trading strategy PDFs (e.g., ICT, Fibonacci, Bollinger Bandit) to ground AI decisions in proven trading methodologies.

---

## 🚀 Key Features

- **Real-Time Data Pipeline**  
  Fetches live 5-minute candlestick data using the OANDA v20 API.

- **Extensive Technical Analysis**  
  Computes 20+ indicators including RSI, MACD, ADX, Ichimoku Clouds, Keltner Channels, and Bollinger Bands.

- **Candlestick Pattern Recognition**  
  Detects patterns like Hammers, Engulfing patterns, and Dojis via TA-Lib.

- **Market Regime Detection**  
  Classifies markets as **Trending** or **Ranging** using ADX analysis.

- **TradingView Integration**  
  Cross-validates signals with TradingView indicators and summaries.

- **AI-Powered RAG Engine**  
  Uses PyPDF2 to ingest strategy PDFs and guide Gemini AI decisions.

- **Smart Risk Management**  
  Dynamically calculates Entry, Stop Loss (SL), and Take Profit (TP) using ATR.

- **Automated Email Alerts**  
  Sends real-time trade decisions and reasoning via SMTP.

---

## 🧰 Tech Stack

| Category              | Tools / Libraries |
|----------------------|-----------------|
| Core                 | Python 3 |
| Market Data          | oandapyV20, tradingview-ta |
| Technical Analysis   | ta, TA-Lib, numpy, pandas |
| AI / LLM             | google-generativeai (Gemini 2.0 Flash) |
| Document Parsing     | PyPDF2 |

---

## 📋 Prerequisites

Before running the bot, ensure you have:

- OANDA Live or Practice Account (Account ID & Access Token)
- Google Gemini API Key (via Google AI Studio)
- Gmail Account with App Password enabled (for alerts)
- C Compiler Tools (required for TA-Lib installation)

---
## 🛠️ Installation

### 2. Install TA-Lib (C Library)

TA-Lib must be installed before the Python wrapper.

**Windows:**
Download precompiled `.whl` from Christoph Gohlke's repo:

    pip install TA_Lib-*.whl

**macOS:**

    brew install ta-lib

**Linux:**

    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib
    ./configure
    make
    sudo make install

### 3. Install Python Dependencies

    pip install -r requirements.txt

Or manually:

    pip install numpy pandas oandapyV20 tradingview-ta ta PyPDF2 google-generativeai pytz

---

## ⚙️ Configuration

### 1. Environment Variables

    export OANDA_ACCESS_TOKEN="your_oanda_token_here"
    export OANDA_ACCOUNT_ID="your_oanda_account_id_here"

### 2. API Keys & Email Setup

Update `API_KEY` in:
* `extract_text_from_pdf.py`
* `oanda_ai.py`

Configure email credentials in:
* `send_email.py`

> ⚠️ **Never commit your API keys to GitHub**

### 3. Strategy PDFs

Create a directory:

    PDF's/

Add your trading strategy PDFs (ICT, Fibonacci, etc.) used by the RAG system.

---

## 💻 Usage

Run the main bot:

    python oanda_ai.py

The bot runs continuously, analyzing the market every 5 minutes.

---

## 📊 Example Output

    Fetching current data for GBP_JPY with granularity M5 (count=100) from OANDA...

    ---------TA Library and OANDA Data---------
    Open: 191.54000, Close: 191.56000, High: 191.58000, Low: 191.52000, Volume: 154
    Latest RSI: 54.20, ADX: 22.10, MACD: 0.01500, ATR: 0.05000

    --- Gemini's Trading Decision ---
    Current Market Regime: RANGING

    --- Final Trading Decision ---
    Decision: BUY
    Reasons:
    - Price bounced off S1 pivot point
    - ICT strategy indicates liquidity sweep
    - RSI remains above 50

    Entry: 191.56000  
    Stop Loss: 191.48500  
    Take Profit: 191.68500  

    Waiting for 5 minutes before next analysis...

---

## 📂 Project Structure

    Trader_Bot_AI/
    │── oanda_ai.py
    │── extract_text_from_pdf.py
    │── send_email.py
    │── requirements.txt
    │── PDF's/
    │── README.md

---

## 🧪 Troubleshooting

**TA-Lib installation errors**
* Ensure C compiler tools are installed
* Verify correct architecture (`.whl` for Windows)

**API connection issues**
* Check OANDA credentials
* Ensure internet access

**Email not sending**
* Confirm Gmail App Password is correct
* Check SMTP settings

---

## 🤝 Contributors

* **DAAS2** (Project Creator)

---

## ⚠️ Disclaimer

This software is for educational purposes only.

* Do NOT trade with money you cannot afford to lose
* Always test using demo/paper trading accounts first
* The authors assume no responsibility for financial losses
