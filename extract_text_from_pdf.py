from google import genai
from PyPDF2 import PdfReader

# === Setup Gemini Client ===
client = genai.Client(api_key=API_KEY)

# === PDF text extractor ===
def extract_text_from_pdf(filepath):
    reader = PdfReader(filepath)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text


def return_trading_info():
    # === Load Trading Strategies
    bollinger_bandit = extract_text_from_pdf("PDF's/Bollinger_Bandit_Trading_strategy.pdf")
    day_trading = extract_text_from_pdf("PDF's/DAY-TRADING.pdf")
    forex_cheat_sheet = extract_text_from_pdf("PDF's/Forex_Traders_Cheat_Sheet.pdf")
    fx_destroyer = extract_text_from_pdf("PDF's/FX_Destroyer.pdf")
    killer_patterns = extract_text_from_pdf("PDF's/Killer_Patterns.pdf")
    fibonacci_practical = extract_text_from_pdf("PDF's/Practical_Fibonacci_Methods_for_Forex_Trading.pdf")
    king_kelter_trading = extract_text_from_pdf("PDF's/King_Keltner_Trading_Strategy.pdf")
    super_combo_trading = extract_text_from_pdf("PDF's/Super_Combo_Day_Trading_Strategy.pdf")
    
    ict_1 = extract_text_from_pdf("PDF's/ICT-Trading-Strategy-1.pdf")
    ict_2 = extract_text_from_pdf("PDF's/ICT-Trading-Strategy-2.pdf")
    ict_3 = extract_text_from_pdf("PDF's/ICT-Trading-Strategy-3.pdf")
    # Shorten if needed to stay within token limits (example: first 4000 chars)
    combined_context = (ict_1 + ict_2 + ict_3)
    
    return combined_context
