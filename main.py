"""
Gold Tracker & AI Market Sentiment - Stateful Data Pipeline
============================================================
Stateful data pipeline & Telegram notification orchestrator:
1. Macroeconomic Ingestion via yfinance (Global Gold XAU/USD & USD/IDR).
2. Playwright Web Scraper (Logam Mulia, UBS Gold, Galeri 24).
3. AI Macroeconomic & News Sentiment Analyzer (requests + feedparser + Google GenAI).
4. Telegram Dispatcher (python-telegram-bot v20+).
5. Stateful Storage via Google Sheets (gspread + service account info).
"""

import asyncio
import html
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output encoding on Windows and Linux consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import gspread
import pytz
import yfinance as yf
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from telegram import Bot
from telegram.constants import ParseMode

from ai_news import FinancialNewsAnalyzer
from scraper import GoldPriceScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("GoldTrackerPipeline")

# Load environment variables
load_dotenv()

JAKARTA_TZ = pytz.timezone("Asia/Jakarta")

# Standard fallback summary when AI generation fails or encounters quota limits
FALLBACK_AI_SUMMARY = "⚠️ AI Summary tidak tersedia saat ini. Mengirim data harga mentah."

# Google Sheets API Scopes
GSPRED_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]


def fetch_macro_data() -> Dict[str, Any]:
    """
    Fetches real-time global gold prices (GC=F / XAUUSD=X) and USD/IDR exchange rate (IDR=X)
    using the yfinance library.
    """
    logger.info("Fetching macroeconomic indicators via yfinance...")
    data: Dict[str, Any] = {
        "global_gold_usd": None,
        "usd_idr": None,
        "global_gold_str": "Tidak tersedia",
        "usd_idr_str": "Tidak tersedia",
    }

    # 1. Fetch Global Gold Price (GC=F: COMEX Gold Futures, fallback: XAUUSD=X)
    for ticker_symbol in ["GC=F", "XAUUSD=X", "XAU=X"]:
        try:
            t = yf.Ticker(ticker_symbol)
            price = getattr(t.fast_info, "last_price", None)
            if price is None:
                hist = t.history(period="1d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])

            if price is not None and price > 0:
                data["global_gold_usd"] = round(float(price), 2)
                data["global_gold_str"] = f"${data['global_gold_usd']:,.2f} / troy oz"
                logger.info(f"Global Gold Price ({ticker_symbol}): {data['global_gold_str']}")
                break
        except Exception as e:
            logger.warning(f"Could not fetch gold price using ticker {ticker_symbol}: {e}")

    # 2. Fetch USD/IDR Currency Rate (IDR=X)
    try:
        t_idr = yf.Ticker("IDR=X")
        idr_price = getattr(t_idr.fast_info, "last_price", None)
        if idr_price is None:
            hist_idr = t_idr.history(period="1d")
            if not hist_idr.empty:
                idr_price = float(hist_idr["Close"].iloc[-1])

        if idr_price is not None and idr_price > 0:
            data["usd_idr"] = round(float(idr_price), 2)
            data["usd_idr_str"] = f"Rp {data['usd_idr']:,.0f}"
            logger.info(f"USD/IDR Exchange Rate (IDR=X): {data['usd_idr_str']}")
    except Exception as e:
        logger.warning(f"Could not fetch USD/IDR rate using ticker IDR=X: {e}")

    return data


def append_to_google_sheet(row_data: List[Any]) -> bool:
    """
    Appends a new daily record to Google Sheets using gspread.
    Securely parses credentials from GCP_CREDENTIALS_JSON environment variable.
    """
    creds_json = os.getenv("GCP_CREDENTIALS_JSON")
    if not creds_json:
        logger.warning("GCP_CREDENTIALS_JSON is not configured in environment. Skipping Google Sheets update.")
        return False

    spreadsheet_identifier = os.getenv("SPREADSHEET_ID") or os.getenv("SPREADSHEET_NAME", "Gold Tracker Data")

    try:
        logger.info("Connecting to Google Sheets via Service Account credentials...")
        service_account_info = json.loads(creds_json)
        credentials = Credentials.from_service_account_info(
            service_account_info,
            scopes=GSPRED_SCOPES
        )
        gc = gspread.authorize(credentials)

        # Open by ID if alphanumeric key provided, otherwise open by title
        if spreadsheet_identifier.startswith("http"):
            sheet = gc.open_by_url(spreadsheet_identifier).sheet1
        elif len(spreadsheet_identifier) > 25 and "/" not in spreadsheet_identifier and " " not in spreadsheet_identifier:
            try:
                sheet = gc.open_by_key(spreadsheet_identifier).sheet1
            except Exception:
                sheet = gc.open(spreadsheet_identifier).sheet1
        else:
            sheet = gc.open(spreadsheet_identifier).sheet1

        # Check if header exists; if empty, initialize headers
        existing_values = sheet.row_values(1)
        if not existing_values:
            header_row = [
                "Date",
                "Antam Buy",
                "Antam Sell",
                "UBS Buy",
                "Galeri 24 Buy",
                "Global Gold Price (USD)",
                "USD/IDR"
            ]
            sheet.append_row(header_row, value_input_option="USER_ENTERED")
            logger.info("Initialized Google Sheet header row.")

        # Append data row
        sheet.append_row(row_data, value_input_option="USER_ENTERED")
        logger.info(f"Successfully appended daily row to Google Sheet: {row_data}")
        return True

    except Exception as exc:
        logger.error(f"Failed to append row to Google Sheets: {exc}", exc_info=True)
        return False


def clean_markdown_to_html(text: str) -> str:
    """
    Converts markdown formatting (bold, italic, bullets) from the AI summary
    into clean, valid Telegram HTML while escaping raw HTML entities.
    """
    escaped = html.escape(text)

    # Convert **bold** to <b>bold</b>
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)

    # Convert *italic* or _italic_ to <i>italic</i>
    escaped = re.sub(r"(?<!\w)\*([^*]+?)\*(?!\w)", r"<i>\1</i>", escaped)
    escaped = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", r"<i>\1</i>", escaped)

    # Convert markdown bullet points '* ' or '- ' to unicode bullet '• '
    escaped = re.sub(r"^\s*[\*\-]\s+", "• ", escaped, flags=re.MULTILINE)

    return escaped.strip()


def format_telegram_report(
    scraped_data: Dict[str, Any],
    ai_summary: str,
    macro_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Formats the scraped gold prices, macro indicators, and AI sentiment summary
    into a clean, mobile-friendly Telegram HTML message.
    """
    now_wib = datetime.now(JAKARTA_TZ)
    date_str = now_wib.strftime("%A, %d %B %Y | %H:%M WIB")

    data = scraped_data.get("data", {})
    lm = data.get("logam_mulia", {})
    ubs = data.get("ubs_gold", {})
    g24 = data.get("galeri24", {})

    def price_val(val: Any) -> str:
        return f"<code>{val}</code>" if val else "<i>Tidak tersedia</i>"

    # Logam Mulia details
    lm_buy = price_val(lm.get("buy_price"))
    lm_tax = f" (+PPh: <code>{lm.get('buy_price_tax')}</code>)" if lm.get("buy_price_tax") else ""
    lm_sell = price_val(lm.get("sell_price")) if lm.get("sell_price") else "<i>Dalam konfirmasi butik</i>"

    # UBS Gold details
    ubs_buy = price_val(ubs.get("buy_price"))
    ubs_sell = price_val(ubs.get("sell_price"))

    # Galeri 24 details
    g24_buy = price_val(g24.get("buy_price"))
    g24_sell = price_val(g24.get("sell_price"))

    # Macro details
    global_gold = macro_data.get("global_gold_str", "Tidak tersedia") if macro_data else "Tidak tersedia"
    usd_idr = macro_data.get("usd_idr_str", "Tidak tersedia") if macro_data else "Tidak tersedia"

    formatted_summary = clean_markdown_to_html(ai_summary)

    message_lines = [
        "📊 <b>DAILY GOLD & MACRO REPORT</b>",
        f"🗓 <i>{date_str}</i>",
        "",
        "🌐 <b>GLOBAL INDICATORS</b>",
        f"• XAU/USD : <code>{global_gold}</code>",
        f"• USD/IDR : <code>{usd_idr}</code>",
        "",
        "💰 <b>LOCAL PRICE (1 GR)</b>",
        "<b>1. ANTAM</b>",
        f"B: {lm_buy}{lm_tax} | S: {lm_sell}",
        "<b>2. UBS Gold</b>",
        f"B: {ubs_buy} | S: {ubs_sell}",
        "<b>3. Galeri 24</b>",
        f"B: {g24_buy} | S: {g24_sell}",
        "",
        "📰 <b>MACRO ANALYSIS & SENTIMENT</b>",
        f"<blockquote>{formatted_summary}</blockquote>",
        "",
        "<i>🤖 Automated by Gold Tracker Bot</i>"
    ]

    return "\n".join(message_lines)


async def send_daily_update() -> None:
    """
    Main stateful data pipeline execution:
    1. Fetches local scraped gold prices and macro indicators concurrently.
    2. Generates Gemini macroeconomic sentiment analysis.
    3. Dispatches formatted report to Telegram.
    4. Persists the daily record to Google Sheets statefully.
    """
    telegram_token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("CHAT_ID")

    if not telegram_token or not chat_id:
        raise ValueError("Missing TELEGRAM_TOKEN or CHAT_ID in environment variables.")

    logger.info("Initializing bot, scraper, and news analyzer components...")
    bot = Bot(token=telegram_token)
    scraper = GoldPriceScraper()
    analyzer = FinancialNewsAnalyzer()

    # Verify bot connectivity
    bot_info = await bot.get_me()
    logger.info(f"Connected to Telegram as @{bot_info.username} ({bot_info.first_name})")

    loop = asyncio.get_running_loop()

    try:
        # Step 1: Run Web Scraping and Macro Data Fetching concurrently
        logger.info("Running concurrent tasks: Playwright Scraper & yfinance Macro Fetcher...")
        scraper_task = scraper.scrape_all()
        macro_task = loop.run_in_executor(None, fetch_macro_data)

        scraped_data, macro_data = await asyncio.gather(scraper_task, macro_task)
        logger.info("Completed scraping and macro data collection.")

        # Step 2: Run Gemini Macroeconomic Sentiment Analysis with fetched numbers
        local_prices = scraped_data.get("data", {})

        async def safe_fetch_ai_summary() -> str:
            try:
                summary = await loop.run_in_executor(
                    None,
                    analyzer.fetch_and_summarize_news,
                    5,
                    macro_data,
                    local_prices
                )
                if not summary or not isinstance(summary, str) or not summary.strip():
                    return FALLBACK_AI_SUMMARY
                return summary
            except Exception as exc:
                logger.error(f"AI news summary failed: {exc}. Using fallback message.")
                return FALLBACK_AI_SUMMARY

        ai_summary = await safe_fetch_ai_summary()

        # Step 3: Format and dispatch Telegram notification with parse_mode='HTML'
        message_html = format_telegram_report(scraped_data, ai_summary, macro_data)

        logger.info(f"Dispatching update to Telegram Chat ID: {chat_id}")
        sent_message = await bot.send_message(
            chat_id=chat_id,
            text=message_html,
            parse_mode='HTML',
            disable_web_page_preview=True
        )
        logger.info(f"Daily update sent successfully! (Message ID: {sent_message.message_id})")

        # Step 4: Stateful Storage via Google Sheets
        # Row format: [Date, Antam Buy, Antam Sell, UBS Buy, Galeri 24 Buy, Global Gold Price, USD/IDR]
        now_wib = datetime.now(JAKARTA_TZ)
        date_iso = now_wib.strftime("%Y-%m-%d %H:%M WIB")

        lm = local_prices.get("logam_mulia", {})
        ubs = local_prices.get("ubs_gold", {})
        g24 = local_prices.get("galeri24", {})

        row_to_append = [
            date_iso,
            lm.get("buy_price") or "-",
            lm.get("sell_price") or "-",
            ubs.get("buy_price") or "-",
            g24.get("buy_price") or "-",
            macro_data.get("global_gold_usd") or "-",
            macro_data.get("usd_idr") or "-"
        ]

        logger.info("Persisting daily record to Google Sheets...")
        await loop.run_in_executor(None, append_to_google_sheet, row_to_append)

    except Exception as exc:
        logger.error(f"Error executing daily update pipeline: {exc}", exc_info=True)
        # Attempt to notify admin on critical pipeline failure
        try:
            error_msg = (
                f"⚠️ <b>Gold Tracker Error Alert</b>\n"
                f"Gagal menjalankan pipeline pembaruan harian:\n"
                f"<code>{html.escape(str(exc))}</code>"
            )
            await bot.send_message(
                chat_id=chat_id,
                text=error_msg,
                parse_mode='HTML'
            )
        except Exception as notify_err:
            logger.error(f"Failed to send failure notification to Telegram: {notify_err}")
        raise


async def main() -> None:
    """Main one-off execution entrypoint."""
    logger.info("=== Gold Tracker Data Pipeline Starting Execution ===")
    await send_daily_update()
    logger.info("=== Gold Tracker Data Pipeline Finished Successfully ===")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        sys.exit(0)
    except Exception as e:
        logger.fatal(f"Fatal error during execution: {e}")
        sys.exit(1)
