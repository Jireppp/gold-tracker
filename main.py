"""
Gold Tracker & AI Market Sentiment - Telegram Automation Bot
============================================================
One-off execution script designed for scheduled serverless CI/CD runners
(e.g., GitHub Actions cron workflow).

Orchestrates:
1. Playwright Web Scraper (Logam Mulia, UBS Gold, Galeri 24)
2. AI Financial News Sentiment Analyzer (feedparser + Google GenAI)
3. Telegram Dispatcher (python-telegram-bot v20+)
"""

import asyncio
import html
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict

# Ensure UTF-8 output encoding on Windows and Linux consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pytz
from dotenv import load_dotenv
from telegram import Bot
from telegram.constants import ParseMode

from ai_news import FinancialNewsAnalyzer
from scraper import GoldPriceScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("GoldTrackerBot")

# Load environment variables
load_dotenv()

JAKARTA_TZ = pytz.timezone("Asia/Jakarta")


def clean_markdown_to_html(text: str) -> str:
    """
    Converts markdown formatting (bold, italic, bullets) from the AI summary
    into clean, valid Telegram HTML while escaping raw HTML entities.
    """
    # First escape raw HTML characters
    escaped = html.escape(text)

    # Convert **bold** to <b>bold</b>
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)

    # Convert *italic* or _italic_ to <i>italic</i>
    escaped = re.sub(r"(?<!\w)\*([^*]+?)\*(?!\w)", r"<i>\1</i>", escaped)
    escaped = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", r"<i>\1</i>", escaped)

    # Convert markdown bullet points '* ' or '- ' to unicode bullet '• '
    escaped = re.sub(r"^\s*[\*\-]\s+", "• ", escaped, flags=re.MULTILINE)

    return escaped.strip()


def format_telegram_report(scraped_data: Dict[str, Any], ai_summary: str) -> str:
    """
    Formats the scraped gold prices and AI sentiment summary into a clean,
    visually structured Telegram HTML message.
    """
    now_wib = datetime.now(JAKARTA_TZ)
    date_str = now_wib.strftime("%A, %d %B %Y | %H:%M WIB")

    data = scraped_data.get("data", {})
    lm = data.get("logam_mulia", {})
    ubs = data.get("ubs_gold", {})
    g24 = data.get("galeri24", {})

    # Helper formatters
    def price_val(val: Any) -> str:
        return f"<code>{val}</code>" if val else "<i>Tidak tersedia</i>"

    # Logam Mulia details
    lm_buy = price_val(lm.get("buy_price"))
    lm_tax = f" (+PPh: <code>{lm.get('buy_price_tax')}</code>)" if lm.get("buy_price_tax") else ""
    lm_sell = (
        price_val(lm.get("sell_price"))
        if lm.get("sell_price")
        else "<i>Dalam konfirmasi butik</i>"
    )

    # UBS Gold details
    ubs_buy = price_val(ubs.get("buy_price"))
    ubs_sell = price_val(ubs.get("sell_price"))

    # Galeri 24 details
    g24_buy = price_val(g24.get("buy_price"))
    g24_sell = price_val(g24.get("sell_price"))

    formatted_summary = clean_markdown_to_html(ai_summary)

    message_lines = [
        "📊 <b>UPDATE HARGA EMAS HARIAN & SENTIMEN PASAR</b>",
        f"🗓 <i>{date_str}</i>",
        "",
        "💰 <b>HARGA EMAS 1 GRAM</b>",
        "───────────────────────────────",
        f"🟡 <b>Logam Mulia (ANTAM)</b>",
        f"  • Harga Beli : {lm_buy}{lm_tax}",
        f"  • Buyback    : {lm_sell}",
        "",
        f"🔵 <b>UBS Gold (UBSLifestyle)</b>",
        f"  • Harga Beli : {ubs_buy}",
        f"  • Buyback    : {ubs_sell}",
        "",
        f"🟢 <b>Galeri 24 (Pegadaian)</b>",
        f"  • Harga Beli : {g24_buy}",
        f"  • Buyback    : {g24_sell}",
        "",
        "───────────────────────────────",
        "📰 <b>SENTIMEN PASAR & GEOPOLITIK (AI Executive Summary)</b>",
        f"<blockquote>{formatted_summary}</blockquote>",
        "",
        "<i>🤖 Otomasi Harian oleh Gold Tracker Bot</i>"
    ]

    return "\n".join(message_lines)


async def send_daily_update() -> None:
    """
    Executes the daily tracking workflow:
    1. Validates environment configurations.
    2. Runs Playwright scraping and AI news sentiment analysis concurrently.
    3. Builds the formatted Telegram HTML report.
    4. Sends the message to the configured CHAT_ID.
    """
    telegram_token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("CHAT_ID")

    if not telegram_token or not chat_id:
        raise ValueError(
            "Missing TELEGRAM_TOKEN or CHAT_ID in environment variables."
        )

    logger.info("Initializing bot, scraper, and news analyzer components...")
    bot = Bot(token=telegram_token)
    scraper = GoldPriceScraper()
    analyzer = FinancialNewsAnalyzer()

    # Verify bot connectivity
    bot_info = await bot.get_me()
    logger.info(f"Connected to Telegram as @{bot_info.username} ({bot_info.first_name})")

    loop = asyncio.get_running_loop()

    try:
        # Concurrent execution of Scraper and AI News Analysis
        async def fetch_news_and_analyze() -> str:
            news_items = await loop.run_in_executor(None, analyzer.fetch_latest_news, 5)
            return await loop.run_in_executor(None, analyzer.analyze_sentiments, news_items)

        logger.info("Starting concurrent tasks: Web Scraping & AI News Analysis...")
        scraper_task = scraper.scrape_all()
        ai_task = fetch_news_and_analyze()

        scraped_data, ai_summary = await asyncio.gather(scraper_task, ai_task)
        logger.info("Successfully completed data scraping and sentiment analysis.")

        # Build and send Telegram message
        message_html = format_telegram_report(scraped_data, ai_summary)

        logger.info(f"Dispatching update to Telegram Chat ID: {chat_id}")
        sent_message = await bot.send_message(
            chat_id=chat_id,
            text=message_html,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )
        logger.info(f"Daily update sent successfully! (Message ID: {sent_message.message_id})")

    except Exception as exc:
        logger.error(f"Error executing daily update: {exc}", exc_info=True)
        # Attempt to notify admin on failure
        try:
            error_msg = (
                f"⚠️ <b>Gold Tracker Error Alert</b>\n"
                f"Gagal menjalankan pembaruan harian:\n"
                f"<code>{html.escape(str(exc))}</code>"
            )
            await bot.send_message(
                chat_id=chat_id,
                text=error_msg,
                parse_mode=ParseMode.HTML
            )
        except Exception as notify_err:
            logger.error(f"Failed to send failure notification to Telegram: {notify_err}")
        raise


async def main() -> None:
    """Main one-off execution entrypoint."""
    logger.info("=== Gold Tracker Bot Starting One-Off Execution ===")
    await send_daily_update()
    logger.info("=== Gold Tracker Bot Finished Successfully ===")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        sys.exit(0)
    except Exception as e:
        logger.fatal(f"Fatal error during execution: {e}")
        sys.exit(1)
