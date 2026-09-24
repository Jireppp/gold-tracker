"""
Daily Financial News & Sentiment Analyzer
=========================================
Fetches latest market headlines from stable financial RSS feeds (CNBC Indonesia,
Antara News, Tempo Bisnis, Investing.com) and generates executive gold sentiment
analysis using the official Google GenAI SDK with gemini-3.8-flash (temperature=0.1).

Key Features:
- Uses requests to fetch XML with realistic browser User-Agent headers to bypass bot protection.
- Parses response.content via feedparser.parse().
- Multi-source failover: loops through stable RSS alternatives with per-source try-except blocks.
- Cleans and formats raw RSS content (strips HTML tags and excessive whitespace).
- Invokes Gemini via the official google-genai SDK using gemini-3.8-flash (temperature=0.1).
- Strict try-except error boundary around Gemini API: catches 429 quota limits, client errors,
  or network issues, prints to console, and returns a safe fallback message.
- Ensures the Telegram bot can always deliver raw gold prices even if AI summarization is unavailable.
"""

import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import feedparser
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AINews")

# Load environment variables
load_dotenv()

# Fallback string when AI summary fails or quota is exhausted
FALLBACK_AI_SUMMARY = "⚠️ AI Summary tidak tersedia saat ini. Mengirim data harga mentah."

# List of stable financial RSS feeds (Indonesia & Global Market/Commodities)
DEFAULT_RSS_URLS: List[str] = [
    "https://www.cnbcindonesia.com/market/rss",       # CNBC Indonesia Market
    "https://www.antaranews.com/rss/ekonomi.xml",     # Antara News Ekonomi (LKBN Antara)
    "https://rss.tempo.co/bisnis",                    # Tempo Bisnis
    "https://investing.com/rss/commodities.rss",      # Investing.com Commodities / Gold
    "https://investing.com/rss/news_285.rss",         # Investing.com Financial News
]

# Standard realistic browser headers to prevent 403 / bot protection blocking
BROWSER_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
    "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
}


class FinancialNewsAnalyzer:
    """Fetches financial RSS headlines and analyzes sentiment with Google GenAI."""

    def __init__(self, rss_urls: Optional[List[str]] = None) -> None:
        self.api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        self.client: Optional[genai.Client] = None

        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as init_err:
                print(f"Error initializing Gemini client: {init_err}")
                logger.error(f"Error initializing Gemini client: {init_err}")

        if rss_urls:
            self.rss_urls = list(rss_urls)
        else:
            env_urls = os.getenv("FINANCIAL_RSS_URLS") or os.getenv("FINANCIAL_RSS_URL")
            if env_urls:
                custom_urls = [u.strip() for u in env_urls.split(",") if u.strip()]
                self.rss_urls = custom_urls + [u for u in DEFAULT_RSS_URLS if u not in custom_urls]
            else:
                self.rss_urls = list(DEFAULT_RSS_URLS)

        # Updated default model to gemini-3.8-flash
        self.primary_model: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    def fetch_latest_news(self, limit: int = 5) -> List[Dict[str, str]]:
        """
        Fetches the latest entries from stable financial RSS feeds using requests
        with realistic browser headers. Iterates through fallback URLs if one fails.
        """
        last_error: Optional[Exception] = None

        for url in self.rss_urls:
            try:
                logger.info(f"Fetching RSS feed from: {url}")
                response = requests.get(url, headers=BROWSER_HEADERS, timeout=12)
                response.raise_for_status()

                # Parse raw XML bytes using feedparser
                feed = feedparser.parse(response.content)

                if getattr(feed, "bozo", 0) and not feed.entries:
                    bozo_err = feed.get("bozo_exception", "XML syntax or parsing error")
                    raise RuntimeError(f"Failed to parse XML from {url}: {bozo_err}")

                if not feed.entries:
                    logger.warning(f"No news entries found in feed: {url}. Trying next alternative...")
                    continue

                news_items: List[Dict[str, str]] = []
                for entry in feed.entries[:limit]:
                    title = entry.get("title", "").strip()
                    raw_desc = entry.get("description", entry.get("summary", ""))

                    # Strip HTML tags and normalize whitespace
                    clean_desc = re.sub(r"<[^>]+>", "", raw_desc).strip()
                    clean_desc = re.sub(r"\s+", " ", clean_desc)

                    news_items.append({
                        "title": title,
                        "description": clean_desc
                    })

                if news_items:
                    logger.info(
                        f"Successfully extracted {len(news_items)} news items from {url}."
                    )
                    return news_items

            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"Failed to fetch/parse RSS from {url}: {exc}. Continuing to next source..."
                )
                continue

        raise RuntimeError(
            f"All RSS sources failed to return valid news entries. Last error: {last_error}"
        )

    def analyze_sentiments(self, news_items: List[Dict[str, str]]) -> str:
        """
        Sends the compiled news headlines to Gemini and generates an Indonesian
        executive sentiment summary with gemini-3.8-flash and temperature=0.1.

        Strict try-except block: If the Gemini API hits a quota limit (e.g. 429),
        fails, or throws any error, it catches the exception, prints it to console,
        and returns FALLBACK_AI_SUMMARY.
        """
        if not self.client:
            print("Gemini client is not initialized (missing or invalid API key).")
            logger.warning("Gemini client is not initialized. Using fallback summary.")
            return FALLBACK_AI_SUMMARY

        try:
            # Format news content clearly
            formatted_headlines = []
            for idx, item in enumerate(news_items, start=1):
                formatted_headlines.append(
                    f"{idx}. Judul: {item['title']}\n   Deskripsi: {item['description']}"
                )
            news_payload = "\n\n".join(formatted_headlines)

            prompt = (
                "Analyze these news headlines. Write a maximum 3-bullet-point executive summary in Indonesian "
                "about gold price sentiments or geopolitical impacts. If there is no relevant news, state clearly "
                "that there are no major sentiment drivers today.\n\n"
                f"News Headlines:\n{news_payload}"
            )

            # Strict temperature=0.1 as specified
            config = types.GenerateContentConfig(
                temperature=0.1,
            )

            # Model hierarchy: gemini-3.8-flash first, followed by active production fallbacks
            model_chain = [self.primary_model]
            for fallback in ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash"]:
                if fallback not in model_chain:
                    model_chain.append(fallback)

            last_error = None
            for model in model_chain:
                try:
                    logger.info(f"Invoking Gemini model '{model}' with temperature=0.1...")
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )
                    if response and response.text:
                        return response.text.strip()
                except errors.ClientError as exc:
                    last_error = exc
                    # Quota limit (429) or resource exhausted
                    if exc.code == 429 or "quota" in str(exc).lower() or "resource_exhausted" in str(exc).lower():
                        print(f"Gemini API Quota/Rate Limit Error (429) on model '{model}': {exc}")
                        logger.error(f"Gemini API Quota/Rate Limit Error (429): {exc}")
                        return FALLBACK_AI_SUMMARY
                    # 404 means model unavailable; try next model in chain
                    if exc.code == 404 or "not found" in str(exc).lower():
                        logger.warning(f"Model '{model}' not found on endpoint. Trying next model...")
                        continue
                    print(f"Gemini API ClientError on model '{model}': {exc}")
                    logger.warning(f"Gemini API ClientError on '{model}': {exc}")
                except errors.ServerError as exc:
                    last_error = exc
                    print(f"Gemini API ServerError on model '{model}': {exc}")
                    logger.warning(f"Gemini API ServerError on '{model}': {exc}")
                except Exception as exc:
                    last_error = exc
                    print(f"Unexpected error calling Gemini API on model '{model}': {exc}")
                    logger.warning(f"Unexpected error calling Gemini API: {exc}")

            # If all model calls in the loop failed
            print(f"All Gemini models failed. Last error: {last_error}")
            logger.error(f"All Gemini models failed: {last_error}")
            return FALLBACK_AI_SUMMARY

        except Exception as exc:
            # Catch-all strict try-except around entire Gemini call
            print(f"Error calling Gemini API: {exc}")
            logger.error(f"Error calling Gemini API: {exc}")
            return FALLBACK_AI_SUMMARY

    def fetch_and_summarize_news(self, limit: int = 5) -> str:
        """
        Orchestrates fetching news from stable RSS feeds and summarizing with Gemini (gemini-3.8-flash).
        Catches any exception, prints it to the console, and returns FALLBACK_AI_SUMMARY.
        """
        try:
            news_items = self.fetch_latest_news(limit=limit)
            return self.analyze_sentiments(news_items)
        except Exception as exc:
            print(f"Error in fetch_and_summarize_news: {exc}")
            logger.error(f"Error in fetch_and_summarize_news: {exc}")
            return FALLBACK_AI_SUMMARY


def fetch_and_summarize_news(
    limit: int = 5,
    rss_urls: Optional[List[str]] = None
) -> str:
    """
    Convenience function:
    1. Fetches XML using requests with realistic browser User-Agent headers.
    2. Parses response.content via feedparser.
    3. Handles multi-source failover across stable RSS alternatives with try-except blocks.
    4. Analyzes sentiment using Google GenAI SDK (gemini-3.8-flash, temperature=0.1).
    5. Returns FALLBACK_AI_SUMMARY on quota limits (429) or any failure.
    """
    try:
        analyzer = FinancialNewsAnalyzer(rss_urls=rss_urls)
        return analyzer.fetch_and_summarize_news(limit=limit)
    except Exception as exc:
        print(f"Error in fetch_and_summarize_news: {exc}")
        logger.error(f"Error in fetch_and_summarize_news: {exc}")
        return FALLBACK_AI_SUMMARY


def main() -> None:
    print("\n" + "=" * 60)
    print("TESTING fetch_and_summarize_news() with gemini-3.8-flash")
    print("=" * 60 + "\n")

    summary = fetch_and_summarize_news(limit=5)

    print("\n" + "=" * 60)
    print("RESULTING SUMMARY:")
    print("=" * 60)
    print(summary)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
