"""
Daily Financial News & Sentiment Analyzer
=========================================
Fetches latest market headlines from financial RSS feeds (CNBC Indonesia / Investing.com)
and generates executive gold sentiment analysis using the official Google GenAI SDK.

Key Features:
- Fetches 5 latest headlines and descriptions using feedparser.
- Cleans and formats raw RSS content (strips HTML tags and excessive whitespace).
- Invokes Gemini via the official google-genai SDK (types.GenerateContentConfig).
- Uses temperature=0.1 for high determinism and grounded summaries.
- Reads configuration and GEMINI_API_KEY from .env.
- Implements robust retry and model fallback chain (gemini-1.5-flash -> gemini-3.8-flash -> gemini-3.5-flash).
- Handles Windows UTF-8 stdout encoding cleanly.
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


class FinancialNewsAnalyzer:
    """Fetches financial RSS headlines and analyzes sentiment with Google GenAI."""

    def __init__(self) -> None:
        self.api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not defined in .env file. Please check your configuration."
            )

        self.rss_url: str = os.getenv(
            "FINANCIAL_RSS_URL",
            "https://www.cnbcindonesia.com/market/rss"
        )
        self.primary_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.client = genai.Client(api_key=self.api_key)

    def fetch_latest_news(self, limit: int = 5) -> List[Dict[str, str]]:
        """
        Fetches the latest entries from the RSS feed, extracting and cleaning
        title and description fields.
        """
        logger.info(f"Fetching RSS feed from: {self.rss_url}")
        feed = feedparser.parse(self.rss_url)

        if getattr(feed, "bozo", 0) and not feed.entries:
            raise RuntimeError(
                f"Failed to parse RSS feed from {self.rss_url}: {feed.get('bozo_exception', 'Unknown error')}"
            )

        if not feed.entries:
            raise ValueError(f"No news entries found in feed: {self.rss_url}")

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

        logger.info(f"Successfully extracted {len(news_items)} news items.")
        return news_items

    def analyze_sentiments(self, news_items: List[Dict[str, str]]) -> str:
        """
        Sends the compiled news headlines to Gemini and generates an Indonesian
        executive sentiment summary with temperature=0.1.
        Uses a resilient fallback chain if gemini-1.5-flash is sunset on the endpoint.
        """
        # Format news content clearly
        formatted_headlines = []
        for idx, item in enumerate(news_items, start=1):
            formatted_headlines.append(
                f"{idx}. Judul: {item['title']}\n   Deskripsi: {item['description']}"
            )
        news_payload = "\n\n".join(formatted_headlines)

        # Exact user-specified prompt
        prompt = (
            "Analyze these news headlines. Write a maximum 3-bullet-point executive summary in Indonesian "
            "about gold price sentiments or geopolitical impacts. If there is no relevant news, state clearly "
            "that there are no major sentiment drivers today.\n\n"
            f"News Headlines:\n{news_payload}"
        )

        config = types.GenerateContentConfig(
            temperature=0.1,
        )

        # Model hierarchy: primary requested model first, followed by active production fallbacks
        model_chain = [self.primary_model]
        for fallback in ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3-flash-preview"]:
            if fallback not in model_chain:
                model_chain.append(fallback)

        last_error = None
        for model in model_chain:
            for attempt in range(2):
                try:
                    logger.info(f"Invoking Gemini model '{model}' (attempt {attempt + 1})...")
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )
                    if response.text:
                        return response.text
                except errors.ClientError as exc:
                    last_error = exc
                    # 404 means model is sunset or unavailable on this API version; proceed to next model immediately
                    if exc.code == 404 or "not found" in str(exc).lower():
                        logger.warning(
                            f"Model '{model}' is not available on this API endpoint ({exc.message}). "
                            "Proceeding to next fallback model..."
                        )
                        break
                    logger.warning(f"ClientError on '{model}': {exc}")
                    time.sleep(1)
                except errors.ServerError as exc:
                    last_error = exc
                    # 503 transient spike in demand; wait briefly and retry
                    logger.warning(f"ServerError on '{model}': {exc}. Retrying...")
                    time.sleep(2)
                except Exception as exc:
                    last_error = exc
                    logger.warning(f"Unexpected error invoking '{model}': {exc}")
                    time.sleep(1)

        raise RuntimeError(f"All model invocations failed. Last error: {last_error}")


def main() -> None:
    analyzer = FinancialNewsAnalyzer()
    news = analyzer.fetch_latest_news(limit=5)
    summary = analyzer.analyze_sentiments(news)

    print("\n" + "=" * 60)
    print("AI EXECUTIVE GOLD & MARKET SENTIMENT SUMMARY")
    print("=" * 60 + "\n")
    print(summary)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
