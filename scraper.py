"""
Daily Gold Price Scraper
========================
Scrapes buy and sell gold prices (1 gram) from three primary Indonesian sources:
1. Logam Mulia (PT ANTAM Tbk)
2. UBS Gold (PT Untung Bersama Sejahtera / UBSLifestyle)
3. Galeri 24 (PT Pegadaian Galeri Dua Empat)

Key Architectural Features:
- Playwright Async API for dynamic JavaScript-rendered SPAs.
- No BeautifulSoup dependency.
- Isolated error boundaries: Individual try-except blocks per source ensure
  DOM changes or network/WAF anomalies never crash the overall execution.
- Configurable environment via python-dotenv (.env).
- Portable timezone handling (WIB / UTC+7).
- Clean formatted JSON output.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("GoldScraper")

# Load environment configuration from .env file
load_dotenv()


class GoldPriceScraper:
    """Production-grade gold price scraper using Playwright async API."""

    def __init__(self) -> None:
        self.logam_mulia_url: str = os.getenv(
            "LOGAM_MULIA_URL",
            "https://www.logammulia.com/id/harga-emas-hari-ini"
        )
        self.ubs_gold_url: str = os.getenv(
            "UBS_GOLD_URL",
            "https://ubslifestyle.com/harga-buyback-hari-ini/"
        )
        self.galeri24_url: str = os.getenv(
            "GALERI24_URL",
            "https://galeri24.co.id/harga-emas"
        )
        self.headless: bool = os.getenv("HEADLESS", "true").lower() in ("true", "1", "yes")
        self.timeout_ms: int = int(os.getenv("SCRAPER_TIMEOUT_MS", "30000"))

    async def _create_context(self, browser: Browser) -> BrowserContext:
        """Creates a browser context with realistic desktop fingerprinting."""
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="id-ID",
            timezone_id="Asia/Jakarta",
            viewport={"width": 1280, "height": 800},
            extra_http_headers={
                "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
            }
        )
        # Suppress navigator.webdriver flag to mitigate basic bot detection
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        return context

    async def scrape_logam_mulia(self, page: Page) -> Dict[str, Any]:
        """
        Scrapes gold prices from PT ANTAM Tbk (Logam Mulia).
        Extracts 1 gr base price and tax price from the official price page.
        Gracefully handles buyback endpoint access.
        """
        logger.info(f"Navigating to Logam Mulia (ANTAM): {self.logam_mulia_url}")
        response = await page.goto(
            self.logam_mulia_url,
            wait_until="domcontentloaded",
            timeout=self.timeout_ms
        )
        if not response or response.status >= 400:
            raise RuntimeError(
                f"Failed to load Logam Mulia page, HTTP status: {response.status if response else 'No response'}"
            )

        # Allow dynamic table hydration
        await page.wait_for_timeout(2000)

        # Extract 1 gr price row from the main pricing table
        table_data = await page.evaluate('''() => {
            const tables = document.querySelectorAll('table');
            for (const table of tables) {
                const rows = table.querySelectorAll('tr');
                for (const row of rows) {
                    const cells = Array.from(row.querySelectorAll('td, th')).map(c => c.innerText.trim());
                    if (cells.length >= 2 && cells[0].toLowerCase() === '1 gr') {
                        return {
                            weight: cells[0],
                            base_price: cells[1],
                            tax_price: cells.length >= 4 ? cells[3] : (cells.length >= 3 ? cells[2] : null)
                        };
                    }
                }
            }
            return null;
        }''')

        if not table_data:
            raise ValueError(
                "Could not find '1 gr' price row in Logam Mulia table. DOM structure may have changed."
            )

        buy_price = f"Rp{table_data['base_price'].replace(',', '.')}" if table_data.get('base_price') else None
        buy_price_tax = f"Rp{table_data['tax_price'].replace(',', '.')}" if table_data.get('tax_price') else None

        # Check buyback (sell) price
        sell_price: Optional[str] = None
        sell_error: Optional[str] = None
        try:
            sell_url = "https://www.logammulia.com/id/sell/gold"
            sell_resp = await page.goto(
                sell_url,
                referer=self.logam_mulia_url,
                timeout=10000,
                wait_until="domcontentloaded"
            )
            if sell_resp and sell_resp.status == 403:
                sell_error = "Cloudflare bot protection active on /sell/gold (HTTP 403)"
            elif sell_resp and sell_resp.status < 400:
                bb_text = await page.evaluate('''() => {
                    const el = document.querySelector('.price, .buyback-price, .total-price');
                    return el ? el.innerText.trim() : null;
                }''')
                if bb_text:
                    sell_price = bb_text
                else:
                    sell_error = "Sell price element not found on sell page"
            else:
                sell_error = f"Sell page returned HTTP {sell_resp.status if sell_resp else 'N/A'}"
        except Exception as e:
            sell_error = f"Buyback endpoint access error: {str(e)}"

        return {
            "source": "Logam Mulia (ANTAM)",
            "status": "success",
            "weight": table_data["weight"],
            "buy_price": buy_price,
            "buy_price_tax": buy_price_tax,
            "sell_price": sell_price,
            "sell_price_note": sell_error if not sell_price else None
        }

    async def scrape_ubs_gold(self, page: Page) -> Dict[str, Any]:
        """
        Scrapes gold prices from UBS Gold (UBSLifestyle).
        Extracts 1 Gram buy price (Harga Beli) and sell price (Harga Jual / Buyback).
        """
        logger.info(f"Navigating to UBS Gold: {self.ubs_gold_url}")
        response = await page.goto(
            self.ubs_gold_url,
            wait_until="domcontentloaded",
            timeout=self.timeout_ms
        )
        if not response or response.status >= 400:
            raise RuntimeError(
                f"Failed to load UBS Gold page, HTTP status: {response.status if response else 'No response'}"
            )

        # Wait for table rendering
        await page.wait_for_selector("table", timeout=self.timeout_ms)

        # Extract 1 Gram row from the pricing table
        ubs_data = await page.evaluate('''() => {
            const rows = document.querySelectorAll('table tr');
            for (const row of rows) {
                const cells = Array.from(row.querySelectorAll('td, th')).map(c => c.innerText.trim());
                if (cells.length >= 3 && cells[0].toLowerCase() === '1 gram') {
                    return {
                        weight: cells[0],
                        buy_price: cells[1],
                        sell_price: cells[2]
                    };
                }
            }
            return null;
        }''')

        if not ubs_data:
            raise ValueError(
                "Could not find '1 Gram' row in UBS Gold pricing table. DOM structure may have changed."
            )

        return {
            "source": "UBS Gold",
            "status": "success",
            "weight": ubs_data["weight"],
            "buy_price": ubs_data["buy_price"],
            "sell_price": ubs_data["sell_price"]
        }

    async def scrape_galeri24(self, page: Page) -> Dict[str, Any]:
        """
        Scrapes gold prices from Galeri 24 (PT Pegadaian Galeri Dua Empat).
        Extracts 1 gr buy price (Harga Jual Galeri 24) and sell price (Harga Buyback).
        """
        logger.info(f"Navigating to Galeri 24: {self.galeri24_url}")
        response = await page.goto(
            self.galeri24_url,
            wait_until="domcontentloaded",
            timeout=self.timeout_ms
        )
        if not response or response.status >= 400:
            raise RuntimeError(
                f"Failed to load Galeri 24 page, HTTP status: {response.status if response else 'No response'}"
            )

        # Wait for dynamic table hydration
        await page.wait_for_timeout(3000)

        # Parse vendor Galeri 24 card
        g24_data = await page.evaluate('''() => {
            // Find container block for 'Harga GALERI 24'
            const blocks = Array.from(document.querySelectorAll('div')).filter(el => {
                const text = el.innerText || '';
                return text.includes('Harga GALERI 24') && text.includes('Harga Jual') && text.includes('Harga Buyback');
            });

            for (const block of blocks) {
                const rows = Array.from(block.querySelectorAll('div, tr')).map(r => r.innerText.trim());
                for (const row of rows) {
                    const parts = row.split(/\\s+/);
                    // Match line format: '1 Rp2.573.000 Rp2.434.000'
                    if (parts[0] === '1' && parts.length >= 3 && parts[1].startsWith('Rp') && parts[2].startsWith('Rp')) {
                        return {
                            vendor: 'GALERI 24',
                            weight: '1 gr',
                            buy_price: parts[1],
                            sell_price: parts[2]
                        };
                    }
                }
            }
            return null;
        }''')

        if not g24_data:
            raise ValueError(
                "Could not find '1 gr' price row in Galeri 24 vendor table. DOM structure may have changed."
            )

        return {
            "source": "Galeri 24",
            "status": "success",
            "weight": g24_data["weight"],
            "buy_price": g24_data["buy_price"],
            "sell_price": g24_data["sell_price"]
        }

    async def _safe_scrape_source(
        self,
        name: str,
        scraper_func,
        context: BrowserContext
    ) -> Dict[str, Any]:
        """
        Executes an individual scraper function in its own dedicated page tab
        with strict error boundary isolation.
        """
        page = await context.new_page()
        try:
            return await scraper_func(page)
        except Exception as exc:
            logger.error(f"[{name}] Scraping failed: {exc}", exc_info=False)
            return {
                "source": name,
                "status": "error",
                "error": f"Scraping failed: {str(exc)}",
                "buy_price": None,
                "sell_price": None
            }
        finally:
            await page.close()

    async def scrape_all(self) -> Dict[str, Any]:
        """
        Executes scraping across all three sources.
        Guarantees that a failure in one source will never halt or crash the overall process.
        """
        wib_tz = timezone(timedelta(hours=7))
        results: Dict[str, Any] = {
            "scraped_at": datetime.now(wib_tz).isoformat(),
            "data": {}
        }

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=self.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-infobars"
                ]
            )

            try:
                context = await self._create_context(browser)

                # Execute all three scrapers concurrently with isolated error boundaries
                lm_task = self._safe_scrape_source("Logam Mulia (ANTAM)", self.scrape_logam_mulia, context)
                ubs_task = self._safe_scrape_source("UBS Gold", self.scrape_ubs_gold, context)
                g24_task = self._safe_scrape_source("Galeri 24", self.scrape_galeri24, context)

                lm_res, ubs_res, g24_res = await asyncio.gather(lm_task, ubs_task, g24_task)

                results["data"]["logam_mulia"] = lm_res
                results["data"]["ubs_gold"] = ubs_res
                results["data"]["galeri24"] = g24_res

                await context.close()
            finally:
                await browser.close()

        return results


async def main() -> None:
    scraper = GoldPriceScraper()
    scraped_data = await scraper.scrape_all()
    # Output cleanly formatted JSON to the console
    print(json.dumps(scraped_data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
