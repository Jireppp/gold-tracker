# Gold Tracker & Macro Analyzer Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg?logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![Playwright](https://img.shields.io/badge/Scraper-Playwright-2EAD33.svg?logo=playwright&logoColor=white)](https://playwright.dev/python/)
[![Google Gemini](https://img.shields.io/badge/AI-Gemini%203.8%20Flash-orange.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An automated, serverless data pipeline that monitors physical gold bullion prices in Indonesia, pairs them with global macroeconomic indicators, generates AI-synthesized market insights, and delivers actionable daily briefings via Telegram and Google Sheets.

---

## 1. Project Overview (The "Why")

Tracking physical gold bullion prices in Indonesia typically requires visiting multiple dealer websites with inconsistent layouts and Cloudflare-protected single-page applications. Furthermore, local gold prices do not move in isolation—they reflect a synthesis of **global spot gold price dynamics (XAU/USD)** and the **currency exchange rate (USD/IDR)**.

This project goes beyond basic web scraping by providing an end-to-end automated analytics pipeline:

* **Multilateral Ingestion:** Concurrently captures live retail buy and buyback prices across Indonesia's major bullion distributors: **Logam Mulia (PT ANTAM Tbk)**, **UBS Gold (PT Untung Bersama Sejahtera)**, and **Galeri 24 (PT Pegadaian)**.
* **Macro Correlation:** Ingests live global spot gold metrics and foreign exchange rates via Yahoo Finance.
* **Data-Grounded AI Analysis:** Uses Google Gemini (Gemini 3.8 Flash, `temperature=0.1`) to evaluate whether local price movements are driven by international bullion demand or domestic currency devaluation.
* **Smart Decision Support:** Delivers formatted briefings with grounded Buy/Hold/Wait indicators to investors every morning at 10:30 AM WIB (03:30 UTC).

---

## 2. Key Features

* **Automated Web Scraping (Playwright Async):** Handles single-page applications (SPAs) and dynamic DOM hydration across three e-commerce platforms concurrently. Includes independent try-except boundaries to ensure single-source DOM shifts never crash the broader pipeline.
* **Macroeconomic Ingestion (`yfinance`):** Collects real-time international financial indicators (`GC=F` COMEX Gold Futures, `XAUUSD=X` Spot Gold, and `IDR=X` Exchange Rate).
* **AI-Powered Sentiment Analysis (Gemini 3.8 Flash):** Parses top financial news feeds (CNBC Indonesia, Antara, Tempo, Investing.com) and combines market text with numeric indicators to generate a three-point executive summary.
* **Stateful Data Persistence (Google Sheets via `gspread`):** Appends daily price rows (`[Date, Antam Buy, Antam Sell, UBS Buy, Galeri 24 Buy, Global Gold Price, USD/IDR]`) using service account credentials securely parsed in-memory.
* **Mobile-Optimized Telegram Delivery:** Sends clean, monospaced HTML reports directly to the user's Telegram chat, complete with fallback handling if external APIs encounter rate limits.

---

## 3. Architecture & Data Flow

```text
+-----------------------------------------------------------------------------------+
|                            EXECUTION: GitHub Actions                              |
|                Cron Schedule (Daily 10:30 AM WIB / 03:30 UTC)                    |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                               DATA INGESTION                                      |
|  +--------------------------------+   +----------------------------------------+  |
|  |   Playwright Async Scraper     |   |          yfinance API                  |  |
|  |   - Antam (Logam Mulia)        |   |   - Global Gold (GC=F / XAU/USD)       |  |
|  |   - UBS Gold                   |   |   - USD/IDR Currency Rate (IDR=X)      |  |
|  |   - Galeri 24 (Pegadaian)      |   +----------------------------------------+  |
|  +--------------------------------+   +----------------------------------------+  |
|                                       |       RSS Financial News Feeds         |  |
|                                       |       - CNBC, Antara, Tempo            |  |
|                                       +----------------------------------------+  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        PROCESSING & AI REASONING                                  |
|   Google GenAI SDK (Gemini 3.8 Flash @ temperature=0.1)                          |
|   - Evaluates correlation: Global Bullion vs. Currency Fluctuation                |
|   - Generates 3-point Indonesian Executive Briefing + Buy/Hold/Wait Stance        |
+--------------------+---------------------------------------------+----------------+
                     |                                             |
                     v                                             v
+-----------------------------------------+   +-------------------------------------+
|           STATEFUL STORAGE              |   |          DELIVERY LAYER             |
|   Google Sheets API (via gspread)       |   |   Telegram Bot API (HTML)           |
|   Appends timestamped row for auditing  |   |   Mobile-optimized daily alert sent |
|   and historical timeseries analysis    |   |   to target Chat ID                 |
+-----------------------------------------+   +-------------------------------------+
```

---

## 4. Prerequisites & Installation

### Prerequisites
* Python 3.10 or higher
* Google Cloud Platform service account key (with Google Sheets & Drive API access)
* Telegram Bot API Token (obtained via [@BotFather](https://t.me/botfather))
* Google AI Studio Gemini API Key

### Step-by-Step Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/gold-tracker-bot.git
   cd gold-tracker-bot
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install project dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install Playwright Chromium browser binary:**
   ```bash
   playwright install chromium
   ```

---

## 5. Environment Variables Configuration

Create a `.env` file in the root directory:

```ini
# ==========================================
# Telegram Bot Credentials
# ==========================================
TELEGRAM_TOKEN=your_telegram_bot_token_here
CHAT_ID=your_telegram_chat_id_here

# ==========================================
# AI Model Configuration
# ==========================================
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash

# ==========================================
# Google Sheets Stateful Storage
# ==========================================
SPREADSHEET_ID=your_google_spreadsheet_id_here
# Minified Google Cloud Service Account JSON string
GCP_CREDENTIALS_JSON='{"type":"service_account","project_id":"your-project","private_key_id":"...","private_key":"...","client_email":"...","client_id":"...","auth_uri":"...","token_uri":"...","auth_provider_x509_cert_url":"...","client_x509_cert_url":"..."}'

# ==========================================
# Scraper & Ingestion Parameters
# ==========================================
LOGAM_MULIA_URL=https://www.logammulia.com/id/harga-emas-hari-ini
UBS_GOLD_URL=https://ubslifestyle.com/harga-buyback-hari-ini/
GALERI24_URL=https://galeri24.co.id/harga-emas
HEADLESS=true
SCRAPER_TIMEOUT_MS=30000
```

> [!IMPORTANT]
> Share your Google Sheet with the `client_email` listed in your `GCP_CREDENTIALS_JSON` and assign it the **Editor** role so the service account can append rows.

---

## 6. Execution & Verification

Run the pipeline manually to perform a dry run:

```bash
python main.py
```

### Expected Output
1. Fetches spot gold and currency rates via `yfinance`.
2. Concurrently scrapes Antam, UBS, and Galeri 24 1-gram buy and sell prices.
3. Downloads the top 5 news entries from CNBC Indonesia and generates a Gemini market sentiment summary.
4. Formats and sends an HTML report to Telegram:
   ```text
   📊 DAILY GOLD & MACRO REPORT
   🗓 Monday, 28 September 2026 | 11:40 WIB

   🌐 GLOBAL INDICATORS
   • XAU/USD : $4,232.50 / troy oz
   • USD/IDR : Rp 17,945

   💰 LOCAL PRICE (1 GR)
   1. ANTAM
   B: Rp2.597.000 | S: Dalam konfirmasi butik
   2. UBS Gold
   B: Rp2.531.900 | S: Tidak tersedia
   3. Galeri 24
   B: Rp2.527.000 | S: Tidak tersedia

   📰 MACRO ANALYSIS & SENTIMENT
   > • Global gold prices stabilized near historic highs while USD/IDR trades at Rp 17,945.
   > • Local retail increases were primarily driven by domestic currency fluctuations.
   > • Stance: Hold existing bullion; wait for technical pullbacks for new entry.

   🤖 Automated by Gold Tracker Bot
   ```
5. Appends a new timestamped row to the configured Google Sheet.

---

## 7. Cloud Deployment (GitHub Actions)

The repository includes a ready-to-run GitHub Actions workflow located at [`.github/workflows/schedule.yml`](.github/workflows/schedule.yml).

### Workflow Configuration
* **Trigger Mechanism:**
  * **Cron:** Runs daily at `30 3 * * *` (03:30 AM UTC = 10:30 AM WIB).
  * **Manual Dispatch:** Can be triggered on-demand via the GitHub Actions UI (`workflow_dispatch`).
* **Runner Environment:** `ubuntu-latest` with Python 3.10 and automated dependency caching.
* **Browser Installation:** Automatically downloads Chromium system dependencies using `playwright install --with-deps chromium`.

### Required GitHub Repository Secrets
Under your GitHub repository settings (**Settings** > **Secrets and variables** > **Actions**), add the following secrets:

| Secret Name | Description |
| :--- | :--- |
| `TELEGRAM_TOKEN` | Bot API Token generated by BotFather |
| `CHAT_ID` | Telegram chat or channel ID to receive updates |
| `GEMINI_API_KEY` | Google AI Studio Gemini API key |
| `GCP_CREDENTIALS_JSON` | Minified Google Cloud Service Account JSON key |
| `SPREADSHEET_ID` | Google Sheet unique ID string |

---

## 8. License

This project is licensed under the [MIT License](LICENSE).
