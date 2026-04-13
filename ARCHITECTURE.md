# IDX Bot — Architecture & Technical Specification

---

## File Structure

```
idx_bot/
├── BRIEF.md                  ← Read first (project overview)
├── ARCHITECTURE.md           ← This file (technical spec)
├── main.py                   ← Orchestrator + scheduler
├── config.py                 ← All settings + env vars
├── fetcher.py                ← yfinance data + IDX foreign flow
├── indicators.py             ← RSI, MA, volume (same as crypto)
├── forecaster.py             ← Prophet 5-day price forecast
├── news_fetcher.py           ← Google News RSS + IDX announcements
├── scorer.py                 ← Combines all 4 layers → score 0-100
├── ai_agent.py               ← Claude API → structured verdict
├── telegram_bot.py           ← Alerts + commands + polling
├── firestore_client.py       ← Firebase read/write
├── risk_manager.py           ← Lot sizing, daily loss limit
├── scheduler.py              ← WIB-aware market hours scheduler
├── requirements.txt
├── Procfile
├── .env.example
└── .python-version           ← 3.11
```

---

## Module Specs

### config.py

```python
import os
from dotenv import load_dotenv
load_dotenv()

# ── Stocks ────────────────────────────────────
STOCKS = [
    "BBCA.JK", "BBRI.JK", "TLKM.JK", "ASII.JK",
    "GOTO.JK", "BREN.JK", "BMRI.JK", "UNVR.JK",
]

# ── Timeframes ────────────────────────────────
CANDLE_INTERVAL  = "5m"     # 5-minute candles during market
CANDLE_LIMIT     = 60       # 60 candles = 5 hours history
DAILY_CANDLES    = 365      # 1yr for Prophet training

# ── Market hours (WIB = UTC+7) ────────────────
MARKET_TZ        = "Asia/Jakarta"
SESSION1_START   = (9, 0)    # 09:00
SESSION1_END     = (12, 0)   # 12:00
SESSION2_START   = (13, 0)   # 13:00
SESSION2_END     = (15, 49)  # 15:49
BRIEFING1_TIME   = (8, 45)   # Pre-session 1 briefing
BRIEFING2_TIME   = (12, 30)  # Pre-session 2 briefing
EOD_TIME         = (16, 0)   # End of day summary

# ── Scan interval ─────────────────────────────
SCAN_INTERVAL_SEC = 300      # 5 minutes during market hours

# ── Scoring weights ───────────────────────────
WEIGHT_TECHNICAL  = 35
WEIGHT_PROPHET    = 25
WEIGHT_FOREIGN    = 20
WEIGHT_NEWS       = 20

# ── Risk ──────────────────────────────────────
IDX_CAPITAL           = int(os.getenv("IDX_CAPITAL", "5000000"))
IDX_MAX_LOTS          = int(os.getenv("IDX_MAX_LOTS_PER_STOCK", "5"))
IDX_DAILY_LOSS_PCT    = float(os.getenv("IDX_DAILY_LOSS_LIMIT_PCT", "3.0"))
IDX_MIN_SCORE         = int(os.getenv("IDX_MIN_SCORE", "60"))

# ── Fees ──────────────────────────────────────
BUY_FEE_PCT   = 0.0015   # 0.15%
SELL_FEE_PCT  = 0.0025   # 0.25% (incl. 0.10% tax)
FEE_RT        = BUY_FEE_PCT + SELL_FEE_PCT   # 0.40% round trip
MIN_TARGET_PCT = 1.5      # minimum viable target after fees

# ── Firebase ──────────────────────────────────
FIREBASE_CRED_JSON = os.getenv("FIREBASE_CRED_JSON", "")

# ── Anthropic ─────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = "claude-haiku-4-5-20251001"

# ── Telegram ──────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
```

---

### fetcher.py

```python
# Two responsibilities:
# 1. OHLCV candles + live quote via yfinance
# 2. IDX foreign flow (net buy/sell) via IDX public endpoint

import yfinance as yf
import requests
import pandas as pd

def fetch_candles(symbol: str, interval="5m", period="1d") -> pd.DataFrame | None:
    """
    Fetch OHLCV candles for a .JK stock.
    Returns DataFrame with: open, high, low, close, volume
    symbol: e.g. "BBCA.JK"
    interval: "5m" for live, "1d" for Prophet training
    period: "1d" for intraday, "1y" for Prophet
    """

def fetch_quote(symbol: str) -> dict | None:
    """
    Fetch live quote for a single stock.
    Returns: { price, change_pct, high, low, volume, market_cap }
    """

def fetch_foreign_flow(symbol: str) -> dict | None:
    """
    Fetch today's foreign net buy/sell for a stock from IDX.
    IDX endpoint: https://www.idx.co.id/umbraco/Surface/StockData/
    Returns: { net_foreign_buy_idr, foreign_buy_vol, foreign_sell_vol, days_consecutive }
    days_consecutive: how many days in a row foreign has been net buying/selling
    """

def fetch_jci_summary() -> dict | None:
    """
    Fetch JCI (Composite Index) daily summary.
    Returns: { jci_close, jci_change_pct, total_foreign_net_idr, market_status }
    Used in pre-session briefing header.
    """
```

---

### forecaster.py

```python
# Prophet-based 5-day price forecast
# Train on 1yr daily OHLCV, predict next 5 trading days
# Cache model per stock, retrain weekly

from prophet import Prophet
import pandas as pd

def train_model(symbol: str, df_daily: pd.DataFrame) -> Prophet:
    """
    Train Prophet on 1yr daily close prices.
    df_daily must have columns: ds (date), y (close price)
    Returns fitted Prophet model.
    """

def forecast_5d(symbol: str) -> dict | None:
    """
    Returns 5-day forecast for a stock.
    {
        current_price: float,
        forecast_5d: float,       # predicted price in 5 days
        trend_pct: float,         # % change expected
        trend: "UP" | "DOWN" | "FLAT",
        confidence_interval: { lower: float, upper: float },
        prophet_score: int,       # 0-25 pts for scoring layer
    }
    """

def get_prophet_score(trend_pct: float) -> int:
    """
    Convert forecast to score points (0-25):
    trend >= +3%   → 25 pts
    trend >= +2%   → 20 pts
    trend >= +1%   → 15 pts
    trend >= +0.5% → 10 pts
    trend FLAT     → 5 pts
    trend negative → 0 pts
    """
```

---

### news_fetcher.py

```python
# Two sources:
# 1. Google News RSS — search by stock name/ticker
# 2. IDX keterbukaan informasi — official corporate announcements

import feedparser
import requests

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id"

IDX_ANNOUNCEMENTS = "https://www.idx.co.id/umbraco/Surface/ListedCompany/GetAnnouncement"

STOCK_NAMES = {
    "BBCA.JK": "Bank Central Asia BCA",
    "BBRI.JK": "Bank Rakyat Indonesia BRI",
    "TLKM.JK": "Telkom Indonesia TLKM",
    "ASII.JK": "Astra International ASII",
    "GOTO.JK": "GoTo Gojek Tokopedia GOTO",
    "BREN.JK": "Barito Renewables BREN",
    "BMRI.JK": "Bank Mandiri BMRI",
    "UNVR.JK": "Unilever Indonesia UNVR",
}

def fetch_news(symbol: str, max_age_hours: int = 24) -> list[dict]:
    """
    Fetch recent news for a stock from Google News RSS.
    Returns list of: { headline, source, age_hours, url }
    max_age_hours: filter out news older than this
    """

def fetch_idx_announcements(symbol: str) -> list[dict]:
    """
    Fetch official IDX corporate announcements.
    Returns list of: { title, date, type, url }
    Types include: keterbukaan, dividen, RUPS, rights issue, etc.
    """

def score_news_sentiment(news_items: list[dict], announcements: list[dict]) -> dict:
    """
    Uses Claude to score news sentiment for a stock.
    Returns: {
        score: int (0-20 pts),
        sentiment: "POSITIVE" | "NEUTRAL" | "NEGATIVE",
        key_headline: str,
        news_items: list (top 3 most relevant)
    }

    Scoring rules:
    +20: earnings beat, dividend, gov contract, rating upgrade
    +12: analyst upgrade, sector tailwind
    +5:  neutral routine disclosure
    0:   no recent news
    -8:  earnings miss, analyst downgrade
    -15: rights issue, OJK issue, insider selling
    """
```

---

### scorer.py

```python
# Combines all 4 layers into a single verdict

def score_stock(
    symbol: str,
    technical: dict,    # from indicators.py
    forecast: dict,     # from forecaster.py
    foreign: dict,      # from fetcher.py
    news: dict,         # from news_fetcher.py
) -> dict:
    """
    Returns full scoring result:
    {
        symbol: str,
        total_score: int (0-100),
        verdict: "STRONG_BUY" | "BUY" | "WATCH" | "SKIP",
        scores: {
            technical: int (0-35),
            prophet: int (0-25),
            foreign: int (0-20),
            news: int (0-20),
        },
        reasons: list[str],   # human-readable reason per layer
    }
    """
```

---

### ai_agent.py

```python
# Claude API — reads all 4 layers and returns structured trade verdict
# Same pattern as crypto bot ai_agent.py

SYSTEM_PROMPT = """
You are a professional IDX (Indonesia Stock Exchange) analyst.
Analyze the provided stock data and give a trade recommendation.

Fee structure: Buy 0.15% + Sell 0.25% = 0.40% round trip.
Minimum viable target: +1.5%. Recommended R/R: 1:2 minimum.
T+2 settlement: capital locked for 2 days after sell.
Minimum trade unit: 1 lot = 100 shares.

Respond with ONLY a valid JSON object:
{
    "action": "ENTER" | "WAIT" | "SKIP",
    "confidence": int (0-100),
    "entry_price": float,
    "target_price": float,
    "target_pct": float,
    "stop_loss": float,
    "stop_loss_pct": float,
    "risk_reward": float,
    "hold_duration": "intraday" | "1-2 days" | "3-5 days",
    "lots": int,
    "capital_idr": int,
    "risk_level": "LOW" | "MEDIUM" | "HIGH",
    "reasoning": str (2-3 sentences referencing specific indicators)
}
"""

# Prompt includes: technical scores, Prophet forecast, foreign flow,
# news summary, current price, target/SL suggestions from scorer
```

---

### scheduler.py

```python
# WIB-aware scheduler — only runs during IDX market hours
# Uses APScheduler with Asia/Jakarta timezone

from apscheduler.schedulers.blocking import BlockingScheduler
import pytz

WIB = pytz.timezone("Asia/Jakarta")

def is_market_open() -> bool:
    """Returns True if currently in Session 1 or Session 2"""

def is_trading_day() -> bool:
    """Returns True if today is a weekday (Mon-Fri)
    TODO: Add IDX holiday calendar check"""

# Jobs:
# cron 08:45 WIB Mon-Fri  → run_presession_briefing(1)
# cron 12:30 WIB Mon-Fri  → run_presession_briefing(2)
# cron 16:00 WIB Mon-Fri  → run_eod_summary()
# interval 5min           → run_scan() [only if is_market_open()]
```

---

### telegram_bot.py

```python
# Same raw requests polling pattern as crypto bot
# Commands:

COMMANDS = {
    "/status":    show bot status, market status, today P&L
    "/stocks":    show current watchlist with latest scores
    "/briefing":  manually trigger pre-session briefing
    "/forecast":  show Prophet forecasts for all stocks
    "/flow":      show today's IDX foreign flow summary
    "/news":      show recent news for all watchlist stocks
    "/pnl":       today's signal performance
    "/add BBCA":  add stock to watchlist
    "/remove BBCA": remove stock from watchlist
    "/help":      list all commands
}
```

---

### firestore_client.py

```python
# Collections:
#
# idx_snapshots/{symbol}/snapshots/{timestamp}
#   Full 5-min data point: price, scores, indicators, forecast
#
# idx_signals/{timestamp}_{symbol}
#   Only BUY or STRONG_BUY signals with AI verdict
#
# idx_forecasts/{symbol}
#   Latest Prophet forecast, updated daily
#
# idx_news/{symbol}
#   Latest news items, TTL 24hr
#
# idx_bot_config/settings
#   { stocks: [...], capital: int, updated_at: timestamp }
```

---

### risk_manager.py

```python
# IDX-specific risk management

def calc_lot_size(symbol: str, price: float, confidence: int) -> int:
    """
    Calculate how many lots to buy.
    Base: IDX_CAPITAL * 10% per stock / (price * 100)
    Scale up to IDX_MAX_LOTS based on confidence.
    Always rounds DOWN to nearest lot.
    Returns: int (number of lots, min 1)
    """

def check_risk_gates() -> tuple[bool, str]:
    """
    Same pattern as crypto bot:
    Gate 1: daily loss limit not breached
    Gate 2: not already holding this stock (T+2 check)
    Gate 3: enough capital (IDX_CAPITAL / max simultaneous positions)
    """

def get_t2_positions() -> list[str]:
    """
    Returns list of stocks currently in T+2 settlement
    (sold but capital not yet released)
    """
```

---

## Data Flow

```
Every 5 minutes during market hours:
    for each stock in watchlist:
        1. fetch_candles()         → OHLCV 5-min
        2. calculate_indicators()  → RSI, MA, volume
        3. get_prophet_forecast()  → cached, updated daily
        4. fetch_foreign_flow()    → IDX endpoint
        5. get_news_sentiment()    → cached per 30min
        6. score_stock()           → 0-100 total score
        7. save_snapshot()         → Firestore
        8. if score >= MIN_SCORE:
              get_ai_verdict()     → Claude API
              send_signal()        → Telegram
              save_signal()        → Firestore

At 08:45 WIB:
    for each stock:
        1. fetch daily OHLCV (yesterday close)
        2. run Prophet if not cached today
        3. fetch JCI overnight sentiment
        4. fetch foreign flow (yesterday's data)
        5. fetch latest news
        6. score all stocks
        7. generate briefing → Claude synthesizes → Telegram

At 12:30 WIB:
    Same as above but with Session 1 recap included

At 16:00 WIB:
    Summarize: signals fired, hypothetical P&L, tomorrow outlook
```

---

## yfinance Notes

```python
import yfinance as yf

# Intraday (5-min candles, last 1 day):
df = yf.download("BBCA.JK", period="1d", interval="5m")

# Daily (for Prophet, 1 year):
df = yf.download("BBCA.JK", period="1y", interval="1d")

# Live quote:
ticker = yf.Ticker("BBCA.JK")
info = ticker.fast_info  # price, volume, market cap

# Important: yfinance uses Jakarta timezone for IDX
# Candles have timezone-aware index — convert to WIB for display
```

---

## Prophet Notes

```python
from prophet import Prophet

# Input must be DataFrame with columns:
# ds: datetime, y: float (close price)

# IDX has weekly seasonality (Mon–Fri trading)
# and yearly seasonality (Ramadan dip, year-end rally)

model = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False,      # no intraday pattern at daily level
    changepoint_prior_scale=0.05, # conservative — IDX less volatile than crypto
)

# Make_future_dataframe skips weekends automatically
# with freq='B' (business days)
future = model.make_future_dataframe(periods=5, freq='B')
forecast = model.predict(future)
# forecast['yhat'][-5:]  → next 5 business days predicted price
# forecast['yhat_lower'] / forecast['yhat_upper'] → confidence interval
```

---

## IDX Foreign Flow Endpoint

```python
# IDX publishes broker summary data publicly
# No auth required

# Foreign net flow (market-wide):
url = "https://www.idx.co.id/umbraco/Surface/TradingSummary/GetForeignFlow"

# Per-stock broker summary:
url = "https://www.idx.co.id/umbraco/Surface/TradingSummary/GetBrokerSummary"
params = { "stockCode": "BBCA", "tradingDate": "2026-04-14" }

# Response includes: foreignBuyVal, foreignSellVal, foreignNetVal (in IDR)
# If foreignNetVal > 0 → foreigners are net buyers (bullish signal)
```

---

## Google News RSS

```python
import feedparser

def fetch_google_news(query: str) -> list[dict]:
    url = f"https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id"
    feed = feedparser.parse(url)
    return [
        {
            "headline": entry.title,
            "source":   entry.source.title if hasattr(entry, 'source') else "",
            "url":      entry.link,
            "published": entry.published,
        }
        for entry in feed.entries[:5]  # top 5 results
    ]

# Good queries:
# "BBCA saham" → Indonesian news about BCA stock
# "Bank Central Asia laba" → earnings news
# "BBCA BEI" → IDX-specific news
```

---

## requirements.txt

```
yfinance==0.2.40
prophet==1.1.5
feedparser==6.0.11
pystan==3.9.1
pandas==2.1.4
numpy==1.26.4
ta==0.11.0
firebase-admin==6.5.0
APScheduler==3.10.4
python-dotenv==1.0.1
requests==2.31.0
pytz==2024.1
```

> Note: `prophet` + `pystan` installation is slow (~3-4 min on Railway).
> Add `--no-build-isolation` flag if pystan fails to build.

---

## Railway Deployment

```
# Same Railway project as crypto bot
# Add as second service: "idx-bot"
# Procfile:
worker: python main.py

# Key env vars (reuse from crypto bot project):
FIREBASE_CRED_JSON
ANTHROPIC_API_KEY
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID

# New IDX-specific:
IDX_CAPITAL=5000000
IDX_MAX_LOTS_PER_STOCK=5
IDX_DAILY_LOSS_LIMIT_PCT=3.0
IDX_MIN_SCORE=60
```

---

## Dashboard Integration

Add a new tab to the existing React crypto dashboard on Vercel.

New tab: **"IDX Stocks"**

Components to build:
- `IDXScores.jsx` — grid of 8 stocks with score badges (same pattern as CoinScores)
- `IDXForecasts.jsx` — Prophet chart per stock (5-day price band)
- `IDXSignals.jsx` — live signal feed
- `IDXForeignFlow.jsx` — bar chart of net foreign flow per stock
- `IDXSettings.jsx` — manage watchlist, capital settings

Firebase collections to read:
- `idx_snapshots` — live scores
- `idx_signals` — signal history
- `idx_forecasts` — Prophet output
