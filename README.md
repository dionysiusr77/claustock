# Claustock IDX v2

Daily stock scanner and morning briefing bot for the Indonesia Stock Exchange (IDX), delivered via Telegram. Powered by Invezgo API for market data and Claude AI for briefing synthesis.

---

## Features

- **Full universe scan** — scans IDX80 / LQ45 / COMPOSITE daily at 16:30 WIB after market close
- **Morning briefing** — AI-written Indonesian-language briefing delivered at 08:30 WIB with actionable picks
- **6-layer scoring model** — Trend + Momentum + Volume + Pattern + Foreign Flow + Market Breadth
- **RSI divergence detection** — bullish/hidden bullish bonus, bearish penalty and hard disqualifier
- **ATR-based trade levels** — Entry, Target, Invalidation with R:R ≥ 1.5 after IDX fees
- **Foreign flow tracking** — per-stock consecutive buy/sell streak via Invezgo
- **Telegram commands** — `/scan`, `/briefing`, `/pick BBCA`, `/status`
- **Firebase Firestore** — persists scans and briefings for on-demand retrieval

---

## Architecture

```
main.py
├── Scheduler (APScheduler via PTB JobQueue)
│   ├── 16:30 WIB Mon–Fri → job_eod_scan      → screener.run_scan()
│   └── 08:30 WIB Mon–Fri → job_morning_briefing → Firestore → broadcast
│
└── Telegram bot (python-telegram-bot v20)
    ├── /scan     → screener.run_scan() + ai_briefing.build_briefing()
    ├── /briefing → Firestore load → send
    ├── /pick     → screener.scan_single()
    └── /status   → market hours + config
```

### Scan pipeline (`screener.py`)

1. `get_universe()` — full IDX equity list (warrants/rights excluded)
2. `filter_liquid()` — price range + min avg daily value (Invezgo screener or OHLCV fallback)
3. `fetch_daily_batch()` — 1y daily OHLCV per stock
4. `fetch_market_breadth()` — IHSG + sector indices
5. `compute_all()` + `latest_snapshot()` — all technical indicators
6. `score_stock()` — first-pass scoring without foreign flow
7. `fetch_foreign_flow_stock()` — foreign flow for top 2× candidates only
8. Re-score top candidates with real foreign data
9. Hard disqualifiers — BEARISH divergence, score < MIN_SCORE, no viable R:R
10. Return top N candidates for AI briefing

---

## Scoring Model

| Layer | Weight | Signals |
|---|---|---|
| Trend | 25 | EMA20/50/200 alignment |
| Momentum | 20 | RSI zone, MACD cross, Stochastic |
| Volume | 20 | Volume ratio vs 20-day MA, OBV slope |
| Pattern | 15 | Breakout, candlestick patterns |
| Foreign | 15 | Net foreign direction + consecutive streak |
| Breadth | 5 | IHSG + sector direction |

**RSI thresholds:** oversold < 30, overbought > 70, sweet spot 45–65

**Divergence modifiers:** BULLISH +5, HIDDEN_BULLISH +3, BEARISH −10, HIDDEN_BEARISH −5

**Setups:** `BREAKOUT` · `PULLBACK` · `OVERSOLD_BOUNCE` · `MOMENTUM_CONTINUATION` · `FOREIGN_ACCUMULATION`

---

## Setup

### 1. Clone and install

```bash
git clone <repo>
cd claustock
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

Copy `.env.example` to `.env` and fill in:

```env
ANTHROPIC_API_KEY=sk-ant-...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_IDS=123456789,987654321
INVEZGO_API_KEY=<jwt_token>
FIREBASE_CRED_JSON={"type":"service_account",...}   # single line JSON

# Optional overrides
UNIVERSE=IDX80           # LQ45 | IDX80 | COMPOSITE
MIN_SCORE=60
TOP_N_AI=15
CAPITAL_IDR=10000000
MAX_POSITIONS=5
```

> **FIREBASE_CRED_JSON** must be a single-line JSON string. Multi-line format will fail python-dotenv parsing.

### 3. Run locally

```bash
python main.py
```

### 4. Deploy to Railway

Push to your Railway-linked branch. Set all environment variables in the Railway dashboard (not via `.env`).

> Run only **one instance** per bot token at a time — parallel instances cause a Telegram polling conflict.

---

## Data Sources

| Data | Source |
|---|---|
| Stock OHLCV | Invezgo API (`/analysis/chart/stock/{code}`) |
| IHSG index | Invezgo API (`/analysis/chart/index/COMPOSITE`) |
| Sector indices | yfinance (`^JKFIN`, `^JKCONS`, etc.) |
| Per-stock foreign flow | Invezgo `get_summary_stock(investor="f")` |
| Market foreign flow | Invezgo `get_top_foreign()` |
| Stock universe | Invezgo `get_stock_list()` (hardcoded IDX80 fallback) |

---

## Telegram Commands

| Command | Description |
|---|---|
| `/scan` | Run a live D-1 scan now and generate briefing |
| `/briefing` | Send today's saved briefing (from Firestore) |
| `/pick BBCA` | Deep-dive score for a single stock |
| `/status` | Bot status, market hours, current config |
| `/help` | Command list |

---

## Fee Structure (IDX)

| Fee | Rate |
|---|---|
| Buy | 0.15% |
| Sell | 0.25% (incl. 0.10% PPh) |
| Round-trip | 0.40% |
| Min viable profit | 1.5% net after fees |

Trade levels use 1.5× ATR for stop loss with minimum R:R of 1.5.

---

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point, scheduler setup |
| `config.py` | All constants and env var loading |
| `invezgo_client.py` | Invezgo API client (OHLCV, foreign flow, universe) |
| `indicators.py` | Technical indicator calculations (EMA, RSI, MACD, ATR, OBV, divergence) |
| `patterns.py` | Candlestick pattern detection |
| `scorer.py` | 6-layer scoring, trade levels, setup classification |
| `screener.py` | Full D-1 scan pipeline |
| `market_breadth.py` | IHSG Fear/Greed, A/D ratio, sector rotation |
| `ai_briefing.py` | Claude API briefing synthesis + Telegram formatter |
| `firestore_client.py` | Firebase Firestore persistence |
| `telegram_bot.py` | PTB v20 bot, commands, scheduled job callbacks |
| `fetcher.py` | yfinance fallback (sector breadth) |
| `universe.py` | Hardcoded LQ45/IDX80 fallback lists |
| `debug_invezgo.py` | Diagnostic script for API response inspection |

---

## Requirements

- Python 3.11+
- Invezgo Prime API access
- Anthropic API key (Claude Haiku)
- Telegram bot token (via BotFather)
- Firebase project with Firestore enabled
