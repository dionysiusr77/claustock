# IDX Stock Bot — Project Brief
> Read this first. This is the complete context for building the IDX Stock Bot.

---

## Who You're Building For

Dion — sound engineer, barista, developer. Already runs a **crypto scalping bot** on Railway (Python, Firebase, Telegram). This IDX bot is a companion project using ~70% of the same patterns. He's comfortable with Python, Firebase, Vercel, and Railway.

---

## What This Bot Does

A **signal-only** stock trading assistant for the Indonesia Stock Exchange (IDX / Bursa Efek Indonesia). It cannot auto-execute orders (no Indonesian broker exposes a public trading API). Instead it:

1. Fetches IDX stock data via **yfinance** (free, no auth)
2. Runs **4-layer analysis** per stock
3. Sends **pre-session briefings** to Telegram before each market session
4. Sends **live signals** during market hours
5. Shows everything on a **React dashboard** (Vercel)

---

## Market Schedule (WIB = UTC+7)

```
Weekdays only. No weekends. No public holidays.

Session 1:  09:00 – 12:00 WIB
Session 2:  13:00 – 15:49 WIB  (Friday: 14:00 – 15:49)

Pre-session briefings:
  08:45 WIB  →  Session 1 briefing sent to Telegram
  12:30 WIB  →  Session 2 briefing sent to Telegram

End of day:
  16:00 WIB  →  Daily summary sent

Outside market hours: bot is fully idle. No API calls.
```

---

## Stock Watchlist (LQ45 Blue Chips)

```python
STOCKS = [
    "BBCA.JK",   # Bank Central Asia — highest liquidity
    "BBRI.JK",   # Bank Rakyat Indonesia — high volume
    "TLKM.JK",   # Telkom Indonesia — stable oscillation
    "ASII.JK",   # Astra International — wide daily range
    "GOTO.JK",   # GoTo Group — high volatility
    "BREN.JK",   # Barito Renewables — hot sector
    "BMRI.JK",   # Bank Mandiri — reliable TA
    "UNVR.JK",   # Unilever Indonesia — defensive
]
```

---

## 4-Layer Scoring System (0–100 pts)

```
Layer 1 — Technical Analysis     35 pts
  RSI(14), MA(7/30), volume ratio, candle pattern
  Same logic as crypto bot — just longer timeframe (5min candles)

Layer 2 — Prophet Forecast        25 pts
  Facebook Prophet trained on 1yr daily OHLCV
  Predicts next 5-day price range + trend direction
  Run once at startup + retrain weekly

Layer 3 — Foreign Flow            20 pts
  IDX publishes daily net foreign buy/sell per stock
  Source: idx.co.id public endpoint (no auth needed)
  Foreigners net buying 3+ days = strong bullish signal

Layer 4 — News Sentiment          20 pts
  Sources: IDX official announcements + Google News RSS
  Claude reads headlines and scores sentiment
  Earnings beat/dividend = +20, Rights issue/OJK issue = -15
```

### Score → Verdict

```
80–100  STRONG BUY   — high conviction, all layers aligned
60–79   BUY          — good setup, most layers positive
40–59   WATCH        — mixed signals, wait for confirmation
0–39    SKIP         — avoid
```

---

## Fee Structure (Critical for Targets)

```
Buy fee:    0.15% (broker commission)
Sell fee:   0.25% (broker commission + 0.10% tax + IDX levy)
Round trip: 0.40%

Minimum profitable target: +1.5%
Recommended target:        +2.0% to +3.0%
Stop loss:                 -1.0% to -1.5%
Minimum R/R ratio:         1:2 after fees
```

---

## Pre-Session Briefing Format

```
📋 SESSION 1 BRIEFING — Mon 14 Apr 2026
━━━━━━━━━━━━━━━━━━━━━━━━━━━
🌍 Market Context
  JCI yesterday: +0.8% | Foreign: Net BUY Rp320B
  Wall Street overnight: S&P +0.4%

📈 Prophet Forecast (5-day)
  BBCA  9,150 → 9,400 (+2.7%)  🟢 UPTREND
  BBRI  4,820 → 4,950 (+2.7%)  🟢 UPTREND
  GOTO  71    → 68    (-4.2%)  🔴 DOWNTREND

🎯 Top Picks — Session 1
  1. BBCA — Entry 9,150 | Target 9,400 | SL 9,050
     Score: 82/100 | Hold: intraday | Lots: 2
     News: Laba Q1 naik 8% YoY ✅

  2. BBRI — Entry 4,820 | Target 4,950 | SL 4,760
     Score: 74/100 | Hold: 1–2 days | Lots: 3

⚠️ Avoid: GOTO (downtrend), UNVR (earnings tomorrow)
```

---

## Live Signal Format (During Market Hours)

```
📡 SIGNAL — BBCA.JK
━━━━━━━━━━━━━━━━━━━━
💰 Price: 9,175  (+0.8%)
📊 Score: 78/100 → BUY

Technical:    26/35  RSI 42, MA uptrend, vol 1.8x
Prophet:      20/25  +2.7% trend next 5d
Foreign flow: 16/20  Net buy Rp180B (3 days)
News:         16/20  Laba Q1 beat estimates ✅

🤖 AI Verdict: ENTER (81% confidence)
🚀 Entry:   9,175
✅ Target:  9,350  (+1.9%)
🛑 SL:      9,075  (-1.1%)
⚖️ R/R:     1:1.9
💼 Lots:    2 (~Rp1.84M)
```

---

## Tech Stack

```
Data:           yfinance (free, IDX with .JK suffix)
Forecasting:    prophet (Meta's time-series library)
Technical TA:   ta (same as crypto bot)
News:           Google News RSS + IDX announcements scraper
AI:             Claude Haiku (Anthropic API)
Telegram:       python-telegram-bot style (same raw requests pattern)
Database:       Firebase Firestore (same project as crypto bot)
Dashboard:      React + Vite (Vercel — new tab in existing dashboard)
Scheduler:      APScheduler (same as crypto bot)
Hosting:        Railway (second service on existing project)
Language:       Python 3.11
```

---

## Key Differences from Crypto Bot

| Aspect | Crypto Bot | IDX Bot |
|--------|-----------|---------|
| Runtime | 24/7 | Weekdays 08:00–16:00 WIB only |
| Scan interval | 90 seconds | 5 minutes |
| Settlement | Instant | T+2 (warn user) |
| Min lot | Any amount | 100 shares = 1 lot |
| Auto-execute | Yes (Bybit API) | No (no broker API) |
| Whale detection | Volume + order book | Foreign flow data |
| Extra forecasting | None | Prophet 5-day + news |
| Scan frequency | Always | Only during market hours |

---

## Reusable from Crypto Bot

These patterns are already proven — copy/adapt them:

- `telegram_bot.py` — same raw requests polling pattern
- `firestore_client.py` — same Firestore read/write structure
- `indicators.py` — same RSI, MA, volume ratio logic
- `ai_agent.py` — same Claude API call pattern
- `risk_manager.py` — adapt for IDX (max lots, daily loss limit)
- `config.py` — same env var pattern
- `main.py` — adapt scheduler for market hours only

---

## Phase Plan

```
Phase 1  Data fetcher + technical indicators + Firestore
Phase 2  Prophet forecasting + foreign flow
Phase 3  News sentiment (Google News RSS + IDX announcements)
Phase 4  Pre-session briefings + live signals → Telegram
Phase 5  AI verdict synthesis (Claude)
Phase 6  React dashboard tab (add to existing crypto dashboard)
```

---

## Environment Variables Needed

```bash
# Reuse from crypto bot (same Railway project):
FIREBASE_CRED_JSON=...
ANTHROPIC_API_KEY=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# IDX bot specific:
IDX_CAPITAL=5000000          # Trading capital in IDR (e.g. Rp5,000,000)
IDX_MAX_LOTS_PER_STOCK=5     # Max lots per position
IDX_DAILY_LOSS_LIMIT_PCT=3.0 # Stop if down 3% on the day
IDX_MIN_SCORE=60             # Minimum score to trigger signal
```

---

## Start Here

Build `Phase 1` first:

1. `config.py` — IDX-specific settings
2. `fetcher.py` — yfinance OHLCV + quote fetcher with .JK suffix handling
3. `indicators.py` — RSI(14), MA(7/30), volume ratio (5-min candles)
4. `firestore_client.py` — save snapshots to `idx_snapshots/{symbol}/snapshots`
5. `scheduler.py` — APScheduler that only runs during WIB market hours
6. `main.py` — orchestrator

Test with: `python main.py` — should print live quotes for all 8 stocks during market hours.
