# IDX Stock Bot

A Telegram bot that monitors the Indonesia Stock Exchange (IDX/BEI), scores intraday scalp candidates, and sends pre-session briefings — powered by Claude AI and Prophet forecasting.

---

## Features

- **Pre-session briefings** — automatic Telegram messages at 08:45 and 12:30 WIB with today's top 2 scalp candidates
- **D-1 screener** (Session 1): screens yesterday's daily data (RSI, Volume, MACD, Bollinger Bands)
- **S1 screener** (Session 2): re-screens using today's Session 1 intraday data (5m candles), volume compared to last 5 days' Session 1 average
- **Intraday scalp scanner**: scans only the D-1 screened stocks every 5 minutes; fires entry alerts when criteria are met
- **Two-watchlist strategy**: SCALP (mean-reversion, closes at 15:49) + MOMENTUM (trend-continuation, carried overnight up to 3 days)
- **Bullish divergence detection**: price making lower lows while RSI makes higher lows; confirmed divergence earns bonus points and `STRONG_SCALP` classification
- **Sector correlation guard**: caps same-sector positions to avoid over-concentration
- **5-day Prophet forecast**: per-stock price forecasting using Facebook Prophet
- **News sentiment scoring**: Google News RSS + IDX official announcements (keterbukaan informasi), scored 0–20 by Claude Haiku
- **Foreign flow tracking**: consecutive-day net buy/sell streaks from IDX broker summary API
- **Whale scanner**: detects abnormal volume surges across the LQ45/IDX80 universe
- **Firestore persistence**: scalp and momentum positions survive restarts; watchlist is dynamic
- **Telegram commands**: `/briefing`, `/scalps`, `/analyze`, `/add`, `/remove`, and more
- **Group chat support**: allowlist-based access for personal chats and group chats

---

## Architecture

```
main.py              — scheduler, scan loop, command handlers
├── fetcher.py       — yfinance candles, foreign flow, JCI summary
├── indicators.py    — RSI, MACD, Bollinger Bands, divergence detection
├── screener.py      — D-1 daily screener + S1 intraday screener
├── scalper.py       — two-watchlist engine (SCALP + MOMENTUM)
├── forecaster.py    — Facebook Prophet 5-day price forecast
├── news_fetcher.py  — Google News RSS + IDX announcements + Claude sentiment
├── scorer.py        — composite score (Technical 35 + Prophet 25 + Foreign 20 + News 20)
├── ai_agent.py      — Claude AI verdict on qualifying signals
├── analyzer.py      — on-demand deep analysis (/analyze command)
├── risk_manager.py  — position sizing, daily loss gate
├── firestore_client.py — Firebase Firestore read/write
├── scheduler.py     — APScheduler + IDX holiday calendar (WIB timezone)
├── telegram_bot.py  — message sending, briefing formatting, command polling
└── config.py        — all constants and env-var bindings
```

---

## Scoring System

### Composite score (0–100)

| Layer | Weight | Source |
|---|---|---|
| Technical | 35 pts | RSI, MACD, MA trend, volume, candle patterns, divergence bonus |
| Prophet forecast | 25 pts | 5-day trend direction and magnitude |
| Foreign flow | 20 pts | Consecutive net-buy days from IDX |
| News sentiment | 20 pts | Claude Haiku scoring of headlines + IDX announcements |

### D-1 / S1 screener (0–100)

| Criterion | Max pts | Pass condition |
|---|---|---|
| RSI(14) | 30 | < 30 (rising vs avg = 30 pts; flat = 15 pts) |
| Volume ratio | 25 | ≥ 2× avg = 25, ≥ 1.5× = 18, ≥ 1× = 10 |
| MACD(12,26,9) | 25 | CROSS = 25, APPROACHING = 18, BULLISH = 10 |
| Bollinger Bands(20,2) | 20 | ≤ lower band = 20, ≤ mid×0.995 = 12, ≤ mid = 5 |

All 4 must pass to qualify. D-1 uses yesterday's daily close; S1 uses today's 09:00–12:00 intraday candles with volume baseline from the last 5 days' Session 1 averages.

---

## Briefing Schedule

| Time (WIB) | Event | Data source |
|---|---|---|
| 08:45 | Pre-session 1 briefing | D-1 daily screener (yesterday's close) |
| 12:30 | Pre-session 2 briefing | S1 intraday screener (today's 09:00–12:00) |
| 16:00 | End-of-day summary | Closed positions + daily P&L |

---

## Scalp Classification

```
classify_signal(result):
  STRONG_SCALP  — classic gates + confirmed bullish divergence
  SCALP         — classic gates, no divergence
  MOMENTUM      — RSI > 50, news score ≥ 12, total score ≥ 60
  SKIP          — none of the above

Classic gates:
  rsi < 40, rsi rising, drop 0.5–6% from open,
  volume_ok, macd_ok, base_score ≥ 55
```

Divergence detection uses a 40-bar window, finds price swing lows, and counts pairs where price makes a lower low but RSI makes a higher low. 1 pair = +12 pts bonus; 2+ pairs = +20 pts bonus.

---

## Telegram Commands

| Command | Description |
|---|---|
| `/briefing` | Trigger today's briefing now |
| `/scalps` | Live scalp + momentum watchlist with P&L |
| `/scalpadd BBCA` | Manually add stock to scalp watchlist |
| `/analyze BBCA` | Full fundamental + technical deep-dive |
| `/stocks` | Current composite scores for all watchlist stocks |
| `/forecast` | 5-day Prophet forecasts |
| `/flow` | Today's foreign flow per stock |
| `/news` | Latest news headlines and sentiment |
| `/add BBCA` | Add stock to watchlist |
| `/remove BBCA` | Remove stock from watchlist |
| `/pnl` | Today's signal P&L |
| `/status` | Bot and market status |
| `/help` | Command list |

---

## Setup

### Prerequisites

- Python 3.11+
- Firebase project with Firestore enabled
- Telegram bot token ([@BotFather](https://t.me/BotFather))
- Anthropic API key

### Installation

```bash
git clone https://github.com/your-username/claustock.git
cd claustock
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file:

```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
TELEGRAM_ALLOWED_USER_IDS=310977969,-5144122635

ANTHROPIC_API_KEY=your_anthropic_key

FIREBASE_CRED_JSON={"type":"service_account",...}   # full JSON string

# Optional
IDX_CAPITAL=5000000           # capital in IDR for lot sizing
IDX_MAX_LOTS_PER_STOCK=5
IDX_DAILY_LOSS_LIMIT_PCT=3.0
IDX_MIN_SCORE=60
REQUIRE_DIVERGENCE=false      # set true to only fire STRONG_SCALP signals
```

`TELEGRAM_ALLOWED_USER_IDS` accepts a comma-separated list of personal user IDs and group chat IDs (negative numbers). Anyone not on the list is silently ignored.

### Run

```bash
python main.py
```

For production, deploy as a worker dyno on Railway or any always-on service.

---

## Data Sources

| Data | Source |
|---|---|
| OHLCV candles | Yahoo Finance (`yfinance`) |
| Foreign flow | IDX Broker Summary API (`idx.co.id`) |
| JCI index | Yahoo Finance (`^JKSE`) |
| News headlines | Google News RSS (Indonesian, `hl=id`) |
| Corporate announcements | IDX keterbukaan informasi API |

---

## Firestore Collections

| Collection | Purpose |
|---|---|
| `idx_bot_watchlist` | Persisted dynamic watchlist |
| `idx_bot_snapshots/{symbol}` | Latest scan snapshot per stock |
| `idx_bot_signals` | Fired signal history |
| `idx_scalp_positions` | Active/closed scalp positions (date-scoped) |
| `idx_momentum_positions` | Active momentum positions (multi-day) |
| `idx_bot_config/daily_scalp_watchlist` | D-1 screened candidates (Session 1) |
| `idx_bot_config/s1_watchlist` | S1 screened candidates (Session 2) |

---

## Market Hours (WIB, UTC+7)

| Session | Hours | Notes |
|---|---|---|
| Session 1 | 09:00–12:00 | All days |
| Break | 12:00–13:30 | Session 2 briefing at 12:30 |
| Session 2 | 13:30–15:49 | Fridays: 14:00–15:49 |

IDX holidays for 2026 are hard-coded in `scheduler.py`.

---

## License

MIT
