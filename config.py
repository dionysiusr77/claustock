import os
from dotenv import load_dotenv

load_dotenv()

# ── Stocks ────────────────────────────────────
STOCKS = [
    "BBCA.JK",
    "BBRI.JK",
    "TLKM.JK",
    "ASII.JK",
    "GOTO.JK",
    "BREN.JK",
    "BMRI.JK",
    "UNVR.JK",
]

# ── Timeframes ────────────────────────────────
CANDLE_INTERVAL = "5m"    # 5-minute candles during market
CANDLE_LIMIT    = 60      # 60 candles = 5 hours history
DAILY_CANDLES   = 365     # 1yr for Prophet training

# ── Market hours (WIB = UTC+7) ────────────────
MARKET_TZ      = "Asia/Jakarta"
SESSION1_START = (9, 0)    # 09:00
SESSION1_END   = (12, 0)   # 12:00
SESSION2_START = (13, 0)   # 13:00
SESSION2_END   = (15, 49)  # 15:49
BRIEFING1_TIME = (8, 45)   # Pre-session 1 briefing
BRIEFING2_TIME = (12, 30)  # Pre-session 2 briefing
EOD_TIME       = (16, 0)   # End of day summary

# ── Scan interval ─────────────────────────────
SCAN_INTERVAL_SEC = 300    # 5 minutes during market hours

# ── Scoring weights ───────────────────────────
WEIGHT_TECHNICAL = 35
WEIGHT_PROPHET   = 25
WEIGHT_FOREIGN   = 20
WEIGHT_NEWS      = 20

# ── Risk ──────────────────────────────────────
IDX_CAPITAL        = int(os.getenv("IDX_CAPITAL", "5000000"))
IDX_MAX_LOTS       = int(os.getenv("IDX_MAX_LOTS_PER_STOCK", "5"))
IDX_DAILY_LOSS_PCT = float(os.getenv("IDX_DAILY_LOSS_LIMIT_PCT", "3.0"))
IDX_MIN_SCORE      = int(os.getenv("IDX_MIN_SCORE", "60"))

# ── Fees ──────────────────────────────────────
BUY_FEE_PCT    = 0.0015   # 0.15%
SELL_FEE_PCT   = 0.0025   # 0.25% (incl. 0.10% tax)
FEE_RT         = BUY_FEE_PCT + SELL_FEE_PCT  # 0.40% round trip
MIN_TARGET_PCT = 1.5       # minimum viable target after fees

# ── Firebase ──────────────────────────────────
FIREBASE_CRED_JSON = os.getenv("FIREBASE_CRED_JSON", "")

# ── Anthropic ─────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = "claude-haiku-4-5-20251001"

# ── Telegram ──────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
