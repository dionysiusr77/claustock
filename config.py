import os
from dotenv import load_dotenv

load_dotenv()

# ── Stocks ────────────────────────────────────
# Default watchlist — loaded into runtime watchlist on first start
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

# Broader universe scanned for whale detection (high volume surges)
# LQ45 + IDX80 liquid names — not all are in the default watchlist
UNIVERSE = [
    # Default watchlist
    "BBCA.JK", "BBRI.JK", "TLKM.JK", "ASII.JK",
    "GOTO.JK", "BREN.JK", "BMRI.JK", "UNVR.JK",
    # Extended LQ45 / IDX80
    "ANTM.JK", "INDF.JK", "KLBF.JK", "MDKA.JK",
    "MEDC.JK", "MIKA.JK", "PGAS.JK", "PTBA.JK",
    "SMGR.JK", "TBIG.JK", "TPIA.JK", "ADRO.JK",
    "AMRT.JK", "CPIN.JK", "EMTK.JK", "EXCL.JK",
    "HMSP.JK", "ICBP.JK", "INCO.JK", "INKP.JK",
    "ISAT.JK", "MNCN.JK", "SIDO.JK", "TOWR.JK",
]

# Whale scanner settings
WHALE_TOP_N           = 3     # number of high-volume stocks to surface per scan
WHALE_VOL_THRESHOLD   = 3.0   # volume ratio must be >= this to qualify
WHALE_MIN_SCORE       = 55    # minimum tech score (0-35 scale) to auto-add
WHALE_AUTO_ADD        = True  # if True, auto-add to watchlist; else just notify

# ── Scalper settings ──────────────────────────
SCALP_MAX_RSI            = 50     # RSI must be BELOW this to qualify as scalp
SCALP_MIN_DROP_FROM_OPEN = 0.5    # minimum % drop from open price to qualify
SCALP_MAX_DROP_FROM_OPEN = 6.0    # maximum % drop — beyond this is a crash
SCALP_MIN_SCORE          = 55     # minimum scalp score (0–100)
REQUIRE_DIVERGENCE       = os.getenv("REQUIRE_DIVERGENCE", "false").lower() == "true"

# ── Momentum settings ─────────────────────────
MOMENTUM_MIN_RSI         = 50     # RSI must be ABOVE this for momentum
MOMENTUM_MIN_NEWS_SCORE  = 12     # minimum news score (0–20) to qualify
MOMENTUM_MIN_SCORE       = 60     # minimum total score (0–100) for momentum
MOMENTUM_HOLD_DAYS       = 3      # hold for up to 3 trading days
MOMENTUM_TARGET_PCT      = 3.0    # take-profit target (%)
MOMENTUM_STOP_LOSS_PCT   = -1.5   # stop-loss level (%)

# Sector correlation guard — max same-sector positions in scalp watchlist
MAX_SAME_SECTOR = 2

# ── Sector mapping ────────────────────────────
SECTORS: dict[str, str] = {
    # Banking
    "BBCA.JK": "BANKING", "BBRI.JK": "BANKING", "BMRI.JK": "BANKING",
    "BNGA.JK": "BANKING", "BBNI.JK": "BANKING",
    # Telco
    "TLKM.JK": "TELCO",   "EXCL.JK": "TELCO",   "ISAT.JK": "TELCO",
    "TBIG.JK": "TELCO",   "TOWR.JK": "TELCO",
    # Consumer
    "UNVR.JK": "CONSUMER","ICBP.JK": "CONSUMER", "INDF.JK": "CONSUMER",
    "AMRT.JK": "CONSUMER","HMSP.JK": "CONSUMER", "SIDO.JK": "CONSUMER",
    "KLBF.JK": "CONSUMER","CPIN.JK": "CONSUMER",
    # Energy / Mining
    "ADRO.JK": "ENERGY",  "PTBA.JK": "ENERGY",   "PGAS.JK": "ENERGY",
    "MEDC.JK": "ENERGY",  "MDKA.JK": "ENERGY",   "INCO.JK": "ENERGY",
    "ANTM.JK": "ENERGY",
    # Industrial / Automotive
    "ASII.JK": "INDUSTRIAL","SMGR.JK": "INDUSTRIAL","INKP.JK": "INDUSTRIAL",
    "TPIA.JK": "INDUSTRIAL",
    # Tech / Media
    "GOTO.JK": "TECH",    "EMTK.JK": "TECH",     "MNCN.JK": "TECH",
    # Healthcare
    "MIKA.JK": "HEALTHCARE",
    # Energy (renewables/new)
    "BREN.JK": "ENERGY",
}

# ── Timeframes ────────────────────────────────
CANDLE_INTERVAL = "5m"    # 5-minute candles during market
CANDLE_LIMIT    = 60      # 60 candles = 5 hours history
DAILY_CANDLES   = 365     # 1yr for Prophet training

# ── Market hours (WIB = UTC+7) ────────────────
MARKET_TZ      = "Asia/Jakarta"
SESSION1_START = (9, 0)    # 09:00
SESSION1_END   = (12, 0)   # 12:00
SESSION2_START = (13, 30)   # 13:30
SESSION2_END   = (15, 49)  # 15:49
BRIEFING1_TIME = (8, 45)   # Pre-session 1 briefing
BRIEFING2_TIME = (13, 15)  # Pre-session 2 briefing
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

# Comma-separated user/group IDs allowed to run commands.
# Supports personal IDs and group chat IDs (negative numbers).
_raw_allowed = os.getenv("TELEGRAM_ALLOWED_USER_IDS", "310977969,-5144122635")
TELEGRAM_ALLOWED_USER_IDS: set[int] = {
    int(x.strip()) for x in _raw_allowed.split(",") if x.strip()
}
