import os
from dotenv import load_dotenv

load_dotenv()

# ── Timezone ──────────────────────────────────────────────────────────────────
MARKET_TZ = "Asia/Jakarta"

# ── Market hours (WIB) ────────────────────────────────────────────────────────
SESSION1_START = (9, 0)
SESSION1_END   = (12, 0)
SESSION2_START = (13, 30)
SESSION2_END   = (15, 49)

# ── Scheduler ─────────────────────────────────────────────────────────────────
EOD_SCAN_TIME     = (16, 30)   # nightly D-1 full universe scan (after market close)
BRIEFING_TIME     = (8, 30)    # morning briefing delivery
MIDDAY_TIME       = (13, 15)   # pre-Sesi 2 briefing (15 min before 13:30 open)

# ── RSI thresholds ────────────────────────────────────────────────────────────
RSI_OVERSOLD        = 30
RSI_OVERBOUGHT      = 70
RSI_SWEET_SPOT_LOW  = 45
RSI_SWEET_SPOT_HIGH = 65

# ── Indicator periods ─────────────────────────────────────────────────────────
EMA_FAST        = 20
EMA_MID         = 50
EMA_SLOW        = 200
RSI_PERIOD      = 14
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
BB_PERIOD       = 20
BB_STD          = 2.0
ATR_PERIOD      = 14
STOCH_K         = 14
STOCH_D         = 3
VOL_MA_PERIOD   = 20
BREAKOUT_PERIOD = 20          # N-day high for breakout detection

# ── Divergence ────────────────────────────────────────────────────────────────
DIV_LOOKBACK          = 60    # candles to look back for divergence
DIV_PIVOT_WINDOW      = 5     # bars left/right to qualify a pivot
DIV_BULLISH_BONUS     = 5     # bonus points added to momentum score
DIV_HIDDEN_BULL_BONUS = 3
DIV_BEARISH_PENALTY   = -10   # subtracted from total score (auto-disqualifier)

# ── Scoring weights (must sum to 100) ─────────────────────────────────────────
WEIGHT_TREND    = 25
WEIGHT_MOMENTUM = 20
WEIGHT_VOLUME   = 20
WEIGHT_PATTERN  = 15
WEIGHT_FOREIGN  = 15
WEIGHT_BREADTH  = 5

# ── Liquidity filter (Stage 1 screen) ─────────────────────────────────────────
MIN_MARKET_CAP_IDR   = 500_000_000_000   # 500B IDR
MIN_AVG_DAILY_VAL    = 5_000_000_000     # 5B IDR/day avg value traded
MIN_PRICE            = 200               # IDR
MAX_PRICE            = 50_000            # IDR

# ── Setup filter (Stage 2 screen) ─────────────────────────────────────────────
MIN_SCORE            = int(os.getenv("MIN_SCORE", "60"))
MIN_VOL_RATIO        = 1.2               # volume must be >= 1.2x 20-day avg
MIN_RR               = 1.5              # minimum risk:reward ratio

# ── AI briefing ───────────────────────────────────────────────────────────────
TOP_N_AI        = int(os.getenv("TOP_N_AI", "15"))
CLAUDE_MODEL    = "claude-haiku-4-5-20251001"

# ── IDX fee structure ─────────────────────────────────────────────────────────
BUY_FEE_PCT    = 0.0015   # 0.15%
SELL_FEE_PCT   = 0.0025   # 0.25% (incl. 0.10% PPh)
FEE_RT         = BUY_FEE_PCT + SELL_FEE_PCT   # 0.40% round trip
MIN_TARGET_PCT = 1.5      # minimum viable profit after fees

# ── Capital & risk ────────────────────────────────────────────────────────────
CAPITAL_IDR          = int(os.getenv("CAPITAL_IDR", "10000000"))
MAX_POSITIONS        = int(os.getenv("MAX_POSITIONS", "5"))
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "3.0"))
MAX_RISK_PER_TRADE   = 0.02   # 2% of capital max risk per trade

# ── Universe ──────────────────────────────────────────────────────────────────
UNIVERSE = os.getenv("UNIVERSE", "IDX80")   # LQ45 | IDX80 | COMPOSITE

# ── Credentials ───────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
INVEZGO_API_KEY    = os.getenv("INVEZGO_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_IDS  = [
    cid.strip()
    for cid in os.getenv("TELEGRAM_CHAT_IDS", "").split(",")
    if cid.strip()
]
FIREBASE_CRED_JSON = os.getenv("FIREBASE_CRED_JSON", "")
