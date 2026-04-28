"""
Market breadth analysis.
Provides context signals used in the briefing header and breadth scorer layer.

Outputs:
  - Advance/Decline ratio (from scan results — no extra API call)
  - Fear/Greed proxy (0–100) based on IHSG RSI + price momentum + volume
  - Sector rotation summary (hot vs cold sectors)
  - Full breadth summary dict consumed by ai_briefing.py
"""

import logging

import numpy as np
import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)

IHSG_TICKER = "^JKSE"

# Fear/Greed thresholds
_FG_LABELS = [
    (80, "GREED"),
    (60, "OPTIMISTIC"),
    (40, "NEUTRAL"),
    (20, "CAUTIOUS"),
    (0,  "FEAR"),
]


# ── Fear / Greed ──────────────────────────────────────────────────────────────

def _fetch_ihsg(period: str) -> "pd.DataFrame":
    """
    Download IHSG daily OHLCV. Tries ^JKSE; falls back to ^JKSE with shorter
    period on JSONDecodeError (known yfinance intermittent failure on index tickers).
    Returns empty DataFrame on total failure — callers must check .empty.
    """
    fallback_periods = {"3mo": "1mo", "1mo": "5d"}
    for attempt_period in [period, fallback_periods.get(period, "5d")]:
        try:
            df = yf.download(
                IHSG_TICKER, period=attempt_period, interval="1d",
                auto_adjust=True, progress=False,
            )
            if not df.empty:
                return df
        except Exception:
            pass
    return pd.DataFrame()


def _compute_ihsg_rsi(period: int = 14) -> float | None:
    """Compute IHSG RSI using the last (period+10) daily candles."""
    try:
        df = _fetch_ihsg("3mo")
        if df.empty or len(df) < period + 2:
            return None
        close  = df["Close"].dropna()
        delta  = close.diff()
        gain   = delta.clip(lower=0)
        loss   = (-delta).clip(lower=0)
        ag     = gain.ewm(alpha=1 / period, adjust=False).mean()
        al     = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs     = ag / al.replace(0, np.nan)
        rsi    = 100 - 100 / (1 + rs)
        return float(rsi.iloc[-1])
    except Exception as e:
        logger.debug("IHSG RSI compute failed: %s", e)
        return None


def _compute_ihsg_momentum(lookback: int = 5) -> float | None:
    """% change of IHSG over last `lookback` trading days."""
    try:
        df = _fetch_ihsg("1mo")
        if df.empty or len(df) < lookback + 1:
            return None
        close = df["Close"].dropna()
        return float((close.iloc[-1] - close.iloc[-(lookback + 1)]) / close.iloc[-(lookback + 1)] * 100)
    except Exception as e:
        logger.debug("IHSG momentum compute failed: %s", e)
        return None


def compute_fear_greed() -> dict:
    """
    Compute a Fear/Greed composite score (0–100) for the IDX market.

    Components (equal weight):
      RSI component   : RSI mapped linearly to 0–100
      Momentum        : 5-day IHSG change, capped at ±5%, mapped to 0–100
      Breadth         : placeholder 50 if A/D not yet available (overridden later)

    Returns { score: int, label: str, components: dict }
    """
    rsi   = _compute_ihsg_rsi()
    mom   = _compute_ihsg_momentum()

    rsi_score = float(rsi) if rsi is not None else 50.0
    # Mom: cap ±5%, map [-5, +5] → [0, 100]
    mom_score = 50.0
    if mom is not None:
        mom_capped = max(-5.0, min(5.0, mom))
        mom_score  = (mom_capped + 5.0) / 10.0 * 100.0

    score = round((rsi_score + mom_score) / 2)
    score = max(0, min(100, score))

    label = "NEUTRAL"
    for threshold, lbl in _FG_LABELS:
        if score >= threshold:
            label = lbl
            break

    return {
        "score":      score,
        "label":      label,
        "components": {
            "ihsg_rsi":       round(rsi, 1) if rsi else None,
            "ihsg_5d_change": round(mom, 2) if mom else None,
        },
    }


# ── Advance / Decline ─────────────────────────────────────────────────────────

def compute_advance_decline(scan_results: list[dict]) -> dict:
    """
    Compute A/D ratio from yesterday's price action across scanned stocks.
    Uses the last two closes in each stock's snapshot.
    scan_results: list of score_stock() output dicts (must include 'snapshot').
    """
    advance = decline = unchanged = 0

    for r in scan_results:
        snap = r.get("snapshot", {})
        close = snap.get("close")
        ema20 = snap.get("ema20")          # proxy for yesterday's level
        if close is None or ema20 is None:
            continue
        # Use breakout_bull / breakout_bear flags as a simpler A/D proxy
        if r["snapshot"].get("breakout_bull"):
            advance += 1
        elif r["snapshot"].get("breakout_bear"):
            decline += 1
        else:
            # Fall back to price vs EMA20 as directional indicator
            if close > ema20:
                advance += 1
            elif close < ema20:
                decline += 1
            else:
                unchanged += 1

    total = advance + decline + unchanged
    ratio = round(advance / decline, 2) if decline else float("inf")

    return {
        "advance":   advance,
        "decline":   decline,
        "unchanged": unchanged,
        "total":     total,
        "ratio":     ratio,
        "breadth":   "POSITIVE" if ratio > 1.5 else ("NEGATIVE" if ratio < 0.7 else "NEUTRAL"),
    }


# ── Sector rotation ───────────────────────────────────────────────────────────

def get_sector_rotation(breadth_data: dict) -> dict:
    """
    From the fetcher.fetch_market_breadth() dict, classify sectors as hot/cold.
    Returns { hot: list[str], cold: list[str], neutral: list[str] }
    """
    hot = cold = neutral = []
    hot, cold, neutral = [], [], []

    for sector, data in breadth_data.items():
        if sector == "IHSG":
            continue
        chg = data.get("change_pct", 0) or 0
        if chg >= 0.5:
            hot.append(f"{sector} +{chg:.1f}%")
        elif chg <= -0.5:
            cold.append(f"{sector} {chg:.1f}%")
        else:
            neutral.append(sector)

    return {"hot": hot, "cold": cold, "neutral": neutral}


# ── Full summary ──────────────────────────────────────────────────────────────

def build_breadth_summary(
    breadth_data:   dict,
    scan_results:   list[dict],
    foreign_market: dict | None,
) -> dict:
    """
    Assemble the full breadth context dict for ai_briefing.py.
    Returns a flat dict ready to be JSON-serialised into the Claude prompt.
    """
    ihsg = breadth_data.get("IHSG", {})
    fg   = compute_fear_greed()
    ad   = compute_advance_decline(scan_results)
    rot  = get_sector_rotation(breadth_data)

    # Patch fear/greed score with A/D ratio (overrides placeholder)
    ad_score = min(100, max(0, int(ad["ratio"] / 3.0 * 100))) if ad["decline"] > 0 else 70
    fg_composite = round((fg["score"] * 2 + ad_score) / 3)
    fg_composite = max(0, min(100, fg_composite))
    fg_label = "NEUTRAL"
    for threshold, lbl in _FG_LABELS:
        if fg_composite >= threshold:
            fg_label = lbl
            break

    fm_dir = foreign_market.get("direction", "NEUTRAL") if foreign_market else "N/A"
    fm_net = foreign_market.get("net_val_idr", 0) or 0

    return {
        "ihsg_close":       ihsg.get("close"),
        "ihsg_change_pct":  ihsg.get("change_pct"),
        "ihsg_direction":   ihsg.get("direction", "FLAT"),
        "fear_greed_score": fg_composite,
        "fear_greed_label": fg_label,
        "advance":          ad["advance"],
        "decline":          ad["decline"],
        "ad_ratio":         ad["ratio"],
        "ad_breadth":       ad["breadth"],
        "foreign_market_direction": fm_dir,
        "foreign_market_net_idr":   round(fm_net / 1_000_000_000, 1),   # → billions IDR
        "sectors_hot":    rot["hot"],
        "sectors_cold":   rot["cold"],
        "ihsg_rsi":       fg["components"].get("ihsg_rsi"),
        "ihsg_5d_change": fg["components"].get("ihsg_5d_change"),
    }
