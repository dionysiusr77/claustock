"""
D-1 daily screener — pre-session scalp candidate selection.

Runs before 08:45 and 12:30 WIB briefings. Uses yesterday's daily OHLCV data
to score each watchlist stock on 4 criteria (RSI, Volume, MACD, Bollinger Bands)
and select the top 2 candidates.

The selected candidates are written to Firestore so the intraday scalp scanner
reads only those 2 stocks instead of the full UNIVERSE.
"""

import logging
from datetime import datetime

import pandas as pd
import ta

import config
import firestore_client as db
from fetcher import fetch_daily_candles
from scheduler import WIB

logger = logging.getLogger(__name__)


def screen_stock_d1(symbol: str) -> dict | None:
    """
    Screen a single stock using D-1 (yesterday's) daily data.
    Returns a full screening dict or None if data is unavailable.

    Scoring (0–100 pts):
      RSI    0–30
      Volume 0–25
      MACD   0–25
      BB     0–20
    """
    # Need 60d of daily data for MACD(26) + BB(20) + 7-day averages
    df = fetch_daily_candles(symbol, period="60d")
    if df is None or len(df) < 27:
        logger.warning(f"screen_stock_d1({symbol}): insufficient data ({len(df) if df is not None else 0} rows)")
        return None

    close  = df["close"]
    volume = df["volume"]

    # ── Param 1 — RSI(14) ─────────────────────────────────────────────────
    rsi_series  = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    rsi_today   = round(float(rsi_series.iloc[-1]), 2)
    rsi_7d_slice = rsi_series.iloc[-8:-1].dropna()
    rsi_7d_avg  = round(float(rsi_7d_slice.mean()), 2) if len(rsi_7d_slice) >= 3 else rsi_today

    if rsi_today < 30 and rsi_today > rsi_7d_avg:
        rsi_score, rsi_pass = 30, True
    elif rsi_today < 30:
        rsi_score, rsi_pass = 15, True
    else:
        rsi_score, rsi_pass = 0, False

    # ── Param 2 — Volume vs 7-day avg ─────────────────────────────────────
    vol_today   = float(volume.iloc[-1])
    vol_7d_slice = volume.iloc[-8:-1]
    vol_7d_avg  = float(vol_7d_slice.mean()) if len(vol_7d_slice) >= 3 else vol_today
    vol_ratio   = round(vol_today / vol_7d_avg, 2) if vol_7d_avg > 0 else 1.0

    if vol_ratio >= 2.0:
        vol_score, vol_pass = 25, True
    elif vol_ratio >= 1.5:
        vol_score, vol_pass = 18, True
    elif vol_ratio >= 1.0:
        vol_score, vol_pass = 10, True
    else:
        vol_score, vol_pass = 0, False

    # ── Param 3 — MACD(12, 26, 9) ─────────────────────────────────────────
    macd_obj    = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    macd_line   = macd_obj.macd()
    signal_line = macd_obj.macd_signal()
    histogram   = macd_obj.macd_diff()

    ml_d1   = float(macd_line.iloc[-1])
    ml_d2   = float(macd_line.iloc[-2])
    sl_d1   = float(signal_line.iloc[-1])
    sl_d2   = float(signal_line.iloc[-2])
    hist_d1 = float(histogram.iloc[-1])
    hist_d2 = float(histogram.iloc[-2])

    gap_d1 = sl_d1 - ml_d1   # positive = MACD below signal (bearish)
    gap_d2 = sl_d2 - ml_d2

    disqualify_reason = None

    if ml_d2 < sl_d2 and ml_d1 >= sl_d1:
        # Case A: bullish cross just happened
        macd_status, macd_score, macd_pass = "CROSS", 25, True
    elif hist_d1 > hist_d2 and hist_d1 < 0 and gap_d1 < gap_d2:
        # Case B: histogram rising toward zero, gap narrowing
        macd_status, macd_score, macd_pass = "APPROACHING", 18, True
    elif ml_d1 > sl_d1 and hist_d1 > 0:
        # Case C: already above signal line
        macd_status, macd_score, macd_pass = "BULLISH", 10, True
    else:
        # Case D: bearish — disqualify
        macd_status, macd_score, macd_pass = "BEARISH", 0, False
        disqualify_reason = "MACD bearish"

    macd_gap = round(gap_d1, 4)

    # ── Param 4 — Bollinger Bands(20, 2) ──────────────────────────────────
    bb_obj   = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    bb_upper = round(float(bb_obj.bollinger_hband().iloc[-1]), 2)
    bb_mid   = round(float(bb_obj.bollinger_mavg().iloc[-1]), 2)
    bb_lower = round(float(bb_obj.bollinger_lband().iloc[-1]), 2)
    price_d1 = round(float(close.iloc[-1]), 2)

    bb_range        = bb_upper - bb_lower
    bb_position_pct = round((price_d1 - bb_lower) / bb_range * 100, 1) if bb_range > 0 else 50.0

    if price_d1 <= bb_lower:
        bb_score, bb_pass = 20, True
    elif price_d1 <= bb_mid * 0.995:
        # Clearly below midline
        bb_score, bb_pass = 12, True
    elif price_d1 <= bb_mid:
        # Right at midline (within 0.5%)
        bb_score, bb_pass = 5, True
    else:
        # Above midline — disqualify
        bb_score, bb_pass = 0, False
        if disqualify_reason is None:
            disqualify_reason = "Price above BB midline"

    # ── Param 5 — BB-based target ──────────────────────────────────────────
    if price_d1 <= bb_mid * 0.995:
        # Below midline — target BB mid
        target_price = round(bb_mid)
        target_pct   = round((bb_mid - price_d1) / price_d1 * 100, 2)
        target_label = "BB Middle Band"
        if price_d1 <= bb_lower:
            note = "Strong oversold — potential full BB mid recovery"
        else:
            note = "Below midline — targeting mean reversion to BB mid"
    else:
        # Right at midline — aim for upper band
        target_price = round(bb_upper)
        target_pct   = round((bb_upper - price_d1) / price_d1 * 100, 2)
        target_label = "BB Upper Band"
        note         = "At midline — targeting upper band breakout"

    stop_loss_price = round(price_d1 * 0.985, 2)
    stop_loss_pct   = -1.5

    # ── Qualification ──────────────────────────────────────────────────────
    total_score  = rsi_score + vol_score + macd_score + bb_score
    qualified    = rsi_pass and vol_pass and macd_pass and bb_pass
    disqualified = disqualify_reason is not None

    try:
        date_d1 = str(df.index[-1].date())
    except Exception:
        date_d1 = "unknown"

    return {
        "symbol":    symbol,
        "close_d1":  price_d1,
        "date_d1":   date_d1,

        "total_score": total_score,
        "qualified":   qualified,

        "rsi_today":  rsi_today,
        "rsi_7d_avg": rsi_7d_avg,
        "rsi_score":  rsi_score,
        "rsi_pass":   rsi_pass,

        "vol_today":  vol_today,
        "vol_7d_avg": round(vol_7d_avg, 0),
        "vol_ratio":  vol_ratio,
        "vol_score":  vol_score,
        "vol_pass":   vol_pass,

        "macd_status": macd_status,
        "macd_gap":    macd_gap,
        "macd_score":  macd_score,
        "macd_pass":   macd_pass,

        "bb_upper":        bb_upper,
        "bb_mid":          bb_mid,
        "bb_lower":        bb_lower,
        "bb_position_pct": bb_position_pct,
        "bb_score":        bb_score,
        "bb_pass":         bb_pass,

        "target_price":    target_price,
        "target_pct":      target_pct,
        "target_label":    target_label,
        "stop_loss_price": stop_loss_price,
        "stop_loss_pct":   stop_loss_pct,
        "note":            note,

        "disqualified":     disqualified,
        "disqualify_reason": disqualify_reason,
    }


def find_top2_scalp_candidates(watchlist: list[str]) -> list[dict]:
    """
    Screen all stocks in watchlist using D-1 data.
    Returns top 2 qualified candidates sorted by total_score descending.
    """
    results = []
    for symbol in watchlist:
        try:
            result = screen_stock_d1(symbol)
            if result and result["qualified"] and not result["disqualified"]:
                results.append(result)
                logger.info(
                    f"D-1 screen pass: {symbol} score={result['total_score']} "
                    f"RSI={result['rsi_today']} vol={result['vol_ratio']}x "
                    f"MACD={result['macd_status']} BB={result['bb_position_pct']}%"
                )
            else:
                reason = result.get("disqualify_reason") or "score params not met" if result else "no data"
                logger.debug(f"D-1 screen skip: {symbol} — {reason}")
        except Exception as e:
            logger.warning(f"find_top2_scalp_candidates({symbol}): {e}")

    results.sort(key=lambda x: x["total_score"], reverse=True)
    top2 = results[:2]

    logger.info(f"D-1 screen: {len(watchlist)} stocks, {len(results)} qualified, top {len(top2)} selected")
    return top2


def save_daily_scalp_watchlist(candidates: list[dict]) -> bool:
    """Persist today's D-1 screened candidates to Firestore."""
    today_str = datetime.now(WIB).strftime("%Y-%m-%d")
    try:
        from firebase_admin import firestore
        fdb = db._get_db()
        fdb.collection("idx_bot_config").document("daily_scalp_watchlist").set({
            "date":        today_str,
            "stocks":      [c["symbol"] for c in candidates],
            "screened_at": datetime.now(WIB).isoformat(),
            "details":     candidates,
        })
        logger.info(f"Saved daily scalp watchlist: {[c['symbol'] for c in candidates]}")
        return True
    except Exception as e:
        logger.error(f"save_daily_scalp_watchlist: {e}")
        return False


def get_daily_scalp_watchlist() -> list[str]:
    """
    Load today's D-1 screened stock list from Firestore.
    Returns list of symbols, or empty list if not set for today.
    """
    today_str = datetime.now(WIB).strftime("%Y-%m-%d")
    try:
        fdb = db._get_db()
        doc = fdb.collection("idx_bot_config").document("daily_scalp_watchlist").get()
        if doc.exists:
            data = doc.to_dict()
            if data.get("date") == today_str:
                return data.get("stocks", [])
    except Exception as e:
        logger.error(f"get_daily_scalp_watchlist: {e}")
    return []
