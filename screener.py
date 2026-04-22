"""
Screeners for pre-session briefings.

Session 1 (08:45 WIB): D-1 screener — yesterday's daily OHLCV data.
Session 2 (12:30 WIB): S1 screener  — today's Session 1 intraday (5m) data,
                        volume baseline = avg Session 1 volume over last 5 trading days.

Both use the same 4 criteria: RSI(14), Volume ratio, MACD(12,26,9), BB(20,2).
Top 2 qualified candidates are written to Firestore for the scalp scanner.
"""

import logging
from datetime import datetime

import pandas as pd
import ta

import config
import firestore_client as db
from fetcher import fetch_daily_candles, fetch_candles
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

    # RSI must be < 30 AND rising above 7d avg to confirm oversold + uptrend
    if rsi_today < 30 and rsi_today > rsi_7d_avg:
        rsi_score, rsi_pass = 30, True
    elif rsi_today < 30:
        rsi_score, rsi_pass = 15, False  # oversold but not yet turning up — score but don't pass
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
    elif vol_ratio > 1.0:
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

    if ml_d2 < sl_d2 and ml_d1 >= sl_d1:
        # Bullish cross just happened
        macd_status, macd_score, macd_pass = "CROSS",       25, True
    elif hist_d1 > hist_d2 and hist_d1 < 0 and gap_d1 < gap_d2:
        # Histogram rising toward zero, gap narrowing — approaching cross
        macd_status, macd_score, macd_pass = "APPROACHING", 18, True
    elif ml_d1 > sl_d1 and hist_d1 > 0:
        # Already above signal — scores but cross already passed, not a setup
        macd_status, macd_score, macd_pass = "BULLISH",     10, False
    else:
        macd_status, macd_score, macd_pass = "BEARISH",      0, False

    macd_gap = round(gap_d1, 4)

    # ── Param 4 — Bollinger Bands(20, 2) — optional bonus ─────────────────
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
        bb_score, bb_pass = 12, True
    elif price_d1 <= bb_mid:
        bb_score, bb_pass = 5,  True
    else:
        bb_score, bb_pass = 0,  False   # above midline — no bonus, but not a blocker

    # ── Param 5 — BB-based target ──────────────────────────────────────────
    if price_d1 <= bb_mid * 0.995:
        target_price = round(bb_mid)
        target_pct   = round((bb_mid - price_d1) / price_d1 * 100, 2)
        target_label = "BB Middle Band"
        if price_d1 <= bb_lower:
            note = "Strong oversold — potential full BB mid recovery"
        else:
            note = "Below midline — targeting mean reversion to BB mid"
    else:
        target_price = round(bb_upper)
        target_pct   = round((bb_upper - price_d1) / price_d1 * 100, 2)
        target_label = "BB Upper Band"
        note         = "At midline — targeting upper band breakout"

    stop_loss_price = round(price_d1 * 0.985, 2)
    stop_loss_pct   = -1.5

    # ── Qualification — 3 mandatory criteria; BB is optional bonus ─────────
    total_score      = rsi_score + vol_score + macd_score + bb_score
    qualified        = rsi_pass and vol_pass and macd_pass
    disqualify_reason = (
        "RSI not confirmed uptrend" if not rsi_pass and rsi_today < 30 else
        "RSI not oversold"          if not rsi_pass else
        "Volume below avg"          if not vol_pass else
        "MACD not approaching cross" if not macd_pass else
        None
    )
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


# ── Session 1 intraday screener ───────────────────────────────────────────────

def screen_stock_s1(symbol: str) -> dict | None:
    """
    Screen using today's Session 1 intraday data (5m candles, 09:00–12:00 WIB).
    Volume baseline: avg S1 volume over last 5 trading days.
    Same 4 criteria and scoring as screen_stock_d1.
    Returns screening dict with is_s1=True, or None if insufficient data.
    """
    df = fetch_candles(symbol, interval="5m", period="5d")
    if df is None or df.empty:
        logger.warning(f"screen_stock_s1({symbol}): no candle data")
        return None

    # Normalise to WIB timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(WIB)
    else:
        df.index = df.index.tz_convert(WIB)

    today = datetime.now(WIB).date()

    def _s1(idx):
        return (idx.hour >= 9) & ((idx.hour < 12) | ((idx.hour == 12) & (idx.minute == 0)))

    today_s1 = df[(df.index.date == today)  & _s1(df.index)]
    prev_s1  = df[(df.index.date != today)  & _s1(df.index)]

    if len(today_s1) < 5:
        logger.warning(f"screen_stock_s1({symbol}): only {len(today_s1)} S1 bars today — briefing too early?")
        return None

    price_now  = round(float(today_s1["close"].iloc[-1]), 2)
    open_price = round(float(today_s1["open"].iloc[0]),  2)
    drop_from_open_pct = round((price_now - open_price) / open_price * 100, 2) if open_price else 0.0

    # Combined S1 series (older days first) for indicator history
    all_s1 = pd.concat([prev_s1, today_s1]).sort_index()
    close  = all_s1["close"]

    if len(close) < 27:
        logger.warning(f"screen_stock_s1({symbol}): only {len(close)} combined S1 bars — need ≥27")
        return None

    # ── Param 1 — RSI(14) ─────────────────────────────────────────────────
    rsi_series = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    rsi_today  = round(float(rsi_series.iloc[-1]), 2)

    # rsi_7d_avg: avg end-of-S1 RSI over the last 5 trading days
    prev_dates = sorted(set(prev_s1.index.date))[-5:]
    rsi_end_vals = []
    for d in prev_dates:
        day_close = prev_s1[prev_s1.index.date == d]["close"]
        if len(day_close) < 5:
            continue
        last_ts   = day_close.index[-1]
        pos       = close.index.get_indexer([last_ts], method="nearest")[0]
        if pos >= 14:
            v = float(rsi_series.iloc[pos])
            if not pd.isna(v):
                rsi_end_vals.append(v)

    rsi_7d_avg = round(float(pd.Series(rsi_end_vals).mean()), 2) if rsi_end_vals else rsi_today

    # RSI must be < 30 AND rising above avg to confirm oversold + uptrend
    if rsi_today < 30 and rsi_today > rsi_7d_avg:
        rsi_score, rsi_pass = 30, True
    elif rsi_today < 30:
        rsi_score, rsi_pass = 15, False  # oversold but not yet turning up
    else:
        rsi_score, rsi_pass = 0, False

    # ── Param 2 — S1 volume vs avg last-5-day S1 volume ──────────────────
    vol_s1_today = float(today_s1["volume"].sum())
    prev_vol_by_day = []
    for d in prev_dates:
        day_vol = float(prev_s1[prev_s1.index.date == d]["volume"].sum())
        if day_vol > 0:
            prev_vol_by_day.append(day_vol)

    vol_s1_avg = float(pd.Series(prev_vol_by_day).mean()) if prev_vol_by_day else vol_s1_today
    vol_ratio  = round(vol_s1_today / vol_s1_avg, 2) if vol_s1_avg > 0 else 1.0

    if vol_ratio >= 2.0:
        vol_score, vol_pass = 25, True
    elif vol_ratio >= 1.5:
        vol_score, vol_pass = 18, True
    elif vol_ratio > 1.0:
        vol_score, vol_pass = 10, True
    else:
        vol_score, vol_pass = 0, False

    # ── Param 3 — MACD(12, 26, 9) ─────────────────────────────────────────
    macd_obj    = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    macd_line   = macd_obj.macd()
    signal_line = macd_obj.macd_signal()
    histogram   = macd_obj.macd_diff()

    ml_d1,   ml_d2   = float(macd_line.iloc[-1]),   float(macd_line.iloc[-2])
    sl_d1,   sl_d2   = float(signal_line.iloc[-1]), float(signal_line.iloc[-2])
    hist_d1, hist_d2 = float(histogram.iloc[-1]),   float(histogram.iloc[-2])

    gap_d1 = sl_d1 - ml_d1
    gap_d2 = sl_d2 - ml_d2

    if ml_d2 < sl_d2 and ml_d1 >= sl_d1:
        macd_status, macd_score, macd_pass = "CROSS",       25, True
    elif hist_d1 > hist_d2 and hist_d1 < 0 and gap_d1 < gap_d2:
        macd_status, macd_score, macd_pass = "APPROACHING", 18, True
    elif ml_d1 > sl_d1 and hist_d1 > 0:
        macd_status, macd_score, macd_pass = "BULLISH",     10, False  # cross already happened
    else:
        macd_status, macd_score, macd_pass = "BEARISH",      0, False

    macd_gap = round(gap_d1, 4)

    # ── Param 4 — Bollinger Bands(20, 2) — optional bonus ─────────────────
    bb_obj   = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    bb_upper = round(float(bb_obj.bollinger_hband().iloc[-1]), 2)
    bb_mid   = round(float(bb_obj.bollinger_mavg().iloc[-1]),  2)
    bb_lower = round(float(bb_obj.bollinger_lband().iloc[-1]), 2)

    bb_range        = bb_upper - bb_lower
    bb_position_pct = round((price_now - bb_lower) / bb_range * 100, 1) if bb_range > 0 else 50.0

    if price_now <= bb_lower:
        bb_score, bb_pass = 20, True
    elif price_now <= bb_mid * 0.995:
        bb_score, bb_pass = 12, True
    elif price_now <= bb_mid:
        bb_score, bb_pass = 5,  True
    else:
        bb_score, bb_pass = 0,  False   # above midline — no bonus, not a blocker

    # ── Target / SL ───────────────────────────────────────────────────────
    if price_now <= bb_mid * 0.995:
        target_price = round(bb_mid)
        target_pct   = round((bb_mid   - price_now) / price_now * 100, 2)
        target_label = "BB Middle Band"
        note = "Strong oversold — potential BB mid recovery" if price_now <= bb_lower \
               else "Below midline — targeting mean reversion to BB mid"
    else:
        target_price = round(bb_upper)
        target_pct   = round((bb_upper - price_now) / price_now * 100, 2)
        target_label = "BB Upper Band"
        note         = "At midline — targeting upper band breakout"

    stop_loss_price = round(price_now * 0.985, 2)
    stop_loss_pct   = -1.5

    # ── Qualification — 3 mandatory criteria; BB is optional bonus ─────────
    total_score      = rsi_score + vol_score + macd_score + bb_score
    qualified        = rsi_pass and vol_pass and macd_pass
    disqualify_reason = (
        "RSI not confirmed uptrend" if not rsi_pass and rsi_today < 30 else
        "RSI not oversold"          if not rsi_pass else
        "Volume below avg"          if not vol_pass else
        "MACD not approaching cross" if not macd_pass else
        None
    )

    try:
        date_s1 = str(today_s1.index[-1].date())
    except Exception:
        date_s1 = "unknown"

    return {
        "symbol":   symbol,
        "close_d1": price_now,   # same key as D-1 for template compatibility
        "date_d1":  date_s1,
        "is_s1":    True,
        "open_price":           open_price,
        "drop_from_open_pct":   drop_from_open_pct,

        "total_score": total_score,
        "qualified":   qualified,

        "rsi_today":  rsi_today,
        "rsi_7d_avg": rsi_7d_avg,
        "rsi_score":  rsi_score,
        "rsi_pass":   rsi_pass,

        "vol_today":  vol_s1_today,
        "vol_7d_avg": round(vol_s1_avg, 0),
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

        "disqualified":      disqualify_reason is not None,
        "disqualify_reason": disqualify_reason,
    }


def find_top2_s1_candidates(watchlist: list[str]) -> list[dict]:
    """
    Screen all watchlist stocks using Session 1 intraday data.
    Returns top 2 qualified candidates sorted by total_score descending.
    """
    results = []
    for symbol in watchlist:
        try:
            result = screen_stock_s1(symbol)
            if result and result["qualified"] and not result["disqualified"]:
                results.append(result)
                logger.info(
                    f"S1 screen pass: {symbol} score={result['total_score']} "
                    f"RSI={result['rsi_today']} vol={result['vol_ratio']}x "
                    f"MACD={result['macd_status']} BB={result['bb_position_pct']}%"
                )
            else:
                reason = (result.get("disqualify_reason") or "score params not met") if result else "no data"
                logger.debug(f"S1 screen skip: {symbol} — {reason}")
        except Exception as e:
            logger.warning(f"find_top2_s1_candidates({symbol}): {e}")

    results.sort(key=lambda x: x["total_score"], reverse=True)
    top2 = results[:2]
    logger.info(f"S1 screen: {len(watchlist)} stocks, {len(results)} qualified, top {len(top2)} selected")
    return top2


def save_s1_watchlist(candidates: list[dict]) -> bool:
    """Persist today's S1-screened candidates to Firestore."""
    today_str = datetime.now(WIB).strftime("%Y-%m-%d")
    try:
        fdb = db._get_db()
        fdb.collection("idx_bot_config").document("s1_watchlist").set({
            "date":        today_str,
            "stocks":      [c["symbol"] for c in candidates],
            "screened_at": datetime.now(WIB).isoformat(),
            "details":     candidates,
        })
        logger.info(f"Saved S1 watchlist: {[c['symbol'] for c in candidates]}")
        return True
    except Exception as e:
        logger.error(f"save_s1_watchlist: {e}")
        return False
