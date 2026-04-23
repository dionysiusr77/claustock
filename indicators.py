from __future__ import annotations
import pandas as pd
import pytz
import ta
import logging

logger = logging.getLogger(__name__)

_WIB = pytz.timezone("Asia/Jakarta")


def _calc_vwap(df: pd.DataFrame) -> float | None:
    """
    Compute today's session VWAP from 5-minute bars.
    Resets at market open each day — only today's bars are used.
    Returns None if today has fewer than 3 bars (e.g. called pre-market).
    """
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC").tz_convert(_WIB)
    else:
        idx = idx.tz_convert(_WIB)

    today      = idx[-1].date()
    today_mask = idx.date == today
    if today_mask.sum() < 3:
        return None

    t = df[today_mask]
    typical = (t["high"] + t["low"] + t["close"]) / 3
    total_vol = t["volume"].sum()
    if total_vol == 0:
        return None
    return round(float((typical * t["volume"]).sum() / total_vol), 2)

# ── Candle pattern helpers ────────────────────────────────────────────────────

def _is_bullish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    return (
        prev["close"] < prev["open"]        # prev red candle
        and curr["close"] > curr["open"]    # curr green candle
        and curr["open"] < prev["close"]
        and curr["close"] > prev["open"]
    )


def _is_hammer(df: pd.DataFrame) -> bool:
    candle = df.iloc[-1]
    body   = abs(candle["close"] - candle["open"])
    wick   = candle["high"] - max(candle["close"], candle["open"])
    tail   = min(candle["close"], candle["open"]) - candle["low"]
    if body == 0:
        return False
    return tail >= 2 * body and wick <= body * 0.5


def _is_shooting_star(df: pd.DataFrame) -> bool:
    candle = df.iloc[-1]
    body   = abs(candle["close"] - candle["open"])
    wick   = candle["high"] - max(candle["close"], candle["open"])
    tail   = min(candle["close"], candle["open"]) - candle["low"]
    if body == 0:
        return False
    return wick >= 2 * body and tail <= body * 0.5


def detect_bullish_divergence(
    close: "pd.Series",
    rsi_series: "pd.Series",
    window: int = 40,
) -> dict:
    """
    Detect bullish price/RSI divergence over the last `window` bars.

    Bullish divergence: price making lower lows while RSI makes higher lows —
    momentum recovering before price, classic bounce warning.

    Finds swing lows (local minima) and counts how many consecutive pairs
    show the divergence pattern.

    Returns:
        divergence:  bool   — at least one pair confirmed
        swings:      int    — number of swing lows found
        div_pairs:   int    — how many pairs show divergence
        bonus:       int    — 0 / 12 / 20
        label:       str    — human-readable summary
    """
    import numpy as np

    if len(close) < 6:
        return {"divergence": False, "swings": 0, "div_pairs": 0, "bonus": 0, "label": "insufficient data"}

    c = close.iloc[-window:].values
    r = rsi_series.iloc[-window:].values

    # Find swing lows: strict local minima (both neighbours higher)
    swings = [
        i for i in range(1, len(c) - 1)
        if c[i] < c[i - 1] and c[i] < c[i + 1]
    ]

    if len(swings) < 2:
        return {"divergence": False, "swings": len(swings), "div_pairs": 0, "bonus": 0, "label": "not detected"}

    # Count consecutive divergent pairs (price lower low, RSI higher low)
    div_pairs = 0
    for j in range(len(swings) - 1):
        i1, i2 = swings[j], swings[j + 1]
        if c[i2] < c[i1] and r[i2] > r[i1]:
            div_pairs += 1

    if div_pairs >= 2:
        bonus, label = 20, f"confirmed ({len(swings)} swings)"
    elif div_pairs == 1:
        bonus, label = 12, "confirmed (2 swings)"
    else:
        bonus, label = 0, "not detected"

    return {
        "divergence": div_pairs >= 1,
        "swings":     len(swings),
        "div_pairs":  div_pairs,
        "bonus":      bonus,
        "label":      label,
    }


def _detect_candle_pattern(df: pd.DataFrame) -> str:
    """Returns the dominant candle pattern label, or 'neutral'."""
    if _is_bullish_engulfing(df):
        return "bullish_engulfing"
    if _is_hammer(df):
        return "hammer"
    if _is_shooting_star(df):
        return "shooting_star"
    return "neutral"


# ── Core indicator calculation ────────────────────────────────────────────────

def calculate_indicators(df: pd.DataFrame) -> dict | None:
    """
    Calculate RSI(14), MA(7), MA(30), volume ratio, and candle pattern
    from a OHLCV DataFrame (5-min candles).

    Returns:
    {
        rsi:            float,
        ma7:            float,
        ma30:           float,
        ma_trend:       "UP" | "DOWN" | "FLAT",   # ma7 vs ma30
        price:          float,                    # last close
        volume_ratio:   float,                    # last vol / 20-bar avg vol
        candle_pattern: str,
        score:          int,                      # 0–35 pts
        reasons:        list[str],
    }
    """
    if df is None or len(df) < 30:
        logger.warning("Not enough candles for indicator calculation (need ≥30)")
        return None

    try:
        close  = df["close"]
        volume = df["volume"]

        # RSI(14) — current and previous bar
        rsi_series = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        rsi      = round(float(rsi_series.iloc[-1]), 2)
        rsi_prev = round(float(rsi_series.iloc[-2]), 2) if len(rsi_series) >= 2 else rsi

        # MACD(12, 26, 9)
        macd_obj       = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        macd_hist      = macd_obj.macd_diff()   # histogram = MACD line − signal line
        macd_histogram      = round(float(macd_hist.iloc[-1]), 4)
        macd_histogram_prev = round(float(macd_hist.iloc[-2]), 4) if len(macd_hist) >= 2 else macd_histogram

        # MA(7) and MA(30)
        ma7  = round(float(close.rolling(7).mean().iloc[-1]), 2)
        ma30 = round(float(close.rolling(30).mean().iloc[-1]), 2)

        if ma7 > ma30 * 1.001:
            ma_trend = "UP"
        elif ma7 < ma30 * 0.999:
            ma_trend = "DOWN"
        else:
            ma_trend = "FLAT"

        # Volume: use completed candles (iloc[-2] = last closed bar, iloc[-3] = bar before that)
        # iloc[-1] is the live in-progress bar — often reports 0 from yfinance
        vol_completed = float(volume.iloc[-2]) if len(volume) >= 2 else float(volume.iloc[-1])
        vol_prev      = float(volume.iloc[-3]) if len(volume) >= 3 else vol_completed
        avg_vol       = float(volume.iloc[:-1].rolling(20).mean().iloc[-1]) if len(volume) >= 20 else vol_completed
        volume_ratio  = round(vol_completed / avg_vol, 2) if avg_vol > 0 else 1.0
        # True if today's completed bar beat both yesterday AND 20-bar average
        volume_surging = vol_completed > vol_prev and vol_completed > avg_vol

        # Bullish divergence (price/RSI)
        divergence = detect_bullish_divergence(close, rsi_series)

        # Candle pattern — use completed candles only (drop live bar)
        candle_pattern = _detect_candle_pattern(df.iloc[:-1])

        price = round(float(close.iloc[-1]), 2)

        # VWAP — today's session only (needs price to be defined first)
        vwap     = _calc_vwap(df)
        vwap_pct = round((price - vwap) / vwap * 100, 2) if vwap else None
        price_vs_vwap = (
            "BELOW" if vwap and price < vwap * 0.999 else
            "ABOVE" if vwap and price > vwap * 1.001 else
            "AT"
        ) if vwap else "UNKNOWN"

        # Derived momentum signals
        rsi_rising    = rsi > rsi_prev                          # RSI turning up
        macd_turning  = macd_histogram > macd_histogram_prev    # histogram expanding upward
        macd_crossing = macd_histogram >= 0 > macd_histogram_prev  # just crossed zero

        # ── Scoring (0–35 pts) ────────────────────────────────────────────
        score   = 0
        reasons = []

        # RSI scoring (0–12 pts)
        if rsi < 30:
            score += 12
            reasons.append(f"RSI {rsi:.1f} — oversold (strong buy)")
        elif rsi < 40:
            score += 10
            reasons.append(f"RSI {rsi:.1f} — near oversold")
        elif rsi < 50:
            score += 6
            reasons.append(f"RSI {rsi:.1f} — below midline, mild bullish")
        elif rsi < 60:
            score += 3
            reasons.append(f"RSI {rsi:.1f} — neutral")
        elif rsi < 70:
            score += 1
            reasons.append(f"RSI {rsi:.1f} — approaching overbought")
        else:
            score += 0
            reasons.append(f"RSI {rsi:.1f} — overbought, avoid")

        # RSI direction bonus (0–5 pts)
        if rsi_rising and rsi < 50:
            score += 5
            reasons.append(f"RSI rising ({rsi_prev:.1f} → {rsi:.1f}) — momentum recovering")

        # MACD scoring (0–8 pts)
        if macd_crossing:
            score += 8
            reasons.append(f"MACD crossed zero — bullish signal")
        elif macd_turning and macd_histogram < 0:
            score += 5
            reasons.append(f"MACD histogram turning up ({macd_histogram_prev:.4f} → {macd_histogram:.4f})")
        elif macd_turning and macd_histogram >= 0:
            score += 3
            reasons.append(f"MACD histogram expanding positive")

        # MA trend scoring (0–5 pts)
        if ma_trend == "UP":
            score += 5
            reasons.append(f"MA uptrend (MA7 {ma7} > MA30 {ma30})")
        elif ma_trend == "FLAT":
            score += 2
            reasons.append(f"MA flat (MA7 {ma7} ≈ MA30 {ma30})")
        else:
            score += 0
            reasons.append(f"MA downtrend (MA7 {ma7} < MA30 {ma30})")

        # Volume scoring (0–5 pts)
        if volume_surging:
            score += 5
            reasons.append(f"Volume surging ({volume_ratio:.1f}x avg, > yesterday)")
        elif volume_ratio >= 2.0:
            score += 4
            reasons.append(f"Volume {volume_ratio:.1f}x avg — strong")
        elif volume_ratio >= 1.5:
            score += 2
            reasons.append(f"Volume {volume_ratio:.1f}x avg — above average")
        elif volume_ratio >= 1.0:
            score += 1
            reasons.append(f"Volume {volume_ratio:.1f}x avg — normal")
        else:
            reasons.append(f"Volume {volume_ratio:.1f}x avg — low")

        # Candle pattern scoring (0–5 pts)
        pattern_scores = {
            "bullish_engulfing": 5,
            "hammer":            4,
            "shooting_star":     0,
            "neutral":           1,
        }
        score += pattern_scores.get(candle_pattern, 1)
        if candle_pattern != "neutral":
            reasons.append(f"Candle: {candle_pattern.replace('_', ' ')}")

        # VWAP scoring (0–5 pts)
        if vwap is not None:
            if vwap_pct <= -1.0 and volume_surging:
                score += 5
                reasons.append(
                    f"Price {abs(vwap_pct):.1f}% below VWAP ({vwap:,.0f}) + volume surge — capitulation bounce setup"
                )
            elif vwap_pct <= -1.0:
                score += 3
                reasons.append(f"Price {abs(vwap_pct):.1f}% below VWAP ({vwap:,.0f}) — oversold vs session avg")
            elif price_vs_vwap == "BELOW":
                score += 1
                reasons.append(f"Price slightly below VWAP ({vwap:,.0f})")
            elif vwap_pct >= 1.0 and volume_surging:
                score += 3
                reasons.append(
                    f"Price {vwap_pct:.1f}% above VWAP ({vwap:,.0f}) + volume surge — momentum continuation"
                )

        score = min(score, 40)

        return {
            "rsi":                  rsi,
            "rsi_prev":             rsi_prev,
            "rsi_rising":           rsi_rising,
            "macd_histogram":       macd_histogram,
            "macd_histogram_prev":  macd_histogram_prev,
            "macd_turning":         macd_turning,
            "macd_crossing":        macd_crossing,
            "ma7":                  ma7,
            "ma30":                 ma30,
            "ma_trend":             ma_trend,
            "price":                price,
            "volume_ratio":         volume_ratio,
            "volume_surging":       volume_surging,
            "vwap":                 vwap,
            "vwap_pct":             vwap_pct,
            "price_vs_vwap":        price_vs_vwap,
            "candle_pattern":       candle_pattern,
            "divergence":           divergence,
            "score":                score,
            "reasons":              reasons,
        }

    except Exception as e:
        logger.error(f"calculate_indicators: {e}")
        return None
