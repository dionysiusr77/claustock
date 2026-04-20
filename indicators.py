import pandas as pd
import ta
import logging

logger = logging.getLogger(__name__)

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

        # Candle pattern — use completed candles only (drop live bar)
        candle_pattern = _detect_candle_pattern(df.iloc[:-1])

        price = round(float(close.iloc[-1]), 2)

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

        score = min(score, 35)

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
            "candle_pattern":       candle_pattern,
            "score":                score,
            "reasons":              reasons,
        }

    except Exception as e:
        logger.error(f"calculate_indicators: {e}")
        return None
