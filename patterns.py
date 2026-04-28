"""
Candlestick pattern detection on daily OHLCV data.
All detectors operate on the last 1–2 rows of the DataFrame.

Patterns detected (bullish only — bearish used as disqualifiers):
  HAMMER            — small body, long lower wick, short upper wick
  INVERTED_HAMMER   — small body, long upper wick, short lower wick (reversal at bottom)
  BULLISH_ENGULFING — bullish candle body fully engulfs previous bearish body
  MORNING_STAR      — 3-candle reversal: bearish, doji/small, bullish
  INSIDE_BAR        — current range inside previous range (compression before breakout)
  DOJI              — open ≈ close, indecision

Bearish patterns (used as penalty flags):
  BEARISH_ENGULFING
  SHOOTING_STAR     — long upper wick at top (same shape as hammer but at highs)
  EVENING_STAR

Each function returns True/False on the last candle(s) of the supplied DataFrame.
"""

import pandas as pd


# ── Candle anatomy helpers ────────────────────────────────────────────────────

def _body(o: float, c: float) -> float:
    return abs(c - o)

def _upper_wick(o: float, h: float, c: float) -> float:
    return h - max(o, c)

def _lower_wick(o: float, l: float, c: float) -> float:
    return min(o, c) - l

def _total_range(h: float, l: float) -> float:
    return h - l


# ── Single-candle patterns ────────────────────────────────────────────────────

def is_hammer(df: pd.DataFrame) -> bool:
    """
    Bullish reversal signal when appearing after a downtrend.
    Criteria:
      - Lower wick >= 2× body
      - Upper wick <= 0.3× body
      - Body >= 10% of total range (not a doji)
    """
    if len(df) < 1:
        return False
    r = df.iloc[-1]
    o, h, l, c = r["open"], r["high"], r["low"], r["close"]
    body  = _body(o, c)
    low_w = _lower_wick(o, l, c)
    up_w  = _upper_wick(o, h, c)
    rng   = _total_range(h, l)
    if rng == 0:
        return False
    return (
        low_w >= 2.0 * body
        and up_w  <= 0.3 * max(body, 1)
        and body  >= 0.10 * rng
    )


def is_inverted_hammer(df: pd.DataFrame) -> bool:
    """
    Same shape as hammer but with long upper wick.
    Bullish only when appearing at the bottom of a downtrend.
    """
    if len(df) < 1:
        return False
    r = df.iloc[-1]
    o, h, l, c = r["open"], r["high"], r["low"], r["close"]
    body  = _body(o, c)
    up_w  = _upper_wick(o, h, c)
    low_w = _lower_wick(o, l, c)
    rng   = _total_range(h, l)
    if rng == 0:
        return False
    return (
        up_w  >= 2.0 * body
        and low_w <= 0.3 * max(body, 1)
        and body  >= 0.10 * rng
    )


def is_doji(df: pd.DataFrame, threshold: float = 0.10) -> bool:
    """Body <= threshold × total range."""
    if len(df) < 1:
        return False
    r = df.iloc[-1]
    o, h, l, c = r["open"], r["high"], r["low"], r["close"]
    rng = _total_range(h, l)
    if rng == 0:
        return True
    return _body(o, c) <= threshold * rng


def is_shooting_star(df: pd.DataFrame) -> bool:
    """
    Bearish reversal: long upper wick, small body at the bottom of the candle's range.
    Bearish counterpart of inverted hammer — penalty flag when at highs.
    """
    if len(df) < 1:
        return False
    r = df.iloc[-1]
    o, h, l, c = r["open"], r["high"], r["low"], r["close"]
    body  = _body(o, c)
    up_w  = _upper_wick(o, h, c)
    low_w = _lower_wick(o, l, c)
    rng   = _total_range(h, l)
    if rng == 0:
        return False
    return (
        up_w  >= 2.0 * body
        and low_w <= 0.3 * max(body, 1)
        and body  >= 0.10 * rng
        and c < o                    # bearish close
    )


# ── Two-candle patterns ───────────────────────────────────────────────────────

def is_bullish_engulfing(df: pd.DataFrame) -> bool:
    """
    Current candle is bullish AND its body fully engulfs the previous bearish body.
    """
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    prev_bearish = prev["close"] < prev["open"]
    curr_bullish = curr["close"] > curr["open"]
    engulfs = (
        curr["open"]  <= prev["close"]
        and curr["close"] >= prev["open"]
    )
    return prev_bearish and curr_bullish and engulfs


def is_bearish_engulfing(df: pd.DataFrame) -> bool:
    """
    Current candle is bearish AND its body fully engulfs the previous bullish body.
    Penalty flag.
    """
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    prev_bullish = prev["close"] > prev["open"]
    curr_bearish = curr["close"] < curr["open"]
    engulfs = (
        curr["open"]  >= prev["close"]
        and curr["close"] <= prev["open"]
    )
    return prev_bullish and curr_bearish and engulfs


def is_inside_bar(df: pd.DataFrame) -> bool:
    """
    Current candle's high < previous high AND low > previous low.
    Compression — significant when broken upward.
    """
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    return curr["high"] < prev["high"] and curr["low"] > prev["low"]


# ── Three-candle patterns ─────────────────────────────────────────────────────

def is_morning_star(df: pd.DataFrame) -> bool:
    """
    Three-candle bullish reversal:
      Candle -3: large bearish body
      Candle -2: small body / doji (gap down or at low)
      Candle -1: large bullish body closing above midpoint of candle -3
    """
    if len(df) < 3:
        return False
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    # Candle 1: bearish with meaningful body
    if c1["close"] >= c1["open"]:
        return False
    body1 = _body(c1["open"], c1["close"])
    rng1  = _total_range(c1["high"], c1["low"])
    if rng1 == 0 or body1 < 0.5 * rng1:
        return False
    # Candle 2: small body (star)
    rng2 = _total_range(c2["high"], c2["low"])
    if rng2 == 0:
        return False
    if _body(c2["open"], c2["close"]) > 0.3 * rng2:
        return False
    # Candle 3: bullish, closes above midpoint of candle 1
    midpoint_c1 = (c1["open"] + c1["close"]) / 2
    return c3["close"] > c3["open"] and c3["close"] > midpoint_c1


def is_evening_star(df: pd.DataFrame) -> bool:
    """
    Three-candle bearish reversal. Penalty flag.
    Candle -3: large bullish, Candle -2: small/doji, Candle -1: large bearish.
    """
    if len(df) < 3:
        return False
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    if c1["close"] <= c1["open"]:
        return False
    body1 = _body(c1["open"], c1["close"])
    rng1  = _total_range(c1["high"], c1["low"])
    if rng1 == 0 or body1 < 0.5 * rng1:
        return False
    rng2 = _total_range(c2["high"], c2["low"])
    if rng2 == 0 or _body(c2["open"], c2["close"]) > 0.3 * rng2:
        return False
    midpoint_c1 = (c1["open"] + c1["close"]) / 2
    return c3["close"] < c3["open"] and c3["close"] < midpoint_c1


# ── Convenience: scan all patterns ───────────────────────────────────────────

def detect_patterns(df: pd.DataFrame) -> dict:
    """
    Run all pattern detectors on the tail of df.
    Returns {
        bullish: list[str],   # active bullish patterns
        bearish: list[str],   # active bearish/warning patterns
        strongest_bull: str,  # highest-priority bullish pattern or ''
    }
    Priority (highest first): MORNING_STAR > BULLISH_ENGULFING > HAMMER > INVERTED_HAMMER > INSIDE_BAR
    """
    bull_checks = [
        ("MORNING_STAR",      is_morning_star),
        ("BULLISH_ENGULFING", is_bullish_engulfing),
        ("HAMMER",            is_hammer),
        ("INVERTED_HAMMER",   is_inverted_hammer),
        ("INSIDE_BAR",        is_inside_bar),
    ]
    bear_checks = [
        ("EVENING_STAR",      is_evening_star),
        ("BEARISH_ENGULFING", is_bearish_engulfing),
        ("SHOOTING_STAR",     is_shooting_star),
    ]

    bullish = [name for name, fn in bull_checks if fn(df)]
    bearish = [name for name, fn in bear_checks if fn(df)]

    return {
        "bullish":       bullish,
        "bearish":       bearish,
        "strongest_bull": bullish[0] if bullish else "",
    }
