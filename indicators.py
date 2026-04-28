"""
Technical indicators for daily OHLCV data.
All functions operate on a DataFrame with columns: open, high, low, close, volume.
Indicators are added in-place and the enriched DataFrame is returned.

Indicators computed:
  Trend    : ema20, ema50, ema200
  Momentum : rsi, macd, macd_signal, macd_hist, stoch_k, stoch_d
  Volatility: bb_upper, bb_mid, bb_lower, bb_pct, bb_width, atr, atr_pct
  Volume   : obv, vol_ma20, vol_ratio
  Derived  : trend_score, momentum_score (used by scorer)
  Divergence: divergence (string label on the latest row)
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ── Core indicators ───────────────────────────────────────────────────────────

def add_ema(df: pd.DataFrame) -> pd.DataFrame:
    df["ema20"]  = df["close"].ewm(span=config.EMA_FAST, adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=config.EMA_MID,  adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=config.EMA_SLOW, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / config.RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / config.RSI_PERIOD, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)).round(2)
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    ema_fast        = df["close"].ewm(span=config.MACD_FAST,   adjust=False).mean()
    ema_slow        = df["close"].ewm(span=config.MACD_SLOW,   adjust=False).mean()
    df["macd"]      = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=config.MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    sma              = df["close"].rolling(config.BB_PERIOD).mean()
    std              = df["close"].rolling(config.BB_PERIOD).std()
    df["bb_upper"]   = sma + config.BB_STD * std
    df["bb_mid"]     = sma
    df["bb_lower"]   = sma - config.BB_STD * std
    width            = df["bb_upper"] - df["bb_lower"]
    df["bb_width"]   = (width / sma).round(4)
    df["bb_pct"]     = ((df["close"] - df["bb_lower"]) / width.replace(0, np.nan)).round(4)
    return df


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift()).abs()
    lpc = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df["atr"]     = tr.ewm(alpha=1 / config.ATR_PERIOD, adjust=False).mean()
    df["atr_pct"] = (df["atr"] / df["close"] * 100).round(2)
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    direction = np.sign(df["close"].diff().fillna(0))
    df["obv"]  = (direction * df["volume"]).cumsum().astype(np.int64)
    return df


def add_stochastic(df: pd.DataFrame) -> pd.DataFrame:
    low_k    = df["low"].rolling(config.STOCH_K).min()
    high_k   = df["high"].rolling(config.STOCH_K).max()
    range_k  = (high_k - low_k).replace(0, np.nan)
    df["stoch_k"] = ((df["close"] - low_k) / range_k * 100).round(2)
    df["stoch_d"] = df["stoch_k"].rolling(config.STOCH_D).mean().round(2)
    return df


def add_volume_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df["vol_ma20"]  = df["volume"].rolling(config.VOL_MA_PERIOD).mean()
    df["vol_ratio"] = (df["volume"] / df["vol_ma20"].replace(0, np.nan)).round(2)
    return df


# ── Divergence detection ──────────────────────────────────────────────────────

def _pivot_lows(series: pd.Series, window: int) -> pd.Series:
    """Boolean mask: True where series is the minimum of its centered window."""
    roll_min = series.rolling(2 * window + 1, center=True, min_periods=window + 1).min()
    return series == roll_min


def _pivot_highs(series: pd.Series, window: int) -> pd.Series:
    """Boolean mask: True where series is the maximum of its centered window."""
    roll_max = series.rolling(2 * window + 1, center=True, min_periods=window + 1).max()
    return series == roll_max


def detect_divergence(df: pd.DataFrame) -> str:
    """
    Classify RSI divergence on the most recent candles.

    Returns one of:
        'BULLISH'        — price lower low, RSI higher low  (reversal, bullish)
        'HIDDEN_BULLISH' — price higher low, RSI lower low  (continuation, bullish)
        'BEARISH'        — price higher high, RSI lower high (reversal, bearish)
        'HIDDEN_BEARISH' — price lower high, RSI higher high (continuation, bearish)
        'NONE'

    Rules:
      - Uses last DIV_LOOKBACK candles
      - Pivot detection window: DIV_PIVOT_WINDOW bars each side
      - Requires at least 2 confirmed pivots of each type
      - Bearish divergence takes precedence (it triggers a penalty in scorer)
    """
    window   = config.DIV_PIVOT_WINDOW
    lookback = config.DIV_LOOKBACK

    if "rsi" not in df.columns or len(df) < lookback:
        return "NONE"

    sub   = df.tail(lookback).copy()
    price = sub["close"]
    rsi   = sub["rsi"]

    # ── Check bearish divergence first (higher prio — triggers penalty) ───
    ph_mask = _pivot_highs(price, window)
    ph_idx  = price.index[ph_mask]
    if len(ph_idx) >= 2:
        ph1, ph2 = ph_idx[-2], ph_idx[-1]
        if price[ph2] > price[ph1] and rsi[ph2] < rsi[ph1]:
            return "BEARISH"
        if price[ph2] < price[ph1] and rsi[ph2] > rsi[ph1]:
            return "HIDDEN_BEARISH"

    # ── Check bullish divergence ──────────────────────────────────────────
    pl_mask = _pivot_lows(price, window)
    pl_idx  = price.index[pl_mask]
    if len(pl_idx) >= 2:
        pl1, pl2 = pl_idx[-2], pl_idx[-1]
        if price[pl2] < price[pl1] and rsi[pl2] > rsi[pl1]:
            return "BULLISH"
        if price[pl2] > price[pl1] and rsi[pl2] < rsi[pl1]:
            return "HIDDEN_BULLISH"

    return "NONE"


def add_divergence(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'divergence' column with the latest divergence label."""
    div = detect_divergence(df)
    df["divergence"] = "NONE"
    if div != "NONE" and len(df):
        df.at[df.index[-1], "divergence"] = div
    return df


# ── Breakout flag ─────────────────────────────────────────────────────────────

def add_breakout(df: pd.DataFrame) -> pd.DataFrame:
    """
    breakout_bull: today's close > highest close of previous N days
    breakout_bear: today's close < lowest  close of previous N days
    """
    n = config.BREAKOUT_PERIOD
    df["prev_high"] = df["close"].shift(1).rolling(n).max()
    df["prev_low"]  = df["close"].shift(1).rolling(n).min()
    df["breakout_bull"] = df["close"] > df["prev_high"]
    df["breakout_bear"] = df["close"] < df["prev_low"]
    return df


# ── Convenience: run all indicators ──────────────────────────────────────────

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run every indicator on a daily OHLCV DataFrame and return the enriched copy.
    Safe to call on DataFrames with 60+ rows.
    """
    if df is None or len(df) < 60:
        return pd.DataFrame()

    df = df.copy()
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_atr(df)
    df = add_obv(df)
    df = add_stochastic(df)
    df = add_volume_ratio(df)
    df = add_breakout(df)
    df = add_divergence(df)
    return df


# ── Latest-row snapshot ───────────────────────────────────────────────────────

def latest_snapshot(df: pd.DataFrame) -> dict:
    """
    Extract the last row's indicator values into a flat dict.
    Used by scorer.py and ai_briefing.py.
    """
    if df.empty:
        return {}

    row = df.iloc[-1]

    def _v(col: str, default=None):
        val = row.get(col, default)
        return None if (val is None or (isinstance(val, float) and np.isnan(val))) else val

    return {
        # Price
        "date":          str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1]),
        "open":          _v("open"),
        "high":          _v("high"),
        "low":           _v("low"),
        "close":         _v("close"),
        "volume":        _v("volume"),
        # Trend
        "ema20":         _v("ema20"),
        "ema50":         _v("ema50"),
        "ema200":        _v("ema200"),
        # Momentum
        "rsi":           _v("rsi"),
        "macd":          _v("macd"),
        "macd_signal":   _v("macd_signal"),
        "macd_hist":     _v("macd_hist"),
        "stoch_k":       _v("stoch_k"),
        "stoch_d":       _v("stoch_d"),
        # Volatility
        "bb_upper":      _v("bb_upper"),
        "bb_mid":        _v("bb_mid"),
        "bb_lower":      _v("bb_lower"),
        "bb_pct":        _v("bb_pct"),
        "bb_width":      _v("bb_width"),
        "atr":           _v("atr"),
        "atr_pct":       _v("atr_pct"),
        # Volume
        "obv":           _v("obv"),
        "vol_ma20":      _v("vol_ma20"),
        "vol_ratio":     _v("vol_ratio"),
        # Patterns
        "breakout_bull": bool(_v("breakout_bull", False)),
        "breakout_bear": bool(_v("breakout_bear", False)),
        "divergence":    _v("divergence", "NONE"),
        # 52-week context
        "high_52w":      df["close"].max(),
        "low_52w":       df["close"].min(),
        "pct_from_52w_high": round((row["close"] - df["close"].max()) / df["close"].max() * 100, 2),
        "pct_from_52w_low":  round((row["close"] - df["close"].min()) / df["close"].min() * 100, 2),
    }
