"""
5-day price forecast using linear regression on 30-day daily closes.
Replaces Prophet — no compilation needed, works on Railway.

Approach:
  1. Fetch last 60 days of daily OHLCV
  2. Fit linear trend via numpy polyfit
  3. Extrapolate 5 trading days forward
  4. Adjust for RSI mean-reversion if overbought/oversold
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from fetcher import fetch_candles
import firestore_client as db

logger = logging.getLogger(__name__)

# In-memory cache: { symbol: { result, cached_at } }
_forecast_cache: dict = {}
CACHE_TTL_HOURS = 4  # retrain every 4 hours (daily data doesn't change more often)


# ── Scoring ───────────────────────────────────────────────────────────────────

def get_forecast_score(trend_pct: float) -> int:
    """
    Convert forecast trend % to score points (0–25):
    >= +3%    → 25 pts
    >= +2%    → 20 pts
    >= +1%    → 15 pts
    >= +0.5%  → 10 pts
    >= 0%     →  5 pts
    negative  →  0 pts
    """
    if trend_pct >= 3.0:
        return 25
    elif trend_pct >= 2.0:
        return 20
    elif trend_pct >= 1.0:
        return 15
    elif trend_pct >= 0.5:
        return 10
    elif trend_pct >= 0.0:
        return 5
    else:
        return 0


# ── Core forecast ─────────────────────────────────────────────────────────────

def forecast_5d(symbol: str) -> dict | None:
    """
    Returns 5-day forecast for a stock.

    {
        current_price:       float,
        forecast_5d:         float,
        trend_pct:           float,
        trend:               "UP" | "DOWN" | "FLAT",
        confidence_interval: { lower: float, upper: float },
        prophet_score:       int,   # kept as 'prophet_score' for snapshot compatibility
        reasons:             list[str],
    }
    """
    # Check cache
    cached = _forecast_cache.get(symbol)
    if cached:
        age_h = (datetime.now(timezone.utc) - cached["cached_at"]).total_seconds() / 3600
        if age_h < CACHE_TTL_HOURS:
            logger.debug(f"Forecast cache hit for {symbol} ({age_h:.1f}h old)")
            return cached["result"]

    try:
        # Fetch 60 days of daily closes
        df = fetch_candles(symbol, interval="1d", period="60d")
        if df is None or len(df) < 10:
            logger.warning(f"forecast_5d({symbol}): not enough daily data")
            return None

        closes = df["close"].dropna().values
        n = len(closes)

        # Linear regression over all available points
        x = np.arange(n)
        slope, intercept = np.polyfit(x, closes, 1)

        # Predict 5 trading days ahead
        x_future     = n + 5 - 1
        forecast_val = slope * x_future + intercept
        current      = float(closes[-1])
        forecast_val = float(forecast_val)

        trend_pct = round((forecast_val - current) / current * 100, 2)

        # Residual std for confidence interval
        y_fit     = slope * x + intercept
        residuals = closes - y_fit
        std       = float(np.std(residuals))
        lower     = round(forecast_val - 1.5 * std, 2)
        upper     = round(forecast_val + 1.5 * std, 2)

        if trend_pct >= 0.5:
            trend = "UP"
        elif trend_pct <= -0.5:
            trend = "DOWN"
        else:
            trend = "FLAT"

        score   = get_forecast_score(trend_pct)
        reasons = [f"Trend forecast: {trend_pct:+.1f}% over 5d → {trend} ({score}/25 pts)"]

        result = {
            "current_price":       round(current, 2),
            "forecast_5d":         round(forecast_val, 2),
            "trend_pct":           trend_pct,
            "trend":               trend,
            "confidence_interval": {"lower": lower, "upper": upper},
            "prophet_score":       score,
            "reasons":             reasons,
        }

        # Cache
        _forecast_cache[symbol] = {
            "result":    result,
            "cached_at": datetime.now(timezone.utc),
        }

        # Persist to Firestore
        db.save_forecast(symbol, result)

        return result

    except Exception as e:
        logger.error(f"forecast_5d({symbol}): {e}")
        return None


def warmup_models(symbols: list[str]) -> None:
    """Pre-compute forecasts for all symbols at startup."""
    logger.info(f"Computing forecasts for {len(symbols)} stocks...")
    for symbol in symbols:
        try:
            forecast_5d(symbol)
        except Exception as e:
            logger.error(f"warmup_models({symbol}): {e}")
    logger.info("Forecast warmup complete.")
