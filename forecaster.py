"""
Prophet-based 5-day price forecast.
- Trains on 1yr daily close prices
- Caches model per stock in memory, retrains weekly
- Run once at startup for all watchlist stocks
"""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from prophet import Prophet

from fetcher import fetch_candles
import firestore_client as db

logger = logging.getLogger(__name__)

# In-memory cache: { symbol: { model, trained_at } }
_model_cache: dict = {}

# Retrain if model is older than 7 days
RETRAIN_AFTER_DAYS = 7


# ── Scoring ───────────────────────────────────────────────────────────────────

def get_prophet_score(trend_pct: float) -> int:
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


# ── Model training ────────────────────────────────────────────────────────────

def train_model(symbol: str, df_daily: pd.DataFrame) -> Prophet | None:
    """
    Train Prophet on 1yr daily close prices.
    df_daily must have columns: ds (datetime), y (float close price)
    Returns fitted Prophet model.
    """
    try:
        model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,       # no intraday at daily level
            changepoint_prior_scale=0.05,  # conservative — IDX less volatile than crypto
        )
        # Suppress Prophet/Stan output
        import logging as _log
        _log.getLogger("prophet").setLevel(_log.WARNING)
        _log.getLogger("cmdstanpy").setLevel(_log.WARNING)

        model.fit(df_daily[["ds", "y"]])
        return model
    except Exception as e:
        logger.error(f"train_model({symbol}): {e}")
        return None


def _needs_retrain(symbol: str) -> bool:
    if symbol not in _model_cache:
        return True
    trained_at = _model_cache[symbol].get("trained_at")
    if trained_at is None:
        return True
    return datetime.now(timezone.utc) - trained_at > timedelta(days=RETRAIN_AFTER_DAYS)


# ── Main forecast function ────────────────────────────────────────────────────

def forecast_5d(symbol: str) -> dict | None:
    """
    Returns 5-day forecast for a stock.
    Uses cached model if trained within the last 7 days.

    Returns:
    {
        current_price:       float,
        forecast_5d:         float,   # predicted price in 5 trading days
        trend_pct:           float,   # % change expected
        trend:               "UP" | "DOWN" | "FLAT",
        confidence_interval: { lower: float, upper: float },
        prophet_score:       int,     # 0–25 pts
        reasons:             list[str],
    }
    """
    # 1. Load or retrain model
    if _needs_retrain(symbol):
        logger.info(f"Training Prophet model for {symbol}...")
        df_daily = fetch_candles(symbol, interval="1d", period="1y")
        if df_daily is None or df_daily.empty:
            logger.warning(f"forecast_5d({symbol}): no daily data for training")
            return None

        # Prophet needs ds + y columns
        df_prophet = df_daily[["close"]].copy().reset_index()
        df_prophet.columns = ["ds", "y"]
        # Strip timezone from ds — Prophet doesn't handle tz-aware datetimes
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"]).dt.tz_localize(None)
        df_prophet = df_prophet.dropna()

        if len(df_prophet) < 30:
            logger.warning(f"forecast_5d({symbol}): not enough daily data ({len(df_prophet)} rows)")
            return None

        model = train_model(symbol, df_prophet)
        if model is None:
            return None

        _model_cache[symbol] = {
            "model":      model,
            "trained_at": datetime.now(timezone.utc),
            "last_close": float(df_prophet["y"].iloc[-1]),
        }
        logger.info(f"Prophet model trained for {symbol}")
    else:
        logger.debug(f"Using cached Prophet model for {symbol}")

    # 2. Predict next 5 business days
    try:
        cached    = _model_cache[symbol]
        model     = cached["model"]
        cur_price = cached["last_close"]

        future   = model.make_future_dataframe(periods=5, freq="B")
        forecast = model.predict(future)

        forecast_row  = forecast.iloc[-1]
        forecast_price = round(float(forecast_row["yhat"]), 2)
        lower          = round(float(forecast_row["yhat_lower"]), 2)
        upper          = round(float(forecast_row["yhat_upper"]), 2)

        trend_pct = round((forecast_price - cur_price) / cur_price * 100, 2)

        if trend_pct >= 0.5:
            trend = "UP"
        elif trend_pct <= -0.5:
            trend = "DOWN"
        else:
            trend = "FLAT"

        score   = get_prophet_score(trend_pct)
        reasons = [f"Prophet: {trend_pct:+.1f}% over 5d → {trend} ({score}/25 pts)"]

        result = {
            "current_price":       cur_price,
            "forecast_5d":         forecast_price,
            "trend_pct":           trend_pct,
            "trend":               trend,
            "confidence_interval": {"lower": lower, "upper": upper},
            "prophet_score":       score,
            "reasons":             reasons,
        }

        # Persist to Firestore
        db.save_forecast(symbol, result)

        return result

    except Exception as e:
        logger.error(f"forecast_5d({symbol}) predict failed: {e}")
        return None


# ── Startup: train all models ─────────────────────────────────────────────────

def warmup_models(symbols: list[str]) -> None:
    """
    Train Prophet models for all symbols at startup.
    Skips any that already have a fresh cached model.
    """
    logger.info(f"Warming up Prophet models for {len(symbols)} stocks...")
    for symbol in symbols:
        try:
            forecast_5d(symbol)
        except Exception as e:
            logger.error(f"warmup_models({symbol}): {e}")
    logger.info("Prophet warmup complete.")
