import json
import logging
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore

import config

logger = logging.getLogger(__name__)

_db = None


def _get_db():
    global _db
    if _db is not None:
        return _db

    if not firebase_admin._apps:
        if not config.FIREBASE_CRED_JSON:
            raise RuntimeError("FIREBASE_CRED_JSON env var is not set")
        cred_dict = json.loads(config.FIREBASE_CRED_JSON)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)

    _db = firestore.client()
    return _db


# ── Snapshots ─────────────────────────────────────────────────────────────────

def save_snapshot(symbol: str, data: dict) -> bool:
    """
    Save a 5-min data snapshot to:
    idx_snapshots/{symbol}/snapshots/{timestamp}

    data should include: price, scores, indicators, forecast (optional)
    """
    try:
        db = _get_db()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        doc = {
            "symbol":    symbol,
            "timestamp": firestore.SERVER_TIMESTAMP,
            **data,
        }
        db.collection("idx_snapshots") \
          .document(symbol) \
          .collection("snapshots") \
          .document(ts) \
          .set(doc)
        logger.debug(f"Saved snapshot for {symbol} at {ts}")
        return True
    except Exception as e:
        logger.error(f"save_snapshot({symbol}): {e}")
        return False


def get_latest_snapshot(symbol: str) -> dict | None:
    """Retrieve the most recent snapshot for a symbol."""
    try:
        db = _get_db()
        docs = (
            db.collection("idx_snapshots")
              .document(symbol)
              .collection("snapshots")
              .order_by("timestamp", direction=firestore.Query.DESCENDING)
              .limit(1)
              .stream()
        )
        for doc in docs:
            return doc.to_dict()
        return None
    except Exception as e:
        logger.error(f"get_latest_snapshot({symbol}): {e}")
        return None


# ── Signals ───────────────────────────────────────────────────────────────────

def save_signal(symbol: str, signal_data: dict) -> bool:
    """
    Save a BUY/STRONG_BUY signal with AI verdict to:
    idx_signals/{timestamp}_{symbol}
    """
    try:
        db = _get_db()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        doc_id = f"{ts}_{symbol}"
        db.collection("idx_signals").document(doc_id).set({
            "symbol":    symbol,
            "timestamp": firestore.SERVER_TIMESTAMP,
            **signal_data,
        })
        logger.info(f"Saved signal for {symbol}")
        return True
    except Exception as e:
        logger.error(f"save_signal({symbol}): {e}")
        return False


def get_today_signals() -> list[dict]:
    """Retrieve all signals fired today (UTC date)."""
    try:
        db = _get_db()
        today_prefix = datetime.now(timezone.utc).strftime("%Y%m%d")
        docs = db.collection("idx_signals").stream()
        return [
            d.to_dict()
            for d in docs
            if d.id.startswith(today_prefix)
        ]
    except Exception as e:
        logger.error(f"get_today_signals: {e}")
        return []


# ── Forecasts ─────────────────────────────────────────────────────────────────

def save_forecast(symbol: str, forecast_data: dict) -> bool:
    """
    Save latest Prophet forecast to:
    idx_forecasts/{symbol}
    """
    try:
        db = _get_db()
        db.collection("idx_forecasts").document(symbol).set({
            "symbol":    symbol,
            "updated_at": firestore.SERVER_TIMESTAMP,
            **forecast_data,
        })
        return True
    except Exception as e:
        logger.error(f"save_forecast({symbol}): {e}")
        return False


def get_forecast(symbol: str) -> dict | None:
    try:
        db = _get_db()
        doc = db.collection("idx_forecasts").document(symbol).get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        logger.error(f"get_forecast({symbol}): {e}")
        return None


# ── News cache ────────────────────────────────────────────────────────────────

def save_news(symbol: str, news_data: dict) -> bool:
    """
    Cache news items for up to 24hr in:
    idx_news/{symbol}
    """
    try:
        db = _get_db()
        db.collection("idx_news").document(symbol).set({
            "symbol":    symbol,
            "updated_at": firestore.SERVER_TIMESTAMP,
            **news_data,
        })
        return True
    except Exception as e:
        logger.error(f"save_news({symbol}): {e}")
        return False


def get_news(symbol: str) -> dict | None:
    try:
        db = _get_db()
        doc = db.collection("idx_news").document(symbol).get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        logger.error(f"get_news({symbol}): {e}")
        return None


# ── Bot config ────────────────────────────────────────────────────────────────

def get_bot_config() -> dict:
    """
    Load runtime config from idx_bot_config/settings.
    Falls back to config.py defaults if not set.
    """
    try:
        db = _get_db()
        doc = db.collection("idx_bot_config").document("settings").get()
        if doc.exists:
            return doc.to_dict()
    except Exception as e:
        logger.error(f"get_bot_config: {e}")
    return {
        "stocks":   config.STOCKS,
        "capital":  config.IDX_CAPITAL,
    }


def get_watchlist() -> list[str]:
    """Load persisted watchlist from Firestore. Falls back to config.STOCKS."""
    try:
        cfg = get_bot_config()
        stocks = cfg.get("stocks")
        if stocks and isinstance(stocks, list):
            return stocks
    except Exception as e:
        logger.error(f"get_watchlist: {e}")
    return list(config.STOCKS)


def save_watchlist(stocks: list[str]) -> bool:
    """Persist current watchlist to Firestore."""
    try:
        db = _get_db()
        db.collection("idx_bot_config").document("settings").set(
            {"stocks": stocks, "updated_at": firestore.SERVER_TIMESTAMP},
            merge=True,
        )
        return True
    except Exception as e:
        logger.error(f"save_watchlist: {e}")
        return False


def save_bot_config(settings: dict) -> bool:
    try:
        db = _get_db()
        db.collection("idx_bot_config").document("settings").set({
            "updated_at": firestore.SERVER_TIMESTAMP,
            **settings,
        })
        return True
    except Exception as e:
        logger.error(f"save_bot_config: {e}")
        return False
