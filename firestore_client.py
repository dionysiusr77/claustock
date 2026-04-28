"""
Firestore persistence layer.
Bridges the nightly scan (16:30) and the morning briefing (08:30).
Also tracks individual picks for future performance review.

Collections:
  v2_scans/{YYYY-MM-DD}             — full scan result (candidates + market + stats)
  v2_picks/{YYYY-MM-DD}_{symbol}    — individual pick with entry/target/SL
  v2_briefings/{YYYY-MM-DD}         — formatted Telegram briefing text (for /briefing command)
"""

import json
import logging
from datetime import datetime

import pytz

import config

logger = logging.getLogger(__name__)

_WIB = pytz.timezone(config.MARKET_TZ)
_db  = None   # lazy-initialised


def _get_db():
    global _db
    if _db is not None:
        return _db

    if not config.FIREBASE_CRED_JSON:
        logger.warning("FIREBASE_CRED_JSON not set — Firestore disabled")
        return None

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred_dict = json.loads(config.FIREBASE_CRED_JSON)
            cred      = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)

        _db = firestore.client()
        logger.info("Firestore initialised")
        return _db
    except Exception as e:
        logger.error("Firestore init failed: %s", e)
        return None


def _today() -> str:
    return datetime.now(_WIB).strftime("%Y-%m-%d")


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _safe_dict(obj) -> dict:
    """Strip non-serialisable values (DataFrames, NaN, inf) from a dict."""
    import math
    import numpy as np

    out = {}
    for k, v in obj.items():
        if hasattr(v, "to_dict"):         # DataFrame / Series
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        elif isinstance(v, np.bool_):
            out[k] = bool(v)
        elif isinstance(v, dict):
            out[k] = _safe_dict(v)
        elif isinstance(v, list):
            out[k] = [_safe_dict(i) if isinstance(i, dict) else i for i in v]
        else:
            out[k] = v
    return out


# ── Scan ──────────────────────────────────────────────────────────────────────

def save_scan(scan_data: dict, date: str | None = None) -> bool:
    """Persist the full scan result for the given date (defaults to today)."""
    db = _get_db()
    if db is None:
        return False

    date = date or _today()
    try:
        payload = {
            "date":      date,
            "saved_at":  datetime.now(_WIB).isoformat(),
            "stats":     scan_data.get("stats", {}),
            "market":    _safe_dict(scan_data.get("market", {})),
            "foreign_market": _safe_dict(scan_data.get("foreign_market") or {}),
            "candidates": [
                _safe_dict({
                    "symbol":          r["symbol"],
                    "total_score":     r["total_score"],
                    "verdict":         r["verdict"],
                    "setup":           r["setup"],
                    "divergence":      r.get("divergence", "NONE"),
                    "layer_scores":    r.get("layer_scores", {}),
                    "trade_levels":    r.get("trade_levels"),
                    "bearish_warnings": r.get("bearish_warnings", []),
                    "snapshot":        r.get("snapshot", {}),
                })
                for r in scan_data.get("candidates", [])
            ],
        }
        db.collection("v2_scans").document(date).set(payload)
        logger.info("Saved scan for %s (%d candidates)", date, len(payload["candidates"]))
        return True
    except Exception as e:
        logger.error("save_scan failed: %s", e)
        return False


def load_latest_scan() -> dict | None:
    """Load the most recent saved scan. Returns None if unavailable."""
    db = _get_db()
    if db is None:
        return None

    try:
        docs = (
            db.collection("v2_scans")
            .order_by("date", direction="DESCENDING")
            .limit(1)
            .stream()
        )
        for doc in docs:
            return doc.to_dict()
        return None
    except Exception as e:
        logger.error("load_latest_scan failed: %s", e)
        return None


# ── Briefing ──────────────────────────────────────────────────────────────────

def save_briefing(text: str, date: str | None = None) -> bool:
    """Save formatted briefing text for on-demand /briefing command."""
    db = _get_db()
    if db is None:
        return False

    date = date or _today()
    try:
        db.collection("v2_briefings").document(date).set({
            "date":     date,
            "text":     text,
            "saved_at": datetime.now(_WIB).isoformat(),
        })
        return True
    except Exception as e:
        logger.error("save_briefing failed: %s", e)
        return False


def load_latest_briefing() -> str | None:
    """Return the most recently saved briefing text."""
    db = _get_db()
    if db is None:
        return None

    try:
        docs = (
            db.collection("v2_briefings")
            .order_by("date", direction="DESCENDING")
            .limit(1)
            .stream()
        )
        for doc in docs:
            return doc.to_dict().get("text")
        return None
    except Exception as e:
        logger.error("load_latest_briefing failed: %s", e)
        return None


# ── Individual picks ──────────────────────────────────────────────────────────

def save_pick(result: dict, date: str | None = None) -> bool:
    """
    Save one pick for performance tracking.
    doc_id: YYYY-MM-DD_SYMBOL (e.g. 2026-04-28_BBRI)
    """
    db = _get_db()
    if db is None:
        return False

    date   = date or _today()
    symbol = result.get("symbol", "UNKNOWN")
    doc_id = f"{date}_{symbol}"

    try:
        payload = _safe_dict({
            "date":          date,
            "symbol":        symbol,
            "score":         result.get("total_score"),
            "verdict":       result.get("verdict"),
            "setup":         result.get("setup"),
            "divergence":    result.get("divergence"),
            "trade_levels":  result.get("trade_levels"),
            "snapshot":      result.get("snapshot", {}),
            "saved_at":      datetime.now(_WIB).isoformat(),
            "outcome":       None,   # filled later by performance tracker
        })
        db.collection("v2_picks").document(doc_id).set(payload)
        return True
    except Exception as e:
        logger.error("save_pick failed for %s: %s", symbol, e)
        return False
