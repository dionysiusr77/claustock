from __future__ import annotations

"""
Invezgo API client — drop-in replacement for the yfinance + IDX-scraping data layer.

Replaces:
  fetcher.fetch_daily_batch()          → fetch_daily_batch()
  fetcher.fetch_daily()                → fetch_daily()
  fetcher.fetch_market_breadth()       → fetch_market_breadth()
  fetcher.fetch_foreign_flow_stock()   → fetch_foreign_flow_stock()
  fetcher.fetch_foreign_flow_market()  → fetch_foreign_flow_market()
  fetcher.filter_liquid()              → filter_liquid()
  universe.get_universe()              → get_universe()

All functions preserve the same return-type contracts so nothing upstream
(indicators.py, screener.py, market_breadth.py) needs to change.
"""

import logging
from datetime import datetime, timedelta
from functools import lru_cache

import pandas as pd
from invezgo import InvezgoClient as _SDK

import config

logger = logging.getLogger(__name__)

# ── Singleton SDK client ──────────────────────────────────────────────────────

_client = None   # InvezgoClient singleton


def _get_client() -> _SDK:
    global _client
    if _client is None:
        if not config.INVEZGO_API_KEY:
            raise RuntimeError("INVEZGO_API_KEY is not set in .env")
        _client = _SDK(api_key=config.INVEZGO_API_KEY)
    return _client


# ── Date helpers ──────────────────────────────────────────────────────────────

def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _date_range(days: int) -> tuple[str, str]:
    today = datetime.now()
    return _fmt(today - timedelta(days=days)), _fmt(today)


def _last_n_trading_dates(n: int) -> list[str]:
    dates: list[str] = []
    dt = datetime.now()
    while len(dates) < n:
        dt -= timedelta(days=1)
        if dt.weekday() < 5:
            dates.append(_fmt(dt))
    return dates


# ── Symbol normalisation ──────────────────────────────────────────────────────

def _code(symbol: str) -> str:
    """Strip .JK suffix → Invezgo uses plain ticker codes (e.g. 'BBCA')."""
    return symbol.upper().replace(".JK", "")


def _jk(code: str) -> str:
    """Add .JK suffix back for compatibility with the rest of the codebase."""
    return code if code.endswith(".JK") else f"{code}.JK"


# ── OHLCV parsing ─────────────────────────────────────────────────────────────

def _parse_ohlcv(raw) -> pd.DataFrame:
    """
    Normalise Invezgo chart response to a standard OHLCV DataFrame.
    Handles list-of-dicts with various key naming conventions.
    Returns empty DataFrame on failure.
    """
    if not raw:
        return pd.DataFrame()

    # Unwrap if the response is a dict with a data/chart key
    if isinstance(raw, dict):
        raw = raw.get("data") or raw.get("chart") or raw.get("candles") or []

    if not raw:
        return pd.DataFrame()

    records = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        # Key aliases across different Invezgo endpoint responses
        date_val  = row.get("date") or row.get("t") or row.get("d")
        open_val  = row.get("open") or row.get("o")
        high_val  = row.get("high") or row.get("h")
        low_val   = row.get("low")  or row.get("l")
        close_val = row.get("close") or row.get("c")
        vol_val   = row.get("volume") or row.get("v") or row.get("vol")

        if None in (date_val, close_val):
            continue

        records.append({
            "date":   pd.to_datetime(date_val),
            "open":   float(open_val  or close_val),
            "high":   float(high_val  or close_val),
            "low":    float(low_val   or close_val),
            "close":  float(close_val),
            "volume": int(vol_val or 0),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("date").sort_index()
    df.index = pd.DatetimeIndex(df.index)
    return df


# ── Universe ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _cached_stock_list() -> list[dict]:
    try:
        return _get_client().analysis.get_stock_list() or []
    except Exception as e:
        logger.error("get_stock_list failed: %s", e)
        return []


def get_universe(tier: str = config.UNIVERSE) -> list[str]:
    """
    Returns all BEI-listed tickers in 'XXXX.JK' format.
    Tier parameter is accepted for API compatibility but Invezgo always
    returns the full listing — filtering happens in filter_liquid().
    Falls back to the hardcoded IDX80 universe on failure.
    """
    stocks = _cached_stock_list()
    if not stocks:
        logger.warning("Invezgo stock list empty — falling back to local universe")
        from universe import get_universe as _local_universe
        return _local_universe(tier)

    tickers = []
    for s in stocks:
        code = (
            s.get("code") or s.get("stock_code") or
            s.get("stockCode") or s.get("ticker") or ""
        ).strip().upper()
        if code:
            tickers.append(_jk(code))

    logger.info("Invezgo universe: %d stocks", len(tickers))
    return sorted(set(tickers))


# ── Daily OHLCV ───────────────────────────────────────────────────────────────

def fetch_daily(symbol: str, days: int = 365, min_rows: int = 60) -> pd.DataFrame | None:
    """Single-stock daily OHLCV. Returns None on failure."""
    from_date, to_date = _date_range(days)
    try:
        raw = _get_client().analysis.get_chart(
            code=_code(symbol),
            from_date=from_date,
            to_date=to_date,
        )
        df = _parse_ohlcv(raw)
        if len(df) < min_rows:
            logger.debug("Insufficient OHLCV data for %s (%d rows, need %d)", symbol, len(df), min_rows)
            return None
        return df
    except Exception as e:
        logger.debug("fetch_daily failed for %s: %s", symbol, e)
        return None


def fetch_daily_batch(symbols: list[str], days: int = 365, min_rows: int = 60) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLCV for multiple symbols.
    Sequential — Invezgo doesn't need the concurrent pool workaround.
    Returns {symbol: DataFrame}.
    """
    result: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    total = len(symbols)

    for i, sym in enumerate(symbols, 1):
        df = fetch_daily(sym, days=days, min_rows=min_rows)
        if df is not None:
            result[sym] = df
        else:
            failed.append(sym)
        if i % 25 == 0:
            logger.info("OHLCV download: %d/%d done", i, total)

    ok = len(result)
    if failed:
        logger.warning("fetch_daily_batch: %d/%d failed — %s", len(failed), total,
                       ", ".join(failed[:5]) + (" ..." if len(failed) > 5 else ""))
    logger.info("fetch_daily_batch complete: %d/%d symbols OK", ok, total)
    return result


# ── Market breadth (IHSG + sectors) ──────────────────────────────────────────

def fetch_market_breadth() -> dict:
    """
    Fetch yesterday's close + change% for IHSG and sectoral indices.
    Uses yfinance batch for sectors; if ^JKSE fails in the batch (common),
    falls back to the retry-enabled _fetch_ihsg() helper.
    """
    from fetcher import fetch_market_breadth as _yf_breadth
    result = _yf_breadth()

    # ^JKSE often fails in the batch download — retry it individually
    if not result.get("IHSG", {}).get("close"):
        from market_breadth import _fetch_ihsg
        df = _fetch_ihsg("5d")
        if not df.empty and len(df) >= 2:
            try:
                df.columns = [c.lower() for c in df.columns]
                df = df.dropna(subset=["close"])
                if len(df) >= 2:
                    prev  = float(df["close"].iloc[-2])
                    last  = float(df["close"].iloc[-1])
                    chg   = (last - prev) / prev * 100
                    result["IHSG"] = {
                        "close":      round(last, 2),
                        "change_pct": round(chg, 2),
                        "direction":  "UP" if chg > 0 else ("DOWN" if chg < 0 else "FLAT"),
                    }
                    logger.info("IHSG fallback fetch OK: %.2f (%.2f%%)", last, chg)
            except Exception as e:
                logger.warning("IHSG fallback parse failed: %s", e)

    if not result.get("IHSG", {}).get("close"):
        logger.warning("IHSG data unavailable — market breadth header will show None")

    return result


# ── Per-stock foreign flow ─────────────────────────────────────────────────────

def _parse_foreign_day(row: dict, date_str: str) -> dict | None:
    """Normalise one day's foreign flow row from get_summary_stock response."""
    if not row:
        return None

    buy  = float(row.get("foreignBuy")  or row.get("foreign_buy")  or row.get("buyVal")  or 0)
    sell = float(row.get("foreignSell") or row.get("foreign_sell") or row.get("sellVal") or 0)
    net  = float(row.get("foreignNet")  or row.get("foreign_net")  or row.get("netVal")  or (buy - sell))

    return {
        "date":         date_str,
        "buy_val_idr":  buy,
        "sell_val_idr": sell,
        "net_val_idr":  net,
        "direction":    "BUY" if net > 0 else ("SELL" if net < 0 else "NEUTRAL"),
    }


def fetch_foreign_flow_stock(symbol: str, lookback_days: int = 5) -> dict | None:
    """
    Per-stock foreign flow with consecutive streak count.
    Replaces both fetcher.fetch_foreign_flow_stock() and foreign_flow.get_foreign_flow_enriched().

    Returns same enriched dict contract:
      {date, buy_val_idr, sell_val_idr, net_val_idr, direction,
       consecutive_buy_days, consecutive_sell_days, history}
    """
    dates = _last_n_trading_dates(lookback_days)
    from_date = dates[-1]   # oldest
    to_date   = dates[0]    # most recent

    try:
        raw = _get_client().analysis.get_summary_stock(
            code=_code(symbol),
            from_date=from_date,
            to_date=to_date,
            investor="f",
            market="RG",
        )
    except Exception as e:
        logger.debug("get_summary_stock failed for %s: %s", symbol, e)
        return None

    # Unwrap response envelope
    rows = raw
    if isinstance(raw, dict):
        rows = raw.get("data") or raw.get("summary") or raw.get("items") or []

    if not rows:
        return None

    # Sort newest first and parse each day
    history: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        date_str = (row.get("date") or row.get("tradingDate") or row.get("d") or "")
        parsed = _parse_foreign_day(row, date_str)
        if parsed:
            history.append(parsed)

    if not history:
        return None

    # Sort by date descending (most recent first)
    history.sort(key=lambda x: x["date"], reverse=True)

    result = history[0].copy()
    direction_today = result["direction"]

    consec_buy = consec_sell = 0
    if direction_today == "BUY":
        for day in history:
            if day["direction"] == "BUY":
                consec_buy += 1
            else:
                break
    elif direction_today == "SELL":
        for day in history:
            if day["direction"] == "SELL":
                consec_sell += 1
            else:
                break

    result["consecutive_buy_days"]  = consec_buy
    result["consecutive_sell_days"] = consec_sell
    result["history"]               = history
    return result


# ── Market-wide foreign flow ──────────────────────────────────────────────────

def fetch_foreign_flow_market(date_str: str | None = None) -> dict | None:
    """
    Aggregate market foreign net flow for a given date.
    Uses get_top_foreign() and sums net values across all stocks.
    """
    if not date_str:
        dates = _last_n_trading_dates(1)
        date_str = dates[0] if dates else datetime.now().strftime("%Y-%m-%d")

    try:
        raw = _get_client().analysis.get_top_foreign(date=date_str)
    except Exception as e:
        logger.debug("get_top_foreign failed for %s: %s", date_str, e)
        return None

    # Response shape: {"accum": [...], "dist": [...]}
    # accum = stocks where asing is net buying, dist = net selling
    if not isinstance(raw, dict):
        return None

    accum_rows = raw.get("accum") or []
    dist_rows  = raw.get("dist")  or []

    total_buy = total_sell = 0.0
    for row in accum_rows:
        if isinstance(row, dict):
            total_buy += float(row.get("foreignBuy") or row.get("netVal") or 0)
    for row in dist_rows:
        if isinstance(row, dict):
            total_sell += float(row.get("foreignSell") or row.get("netVal") or 0)

    net = total_buy - total_sell
    return {
        "date":         date_str,
        "buy_val_idr":  total_buy,
        "sell_val_idr": total_sell,
        "net_val_idr":  net,
        "direction":    "BUY" if net > 0 else ("SELL" if net < 0 else "NEUTRAL"),
    }


# ── Liquidity filter ──────────────────────────────────────────────────────────

def filter_liquid(
    symbols: list[str],
    min_price: float = config.MIN_PRICE,
    max_price: float = config.MAX_PRICE,
    min_avg_val: float = config.MIN_AVG_DAILY_VAL,
    lookback_days: int = 20,
) -> list[str]:
    """
    Liquidity filter using the Invezgo screener when possible,
    falling back to OHLCV-based filtering if screener fails.
    """
    # ── Try Invezgo screener first ────────────────────────────────────────────
    try:
        raw = _get_client().screener.screen(
            columns=["code", "lastPrice", "avgValue"],
            conditions=[
                {"column": "lastPrice", "operator": ">=", "value": min_price},
                {"column": "lastPrice", "operator": "<=", "value": max_price},
                {"column": "avgValue",  "operator": ">=", "value": min_avg_val},
            ],
        )
        rows = raw if isinstance(raw, list) else (raw or {}).get("data", [])
        if rows:
            liquid = []
            symbol_set = {_code(s) for s in symbols}
            for row in rows:
                code = (row.get("code") or row.get("stockCode") or "").strip().upper()
                if code and code in symbol_set:
                    liquid.append(_jk(code))
            if liquid:
                logger.info("Invezgo screener: %d/%d liquid", len(liquid), len(symbols))
                return liquid
    except Exception as e:
        logger.debug("Invezgo screener failed, falling back to OHLCV filter: %s", e)

    # ── Fallback: OHLCV-based filter (same as fetcher.filter_liquid) ──────────
    data = fetch_daily_batch(symbols, days=30, min_rows=5)
    liquid: list[str] = []
    for sym, df in data.items():
        if df.empty or len(df) < 5:
            continue
        last_price = df["close"].iloc[-1]
        if not (min_price <= last_price <= max_price):
            continue
        recent  = df.tail(lookback_days)
        avg_val = (recent["close"] * recent["volume"]).mean()
        if avg_val >= min_avg_val:
            liquid.append(sym)

    logger.info("OHLCV liquidity filter: %d/%d liquid", len(liquid), len(symbols))
    return liquid
