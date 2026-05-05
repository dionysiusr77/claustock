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


def _is_equity(code: str) -> bool:
    """Filter out warrants (-W), rights (-R), and other non-equity instruments."""
    return "-" not in code


def get_universe(tier: str = config.UNIVERSE) -> list[str]:
    """
    Returns all BEI-listed equity tickers in 'XXXX.JK' format.
    Warrants (-W) and rights (-R) are excluded — they have no OHLCV data.
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
        if code and _is_equity(code):
            tickers.append(_jk(code))

    logger.info("Invezgo universe: %d equities (warrants/rights excluded)", len(tickers))
    return sorted(set(tickers))


# ── Chart request (stock + index) ─────────────────────────────────────────────

_INVEZGO_BASE = "https://api.invezgo.com"


def _chart_request(code: str, from_date: str, to_date: str, kind: str = "stock") -> list:
    """
    GET /analysis/chart/{kind}/{code}
    kind: "stock" for equities, "index" for indices (e.g. COMPOSITE)
    Returns raw list or raises on HTTP error.
    """
    import requests as _req
    url = f"{_INVEZGO_BASE}/analysis/chart/{kind}/{code}"
    resp = _req.get(
        url,
        params={"from": from_date, "to": to_date},
        headers={"Authorization": f"Bearer {config.INVEZGO_API_KEY}"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _intraday_request(code: str, kind: str = "stock") -> list:
    """
    GET /analysis/intraday-data/{code}  for stocks
    GET /analysis/intraday-index/{code} for indices (e.g. COMPOSITE)
    Returns today's intraday candles or raises on HTTP error.
    """
    import requests as _req
    path = "intraday-index" if kind == "index" else "intraday-data"
    url  = f"{_INVEZGO_BASE}/analysis/{path}/{code}"
    resp = _req.get(
        url,
        params={"market": "RG"},
        headers={"Authorization": f"Bearer {config.INVEZGO_API_KEY}"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


# ── Daily OHLCV ───────────────────────────────────────────────────────────────

def fetch_daily(symbol: str, days: int = 365, min_rows: int = 60) -> pd.DataFrame | None:
    """Single-stock daily OHLCV. Returns None on failure."""
    from_date, to_date = _date_range(days)
    try:
        raw = _chart_request(_code(symbol), from_date, to_date, kind="stock")
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
        fail_rate = len(failed) / total if total else 0
        msg = "fetch_daily_batch: %d/%d no data — %s"
        args = (len(failed), total, ", ".join(failed[:5]) + (" ..." if len(failed) > 5 else ""))
        # Only warn if >30% fail — likely an API issue; otherwise it's normal (new/suspended stocks)
        if fail_rate > 0.30:
            logger.warning(msg, *args)
        else:
            logger.debug(msg, *args)
    logger.info("fetch_daily_batch complete: %d/%d symbols OK", ok, total)
    return result


# ── Intraday (Sesi 1 snapshot) ────────────────────────────────────────────────

import pytz as _pytz
_WIB = _pytz.timezone("Asia/Jakarta")

# Sesi 1: 09:00–12:00 WIB
_SESI1_START = "09:00"
_SESI1_END   = "12:00"


def fetch_intraday_sesi1(
    symbol: str,
    prev_close: float | None = None,
    kind: str = "stock",
) -> dict | None:
    """
    Fetch Sesi 1 (09:00–12:00 WIB) intraday snapshot for today.

    kind: "stock" for equities, "index" for indices (e.g. COMPOSITE)
    Returns:
      { open, high, low, close, volume,
        pct_change,   # vs prev_close (D-1) if provided
        candles: int  # number of intraday candles in Sesi 1
      }
    """
    try:
        raw = _intraday_request(_code(symbol), kind=kind)
    except Exception as e:
        logger.warning("intraday request failed for %s: %s", symbol, e)
        return None

    df = _parse_ohlcv(raw)
    if df.empty:
        logger.debug("get_intraday: empty OHLCV for %s (raw type=%s len=%s)",
                     symbol, type(raw).__name__, len(raw) if raw else 0)
        return None

    # Invezgo returns WIB (Asia/Jakarta) timestamps — localise as WIB if naive
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Jakarta")
    else:
        df.index = df.index.tz_convert(_WIB)

    sesi1 = df.between_time(_SESI1_START, _SESI1_END)
    if sesi1.empty:
        return None

    open_s1  = float(sesi1["open"].iloc[0])
    high_s1  = float(sesi1["high"].max())
    low_s1   = float(sesi1["low"].min())
    close_s1 = float(sesi1["close"].iloc[-1])
    vol_s1   = int(sesi1["volume"].sum())

    pct_chg = None
    if prev_close and prev_close > 0:
        pct_chg = round((close_s1 - prev_close) / prev_close * 100, 2)

    return {
        "open":       round(open_s1, 0),
        "high":       round(high_s1, 0),
        "low":        round(low_s1, 0),
        "close":      round(close_s1, 0),
        "volume":     vol_s1,
        "pct_change": pct_chg,
        "candles":    len(sesi1),
    }


def fetch_intraday_batch(
    symbols: list[str],
    prev_closes: dict[str, float] | None = None,
    max_workers: int = 4,
) -> dict[str, dict | None]:
    """
    Fetch Sesi 1 intraday snapshots concurrently for multiple symbols.
    prev_closes: {symbol: D-1 close price} for pct_change calculation.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    prev_closes = prev_closes or {}
    results: dict[str, dict | None] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(fetch_intraday_sesi1, sym, prev_closes.get(sym)): sym
            for sym in symbols
        }
        for future in as_completed(future_map):
            sym = future_map[future]
            try:
                results[sym] = future.result()
            except Exception:
                results[sym] = None

    ok = sum(1 for v in results.values() if v is not None)
    logger.info("fetch_intraday_batch: %d/%d symbols OK", ok, len(symbols))
    return results


# ── Market breadth (IHSG + sectors) ──────────────────────────────────────────

def _fetch_ihsg_invezgo() -> dict | None:
    """Fetch IHSG via GET /analysis/chart/index/COMPOSITE."""
    from_date, to_date = _date_range(7)   # need at least 2 trading days
    try:
        raw = _chart_request("COMPOSITE", from_date, to_date, kind="index")
        df = _parse_ohlcv(raw)
        if df.empty or len(df) < 2:
            logger.debug("IHSG index chart: insufficient rows (%d)", len(df))
            return None
        prev = float(df["close"].iloc[-2])
        last = float(df["close"].iloc[-1])
        chg  = (last - prev) / prev * 100
        logger.info("IHSG via Invezgo: %.2f (%.2f%%)", last, chg)
        return {
            "close":      round(last, 2),
            "change_pct": round(chg, 2),
            "direction":  "UP" if chg > 0 else ("DOWN" if chg < 0 else "FLAT"),
        }
    except Exception as e:
        logger.debug("Invezgo IHSG index chart failed: %s", e)
        return None


def fetch_market_breadth() -> dict:
    """
    Fetch yesterday's close + change% for IHSG and sectoral indices.
    Uses yfinance batch for sectors. IHSG is fetched via Invezgo first
    (more reliable than ^JKSE on yfinance), with yfinance as fallback.
    """
    from fetcher import fetch_market_breadth as _yf_breadth
    result = _yf_breadth()

    # Prefer Invezgo for IHSG — yfinance ^JKSE is unreliable
    ihsg = _fetch_ihsg_invezgo()
    if ihsg:
        result["IHSG"] = ihsg
    elif not result.get("IHSG", {}).get("close"):
        # Last resort: yfinance individual retry with period fallback
        from market_breadth import _fetch_ihsg
        df = _fetch_ihsg("5d")
        if not df.empty and len(df) >= 2:
            try:
                df.columns = [c.lower() for c in df.columns]
                df = df.dropna(subset=["close"])
                if len(df) >= 2:
                    prev = float(df["close"].iloc[-2])
                    last = float(df["close"].iloc[-1])
                    chg  = (last - prev) / prev * 100
                    result["IHSG"] = {
                        "close":      round(last, 2),
                        "change_pct": round(chg, 2),
                        "direction":  "UP" if chg > 0 else ("DOWN" if chg < 0 else "FLAT"),
                    }
                    logger.info("IHSG via yfinance fallback: %.2f (%.2f%%)", last, chg)
            except Exception as e:
                logger.warning("IHSG yfinance fallback parse failed: %s", e)

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
    Aggregate market foreign net flow.
    Uses GET /analysis/top/foreign — sums buy/sell across returned rows.
    """
    import requests as _req

    if not date_str:
        dates = _last_n_trading_dates(1)
        date_str = dates[0] if dates else datetime.now().strftime("%Y-%m-%d")

    url = f"{_INVEZGO_BASE}/analysis/top/foreign"
    try:
        resp = _req.get(
            url,
            params={"date": date_str},
            headers={"Authorization": f"Bearer {config.INVEZGO_API_KEY}"},
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        logger.warning("fetch_foreign_flow_market failed for %s: %s", date_str, e)
        return None

    # Unwrap envelope — could be list, or dict with buy/sell/accum/dist keys
    rows: list = []
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        # Try common envelope keys; merge buy and sell sides
        rows = (
            (raw.get("buy") or raw.get("accum") or []) +
            (raw.get("sell") or raw.get("dist") or [])
        )
        if not rows:
            # Flat dict with aggregate totals
            buy  = float(raw.get("foreignBuy")  or raw.get("buyVal")  or raw.get("buy_val")  or 0)
            sell = float(raw.get("foreignSell") or raw.get("sellVal") or raw.get("sell_val") or 0)
            net  = float(raw.get("foreignNet")  or raw.get("netVal")  or raw.get("net_val")  or (buy - sell))
            logger.info("Foreign market flow %s: net %.1fB IDR", date_str, net / 1e9)
            return {
                "date":         date_str,
                "buy_val_idr":  buy,
                "sell_val_idr": sell,
                "net_val_idr":  net,
                "direction":    "BUY" if net > 0 else ("SELL" if net < 0 else "NEUTRAL"),
            }

    if not rows:
        logger.debug("fetch_foreign_flow_market: empty response for %s", date_str)
        return None

    # Sum across all rows
    total_buy = total_sell = 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        buy  = float(row.get("foreignBuy")  or row.get("buyVal")  or row.get("buy_val")  or 0)
        sell = float(row.get("foreignSell") or row.get("sellVal") or row.get("sell_val") or 0)
        net  = float(row.get("foreignNet")  or row.get("netVal")  or row.get("net_val")  or 0)
        if net != 0:
            if net > 0:
                total_buy += net
            else:
                total_sell += abs(net)
        else:
            total_buy  += buy
            total_sell += sell

    net = total_buy - total_sell
    logger.info("Foreign market flow %s: buy %.1fB sell %.1fB net %.1fB IDR",
                date_str, total_buy / 1e9, total_sell / 1e9, net / 1e9)
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
