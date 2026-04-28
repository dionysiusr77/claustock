"""
Market data fetcher.
  - Daily OHLCV via yfinance (batch + single)
  - IHSG + sectoral indices
  - IDX foreign net flow per stock
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

import config

logger = logging.getLogger(__name__)

# ── Sectoral index tickers ────────────────────────────────────────────────────
SECTOR_TICKERS = {
    "IHSG":        "^JKSE",
    "LQ45":        "^JKLQ45",
    "Finance":     "^JKFIN",
    "Consumer":    "^JKCONS",
    "Mining":      "^JKMING",
    "Infra":       "^JKINFA",
    "Property":    "^JKPROP",
    "Manufacture": "^JKMNFG",
    "Trade":       "^JKTRAD",
    "Agri":        "^JKAGRI",
    "Misc Ind":    "^JKMISC",
    "Basic Ind":   "^JKBIND",
}

# ── Daily OHLCV ───────────────────────────────────────────────────────────────

def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, drop all-NaN rows, ensure standard OHLCV names."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df = df.dropna(how="all")
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    return df[["open", "high", "low", "close", "volume"]]


def fetch_daily_batch(symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """
    Batch download daily OHLCV for multiple symbols.
    Returns {symbol: DataFrame} — missing or broken symbols are silently skipped.
    """
    if not symbols:
        return {}

    try:
        raw = yf.download(
            tickers=symbols,
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        logger.error("yfinance batch download failed: %s", e)
        return {}

    result: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        try:
            if len(symbols) == 1:
                df = raw.copy()
            else:
                df = raw[sym].copy()
            df = _normalise_df(df)
            if len(df) >= 60:          # need at least 60 days for indicators
                result[sym] = df
        except Exception:
            continue

    logger.info("batch download: %d/%d symbols OK", len(result), len(symbols))
    return result


def fetch_daily(symbol: str, period: str = "1y") -> pd.DataFrame | None:
    """Single-symbol daily OHLCV. Returns None on failure."""
    result = fetch_daily_batch([symbol], period=period)
    return result.get(symbol)


# ── IHSG + sector indices ─────────────────────────────────────────────────────

def fetch_market_breadth() -> dict:
    """
    Fetch yesterday's IHSG close, change%, and all sectoral index changes.
    Returns dict ready for the briefing header.
    """
    tickers = list(SECTOR_TICKERS.values())
    try:
        raw = yf.download(
            tickers=tickers,
            period="5d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        logger.error("sector index download failed: %s", e)
        return {}

    out: dict = {}
    for name, ticker in SECTOR_TICKERS.items():
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[ticker].dropna(how="all")
            df.columns = [c.lower() for c in df.columns]
            df = df.dropna(subset=["close"])
            if len(df) < 2:
                continue
            prev_close = df["close"].iloc[-2]
            last_close = df["close"].iloc[-1]
            change_pct = (last_close - prev_close) / prev_close * 100
            out[name] = {
                "close":      round(last_close, 2),
                "change_pct": round(change_pct, 2),
                "direction":  "UP" if change_pct > 0 else ("DOWN" if change_pct < 0 else "FLAT"),
            }
        except Exception:
            continue

    return out


# ── IDX foreign flow ──────────────────────────────────────────────────────────

_IDX_BROKER_URL = (
    "https://www.idx.co.id/umbraco/Surface/TradingSummary/GetBrokerSummary"
)
_IDX_FOREIGN_URL = (
    "https://www.idx.co.id/umbraco/Surface/TradingSummary/GetForeignFlow"
)

_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer":    "https://www.idx.co.id/",
}


def _trading_date_str(offset_days: int = 0) -> str:
    """Return trading date string YYYY-MM-DD, skipping weekends."""
    dt = datetime.now()
    dt -= timedelta(days=offset_days)
    while dt.weekday() >= 5:          # skip Sat/Sun
        dt -= timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


def fetch_foreign_flow_stock(symbol: str, date_str: str | None = None) -> dict | None:
    """
    Fetch per-stock foreign net buy/sell from IDX broker summary.
    symbol: e.g. "BBCA.JK" or "BBCA"
    Returns: {net_val_idr, buy_val_idr, sell_val_idr, net_lot, date} or None.
    """
    code = symbol.replace(".JK", "").upper()
    date_str = date_str or _trading_date_str(1)   # default: yesterday

    params = {"stockCode": code, "tradingDate": date_str}
    try:
        resp = requests.get(
            _IDX_BROKER_URL, params=params, headers=_HEADERS, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        # Find foreign broker rows (code starts with "DB" / "YP" / etc.)
        # IDX marks foreign with foreignBuyVal / foreignSellVal at top level
        foreign_buy  = float(data.get("foreignBuyVal",  0) or 0)
        foreign_sell = float(data.get("foreignSellVal", 0) or 0)
        net = foreign_buy - foreign_sell

        return {
            "date":         date_str,
            "buy_val_idr":  foreign_buy,
            "sell_val_idr": foreign_sell,
            "net_val_idr":  net,
            "direction":    "BUY" if net > 0 else ("SELL" if net < 0 else "NEUTRAL"),
        }
    except Exception as e:
        logger.debug("foreign flow fetch failed for %s: %s", symbol, e)
        return None


def fetch_foreign_flow_market(date_str: str | None = None) -> dict | None:
    """
    Fetch aggregate market-wide foreign net flow from IDX.
    Returns: {net_val_idr, buy_val_idr, sell_val_idr, direction, date} or None.
    """
    date_str = date_str or _trading_date_str(1)
    try:
        resp = requests.get(
            _IDX_FOREIGN_URL,
            params={"date": date_str},
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        buy  = float(data.get("foreignBuyVal",  0) or 0)
        sell = float(data.get("foreignSellVal", 0) or 0)
        net  = buy - sell

        return {
            "date":         date_str,
            "buy_val_idr":  buy,
            "sell_val_idr": sell,
            "net_val_idr":  net,
            "direction":    "BUY" if net > 0 else ("SELL" if net < 0 else "NEUTRAL"),
        }
    except Exception as e:
        logger.debug("market foreign flow fetch failed: %s", e)
        return None


# ── Quick liquidity pre-filter ────────────────────────────────────────────────

def filter_liquid(
    symbols: list[str],
    min_price: float = config.MIN_PRICE,
    max_price: float = config.MAX_PRICE,
    min_avg_val: float = config.MIN_AVG_DAILY_VAL,
    lookback_days: int = 20,
) -> list[str]:
    """
    Fast liquidity filter before full indicator calculation.
    Downloads only 30d of data to check avg daily value and price range.
    Returns symbols that pass all gates.
    """
    data = fetch_daily_batch(symbols, period="1mo")
    liquid: list[str] = []

    for sym, df in data.items():
        if df.empty or len(df) < 5:
            continue
        last_price = df["close"].iloc[-1]
        if not (min_price <= last_price <= max_price):
            continue
        recent = df.tail(lookback_days)
        avg_val = (recent["close"] * recent["volume"]).mean()
        if avg_val >= min_avg_val:
            liquid.append(sym)

    logger.info(
        "liquidity filter: %d/%d symbols passed", len(liquid), len(symbols)
    )
    return liquid
