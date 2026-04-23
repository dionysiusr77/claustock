from __future__ import annotations
import time
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Silence yfinance's own JSONDecodeError noise — we handle failures ourselves
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# ── Persistent HTTP session for all IDX API calls ─────────────────────────────
# Establishes a browser-like cookie context to avoid 403s on Railway (GCP IPs).
_IDX_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
    "Referer":         "https://www.idx.co.id/",
    "Origin":          "https://www.idx.co.id",
}

SESSION = requests.Session()
SESSION.headers.update(_IDX_HEADERS)

def _ensure_idx_cookie():
    """Establish a session cookie with idx.co.id so API calls are not rejected."""
    try:
        SESSION.get("https://www.idx.co.id/", timeout=10)
    except Exception as e:
        logger.warning(f"_ensure_idx_cookie: {e}")

# Establish cookie at import time (non-fatal if it fails)
try:
    _ensure_idx_cookie()
except Exception:
    pass

# ── In-memory cache for consecutive-days calculation ──────────────────────────
# { stock_code: (result: int, expires: datetime) }
_flow_cache: dict[str, tuple[int, datetime]] = {}
_FLOW_CACHE_TTL_HOURS = 1


def fetch_candles(symbol: str, interval: str = "5m", period: str = "1d") -> pd.DataFrame | None:
    """
    Fetch OHLCV candles for a .JK stock.
    Returns DataFrame with lowercase columns: open, high, low, close, volume.
    symbol:   e.g. "BBCA.JK"
    interval: "5m" for live, "1d" for Prophet training
    period:   "1d" for intraday, "1y" for Prophet
    """
    retries = 3
    df = None
    for attempt in range(1, retries + 1):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
            if df is not None and not df.empty:
                break  # success
            if attempt < retries:
                logger.warning(f"No candle data for {symbol} (attempt {attempt}/{retries}), retrying...")
                time.sleep(2 ** attempt)
            else:
                logger.warning(f"No candle data for {symbol} after {retries} attempts")
                return None
        except Exception as e:
            if attempt < retries:
                logger.warning(f"fetch_candles({symbol}) attempt {attempt}/{retries} failed: {e}, retrying...")
                time.sleep(2 ** attempt)
            else:
                logger.error(f"fetch_candles({symbol}) failed after {retries} attempts: {e}")
                return None

    if df is None or df.empty:
        return None

    df.columns = [c.lower() for c in df.columns]
    df.index.name = "datetime"
    return df


def fetch_daily_candles(symbol: str, period: str = "30d") -> pd.DataFrame | None:
    """
    Fetch daily OHLCV candles for D-1 screening.
    period="30d" is the default; pass "60d" for enough history for MACD(26).
    Most recent row = yesterday's finalized close (D-1).
    """
    return fetch_candles(symbol, interval="1d", period=period)


def fetch_quote(symbol: str) -> dict | None:
    """
    Fetch live quote for a single stock.
    Returns: { price, change_pct, high, low, volume, market_cap }
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info

        price = getattr(info, "last_price", None)
        prev_close = getattr(info, "previous_close", None)
        change_pct = ((price - prev_close) / prev_close * 100) if price and prev_close else 0.0

        return {
            "price":      round(price, 2) if price else None,
            "change_pct": round(change_pct, 2),
            "high":       getattr(info, "day_high", None),
            "low":        getattr(info, "day_low", None),
            "volume":     getattr(info, "three_month_average_volume", None),
            "market_cap": getattr(info, "market_cap", None),
        }
    except Exception as e:
        logger.error(f"fetch_quote({symbol}): {e}")
        return None


def fetch_foreign_flow(symbol: str) -> dict | None:
    """
    Fetch today's foreign net buy/sell for a stock from IDX.
    Returns: { net_foreign_buy_idr, foreign_buy_vol, foreign_sell_vol, days_consecutive }
    days_consecutive: positive = consecutive net-buy days, negative = net-sell days
    """
    # Strip .JK suffix for IDX API
    stock_code = symbol.replace(".JK", "")
    today = datetime.now().strftime("%Y-%m-%d")

    try:
        url = "https://www.idx.co.id/umbraco/Surface/TradingSummary/GetBrokerSummary"
        params = {"stockCode": stock_code, "tradingDate": today}
        resp = SESSION.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        foreign_buy  = data.get("foreignBuyVal", 0) or 0
        foreign_sell = data.get("foreignSellVal", 0) or 0
        net          = data.get("foreignNetVal", foreign_buy - foreign_sell)

        # Calculate consecutive days (last 5 trading days)
        days_consecutive = _calc_consecutive_days(stock_code)

        score, reasons = _score_foreign_flow(net, days_consecutive)

        return {
            "net_foreign_buy_idr":  net,
            "foreign_buy_vol":      data.get("foreignBuyVol", 0),
            "foreign_sell_vol":     data.get("foreignSellVol", 0),
            "days_consecutive":     days_consecutive,
            "foreign_score":        score,
            "reasons":              reasons,
        }
    except Exception as e:
        logger.error(f"fetch_foreign_flow({symbol}): {e}")
        return None


def _score_foreign_flow(net_idr: float, days_consecutive: int) -> tuple[int, list[str]]:
    """
    Score foreign flow 0–20 pts.
    days_consecutive >= +3  → 20 pts (strong bullish)
    days_consecutive == +2  → 15 pts
    days_consecutive == +1  → 10 pts
    net == 0 / no data      →  5 pts
    days_consecutive negative → 0 pts
    """
    reasons = []
    net_b = round(net_idr / 1_000_000_000, 2) if net_idr else 0  # in billions IDR

    if days_consecutive >= 3:
        score = 20
        reasons.append(f"Foreign net BUY {days_consecutive}d in a row (+{net_b:.1f}B IDR)")
    elif days_consecutive == 2:
        score = 15
        reasons.append(f"Foreign net BUY 2d in a row (+{net_b:.1f}B IDR)")
    elif days_consecutive == 1:
        score = 10
        reasons.append(f"Foreign net BUY today (+{net_b:.1f}B IDR)")
    elif days_consecutive == 0:
        score = 5
        reasons.append("Foreign flow neutral")
    else:
        score = 0
        reasons.append(f"Foreign net SELL {abs(days_consecutive)}d in a row ({net_b:.1f}B IDR)")

    return score, reasons


def _calc_consecutive_days(stock_code: str, lookback: int = 5) -> int:
    """
    Check last `lookback` trading days and return how many consecutive
    days foreigners have been net buyers (positive) or net sellers (negative).
    Results are cached for 1 hour to avoid 5 HTTP calls per stock per scan.
    """
    # Check cache
    cached = _flow_cache.get(stock_code)
    if cached is not None:
        result, expires = cached
        if datetime.now() < expires:
            return result

    net_vals = []
    for i in range(1, lookback + 1):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            url = "https://www.idx.co.id/umbraco/Surface/TradingSummary/GetBrokerSummary"
            resp = SESSION.get(
                url,
                params={"stockCode": stock_code, "tradingDate": date},
                timeout=8,
            )
            data = resp.json()
            net_vals.append(data.get("foreignNetVal", 0) or 0)
        except Exception:
            break

    if not net_vals:
        result = 0
    else:
        direction = 1 if net_vals[0] >= 0 else -1
        count = 0
        for val in net_vals:
            if (val >= 0 and direction == 1) or (val < 0 and direction == -1):
                count += 1
            else:
                break
        result = count * direction

    # Store in cache with 1-hour TTL
    _flow_cache[stock_code] = (result, datetime.now() + timedelta(hours=_FLOW_CACHE_TTL_HOURS))
    return result


def fetch_jci_summary() -> dict | None:
    """
    Fetch JCI (Composite Index ^JKSE) daily summary.
    Returns: { jci_close, jci_change_pct, total_foreign_net_idr, market_status }
    """
    try:
        jci = yf.Ticker("^JKSE")
        info = jci.fast_info

        price     = getattr(info, "last_price", None)
        prev      = getattr(info, "previous_close", None)
        chg_pct   = ((price - prev) / prev * 100) if price and prev else 0.0

        # Market-wide foreign flow from IDX
        foreign_net = None
        try:
            url = "https://www.idx.co.id/umbraco/Surface/TradingSummary/GetForeignFlow"
            resp = SESSION.get(url, timeout=10)
            fdata = resp.json()
            # IDX returns list; take the most recent entry
            if isinstance(fdata, list) and fdata:
                foreign_net = fdata[0].get("foreignNetVal", None)
            elif isinstance(fdata, dict):
                foreign_net = fdata.get("foreignNetVal", None)
        except Exception:
            pass

        return {
            "jci_close":           round(price, 2) if price else None,
            "jci_change_pct":      round(chg_pct, 2),
            "total_foreign_net_idr": foreign_net,
            "market_status":       "OPEN" if price else "CLOSED",
        }
    except Exception as e:
        logger.error(f"fetch_jci_summary: {e}")
        return None
