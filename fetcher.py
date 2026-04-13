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


def fetch_candles(symbol: str, interval: str = "5m", period: str = "1d") -> pd.DataFrame | None:
    """
    Fetch OHLCV candles for a .JK stock.
    Returns DataFrame with lowercase columns: open, high, low, close, volume.
    symbol:   e.g. "BBCA.JK"
    interval: "5m" for live, "1d" for Prophet training
    period:   "1d" for intraday, "1y" for Prophet
    """
    retries = 3
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
            if df is None or df.empty:
                if attempt < retries:
                    logger.warning(f"No candle data for {symbol} (attempt {attempt}/{retries}), retrying...")
                    time.sleep(2 ** attempt)  # 2s, 4s
                    continue
                logger.warning(f"No candle data for {symbol} after {retries} attempts")
                return None
            break  # success
        except Exception as e:
            if attempt < retries:
                logger.warning(f"fetch_candles({symbol}) attempt {attempt}/{retries} failed: {e}, retrying...")
                time.sleep(2 ** attempt)
                continue
            logger.error(f"fetch_candles({symbol}) failed after {retries} attempts: {e}")
            return None

        # Flatten MultiIndex columns if present (yfinance ≥ 0.2.x)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower() for c in df.columns]
        df.index.name = "datetime"
        return df
    except Exception as e:
        logger.error(f"fetch_candles({symbol}): {e}")
        return None


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
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.idx.co.id/",
        }
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        foreign_buy  = data.get("foreignBuyVal", 0) or 0
        foreign_sell = data.get("foreignSellVal", 0) or 0
        net          = data.get("foreignNetVal", foreign_buy - foreign_sell)

        # Calculate consecutive days (last 5 trading days)
        days_consecutive = _calc_consecutive_days(stock_code)

        return {
            "net_foreign_buy_idr":  net,
            "foreign_buy_vol":      data.get("foreignBuyVol", 0),
            "foreign_sell_vol":     data.get("foreignSellVol", 0),
            "days_consecutive":     days_consecutive,
        }
    except Exception as e:
        logger.error(f"fetch_foreign_flow({symbol}): {e}")
        return None


def _calc_consecutive_days(stock_code: str, lookback: int = 5) -> int:
    """
    Check last `lookback` trading days and return how many consecutive
    days foreigners have been net buyers (positive) or net sellers (negative).
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.idx.co.id/",
    }
    net_vals = []
    for i in range(1, lookback + 1):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            url = "https://www.idx.co.id/umbraco/Surface/TradingSummary/GetBrokerSummary"
            resp = requests.get(
                url,
                params={"stockCode": stock_code, "tradingDate": date},
                headers=headers,
                timeout=8,
            )
            data = resp.json()
            net_vals.append(data.get("foreignNetVal", 0) or 0)
        except Exception:
            break

    if not net_vals:
        return 0

    direction = 1 if net_vals[0] >= 0 else -1
    count = 0
    for val in net_vals:
        if (val >= 0 and direction == 1) or (val < 0 and direction == -1):
            count += 1
        else:
            break
    return count * direction


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
            headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.idx.co.id/"}
            resp = requests.get(url, headers=headers, timeout=10)
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
