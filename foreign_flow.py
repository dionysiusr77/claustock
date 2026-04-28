"""
Multi-day foreign flow tracker.
Wraps fetcher.fetch_foreign_flow_stock() to compute consecutive buy/sell streaks.

The scorer only receives the enriched dict — it never needs to know how many
HTTP calls were made to build it.
"""

import logging
from datetime import datetime, timedelta

from fetcher import fetch_foreign_flow_stock

logger = logging.getLogger(__name__)


def _last_n_trading_dates(n: int) -> list[str]:
    """Return last n weekdays as YYYY-MM-DD strings, most recent first."""
    dates: list[str] = []
    dt = datetime.now()
    while len(dates) < n:
        dt -= timedelta(days=1)
        if dt.weekday() < 5:   # Mon–Fri only
            dates.append(dt.strftime("%Y-%m-%d"))
    return dates


def get_foreign_flow_enriched(symbol: str, lookback_days: int = 5) -> dict | None:
    """
    Fetch per-stock foreign flow for the last `lookback_days` trading days.
    Returns the most recent day's data dict enriched with:
      consecutive_buy_days  : int  — how many days in a row asing has been net buying
      consecutive_sell_days : int  — how many days in a row asing has been net selling
      history               : list — raw flow per day (newest first)

    Returns None if no data is available at all.
    """
    dates = _last_n_trading_dates(lookback_days)
    history: list[dict] = []

    for date in dates:
        ff = fetch_foreign_flow_stock(symbol, date)
        if ff:
            history.append(ff)

    if not history:
        logger.debug("No foreign flow data for %s over last %d trading days", symbol, lookback_days)
        return None

    result = history[0].copy()
    direction_today = result["direction"]

    consec_buy  = 0
    consec_sell = 0

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


def summarise_foreign_batch(symbols: list[str], lookback_days: int = 3) -> dict[str, dict | None]:
    """
    Run get_foreign_flow_enriched for a list of symbols.
    Returns {symbol: enriched_dict_or_None}.
    Callers that need concurrency should use fetcher._fetch_ff_batch directly.
    """
    return {sym: get_foreign_flow_enriched(sym, lookback_days) for sym in symbols}
