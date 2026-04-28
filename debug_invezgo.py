"""
Quick diagnostic script — run with: python debug_invezgo.py
Tests Invezgo API connectivity and response shape on a handful of stocks.
"""

import json
import logging
import sys

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)

TEST_STOCKS = ["BBCA", "BBRI", "TLKM", "ASII", "BMRI"]

def main():
    from invezgo_client import _get_client, _code, _date_range, _parse_ohlcv

    # 1. Test client init
    try:
        client = _get_client()
        logger.info("Client OK: %s", type(client).__name__)
    except Exception as e:
        logger.error("Client init failed: %s", e)
        sys.exit(1)

    # 2. Inspect available analysis methods
    methods = [m for m in dir(client.analysis) if not m.startswith("_")]
    logger.info("analysis methods: %s", methods)

    # 3. Test get_chart on first stock
    sym = TEST_STOCKS[0]
    from_date, to_date = _date_range(30)
    logger.info("Fetching get_chart(%s, %s → %s)", sym, from_date, to_date)

    try:
        raw = client.analysis.get_chart(code=sym, from_date=from_date, to_date=to_date)
        logger.info("raw type : %s", type(raw).__name__)
        logger.info("raw value: %s", str(raw)[:1000])

        # Try parsing
        df = _parse_ohlcv(raw)
        logger.info("parsed df shape: %s", df.shape)
        if not df.empty:
            logger.info("parsed df head:\n%s", df.head(3))
        else:
            logger.warning("_parse_ohlcv returned empty — response format not recognised")

    except Exception as e:
        logger.error("get_chart failed: %s: %s", type(e).__name__, e)

        # Try alternate method names if get_chart doesn't exist
        for alt in ["get_stock_chart", "chart", "get_price", "get_history", "get_candles"]:
            if hasattr(client.analysis, alt):
                logger.info("Found alternate method: analysis.%s", alt)

    # 4. Quick batch test on 5 stocks
    logger.info("\n--- Batch test: %s ---", TEST_STOCKS)
    from invezgo_client import fetch_daily_batch
    results = fetch_daily_batch(TEST_STOCKS, days=365)
    logger.info("Batch result: %d/%d OK — symbols: %s",
                len(results), len(TEST_STOCKS), list(results.keys()))

    logger.info("\n--- IHSG test ---")
    from invezgo_client import _fetch_ihsg_invezgo
    ihsg = _fetch_ihsg_invezgo()
    logger.info("IHSG result: %s", ihsg)


if __name__ == "__main__":
    main()
