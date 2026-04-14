"""
IDX Stock Bot — Phase 1 orchestrator
Fetches live OHLCV data, calculates indicators, saves snapshots to Firestore.
Run: python main.py
"""

import logging
import sys
from datetime import datetime

import pytz

import config
from fetcher import fetch_candles, fetch_quote, fetch_jci_summary
from indicators import calculate_indicators
import firestore_client as db
from scheduler import build_scheduler, is_market_open, is_trading_day, WIB

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("main")


# ── Core scan ─────────────────────────────────────────────────────────────────

def scan_stocks():
    """Fetch + score all watchlist stocks, save snapshots to Firestore."""
    now_wib = datetime.now(WIB).strftime("%H:%M WIB")
    logger.info(f"── Scan @ {now_wib} ──────────────────────────")

    for symbol in config.STOCKS:
        try:
            # 1. Fetch 5-min candles
            df = fetch_candles(symbol, interval=config.CANDLE_INTERVAL, period="5d")
            if df is None:
                logger.warning(f"{symbol}: no candle data, skipping")
                continue

            # 2. Calculate technical indicators
            tech = calculate_indicators(df)
            if tech is None:
                logger.warning(f"{symbol}: indicator calc failed, skipping")
                continue

            # 3. Print to console
            verdict_tag = _score_tag(tech["score"])
            logger.info(
                f"{symbol:10s} | price={tech['price']:>8.2f}"
                f" | RSI={tech['rsi']:>5.1f}"
                f" | MA={tech['ma_trend']:>4s}"
                f" | vol={tech['volume_ratio']:>4.1f}x"
                f" | score={tech['score']:>2d}/35"
                f" | {verdict_tag}"
            )

            # 4. Save snapshot (Phase 1: only technical layer)
            snapshot = {
                "price":          tech["price"],
                "technical_score": tech["score"],
                "total_score":     tech["score"],  # single layer for now
                "rsi":            tech["rsi"],
                "ma7":            tech["ma7"],
                "ma30":           tech["ma30"],
                "ma_trend":       tech["ma_trend"],
                "volume_ratio":   tech["volume_ratio"],
                "candle_pattern": tech["candle_pattern"],
                "reasons":        tech["reasons"],
                "phase":          1,
            }
            db.save_snapshot(symbol, snapshot)

        except Exception as e:
            logger.error(f"{symbol}: unexpected error — {e}")


def _score_tag(score: int) -> str:
    if score >= 28:
        return "STRONG"
    elif score >= 21:
        return "GOOD"
    elif score >= 14:
        return "WATCH"
    else:
        return "SKIP"


# ── Scheduler callbacks (stubs for Phase 1) ───────────────────────────────────

def presession1():
    logger.info("Pre-session 1 briefing triggered (Phase 2+)")


def presession2():
    logger.info("Pre-session 2 briefing triggered (Phase 2+)")


def eod_summary():
    logger.info("End-of-day summary triggered (Phase 2+)")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    now = datetime.now(WIB)
    logger.info("IDX Stock Bot — Phase 1 starting")
    logger.info(f"Time: {now.strftime('%A %d %b %Y %H:%M WIB')}")
    logger.info(f"Trading day: {is_trading_day()}")
    logger.info(f"Market open: {is_market_open()}")
    logger.info(f"Watchlist: {', '.join(config.STOCKS)}")

    # Print JCI overview
    jci = fetch_jci_summary()
    if jci:
        logger.info(
            f"JCI: {jci['jci_close']} ({jci['jci_change_pct']:+.2f}%)"
            f" | Foreign net: {jci['total_foreign_net_idr']}"
        )

    # Run an immediate scan so we see output right away
    scan_stocks()

    # Start scheduler (blocks here)
    logger.info(f"Starting scheduler — scan every {config.SCAN_INTERVAL_SEC}s during market hours")
    scheduler = build_scheduler(
        scan_fn=scan_stocks,
        presession1_fn=presession1,
        presession2_fn=presession2,
        eod_fn=eod_summary,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
