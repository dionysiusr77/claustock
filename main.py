"""
IDX Stock Bot — Phase 2 orchestrator
Adds Prophet 5-day forecast + IDX foreign flow to each snapshot.
Run: python main.py
"""

import logging
import sys
from datetime import datetime

import pytz

import config
from fetcher import fetch_candles, fetch_quote, fetch_jci_summary, fetch_foreign_flow
from indicators import calculate_indicators
from forecaster import forecast_5d, warmup_models
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

            # 2. Technical indicators (Layer 1)
            tech = calculate_indicators(df)
            if tech is None:
                logger.warning(f"{symbol}: indicator calc failed, skipping")
                continue

            # 3. Prophet forecast (Layer 2) — cached, fast after warmup
            forecast = forecast_5d(symbol)

            # 4. Foreign flow (Layer 3)
            flow = fetch_foreign_flow(symbol)

            # 5. Combine scores
            tech_score    = tech["score"]                                     # 0–35
            prophet_score = forecast["prophet_score"] if forecast else 0      # 0–25
            foreign_score = flow["foreign_score"]     if flow     else 0      # 0–20
            # news_score  = 0  — Phase 3
            total_score   = tech_score + prophet_score + foreign_score

            verdict = _verdict(total_score)

            # 6. Log
            logger.info(
                f"{symbol:10s} | {tech['price']:>8.2f}"
                f" | T={tech_score:>2d} P={prophet_score:>2d} F={foreign_score:>2d}"
                f" | total={total_score:>2d}/80"
                f" | {verdict}"
                + (f" | Prophet {forecast['trend_pct']:+.1f}% {forecast['trend']}" if forecast else "")
                + (f" | flow {flow['days_consecutive']:+d}d" if flow else "")
            )

            # 7. Save snapshot
            snapshot = {
                "price":           tech["price"],
                "technical_score": tech_score,
                "prophet_score":   prophet_score,
                "foreign_score":   foreign_score,
                "news_score":      0,
                "total_score":     total_score,
                "verdict":         verdict,
                # Technical
                "rsi":             tech["rsi"],
                "ma7":             tech["ma7"],
                "ma30":            tech["ma30"],
                "ma_trend":        tech["ma_trend"],
                "volume_ratio":    tech["volume_ratio"],
                "candle_pattern":  tech["candle_pattern"],
                # Prophet
                "forecast_5d":     forecast["forecast_5d"]  if forecast else None,
                "trend_pct":       forecast["trend_pct"]    if forecast else None,
                "trend":           forecast["trend"]        if forecast else None,
                # Foreign flow
                "net_foreign_buy_idr": flow["net_foreign_buy_idr"] if flow else None,
                "days_consecutive":    flow["days_consecutive"]     if flow else None,
                # Meta
                "reasons": (
                    tech["reasons"]
                    + (forecast["reasons"] if forecast else [])
                    + (flow["reasons"]    if flow     else [])
                ),
                "phase": 2,
            }
            db.save_snapshot(symbol, snapshot)

        except Exception as e:
            logger.error(f"{symbol}: unexpected error — {e}")


def _verdict(score: int) -> str:
    """Phase 2 max is 80/100 (no news layer yet)."""
    if score >= 64:    # ~80% of 80
        return "STRONG BUY"
    elif score >= 48:  # ~60% of 80
        return "BUY"
    elif score >= 32:  # ~40% of 80
        return "WATCH"
    else:
        return "SKIP"


# ── Scheduler callbacks ───────────────────────────────────────────────────────

def presession1():
    logger.info("Pre-session 1 briefing triggered (Phase 4)")


def presession2():
    logger.info("Pre-session 2 briefing triggered (Phase 4)")


def eod_summary():
    logger.info("End-of-day summary triggered (Phase 4)")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    now = datetime.now(WIB)
    logger.info("IDX Stock Bot — Phase 2 starting")
    logger.info(f"Time: {now.strftime('%A %d %b %Y %H:%M WIB')}")
    logger.info(f"Trading day: {is_trading_day()} | Market open: {is_market_open()}")
    logger.info(f"Watchlist: {', '.join(config.STOCKS)}")

    # JCI overview
    jci = fetch_jci_summary()
    if jci:
        logger.info(
            f"JCI: {jci['jci_close']} ({jci['jci_change_pct']:+.2f}%)"
            f" | Foreign net: {jci['total_foreign_net_idr']}"
        )

    # Train Prophet models for all stocks (slow on first run ~2–3 min)
    warmup_models(config.STOCKS)

    # Immediate scan
    scan_stocks()

    # Start scheduler
    logger.info(f"Scheduler started — scan every {config.SCAN_INTERVAL_SEC}s during market hours")
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
