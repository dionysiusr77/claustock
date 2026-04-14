"""
IDX Stock Bot — Phase 4
Full pipeline: 4-layer scoring → Telegram signals + briefings + commands.
Run: python main.py
"""

import logging
import sys
from datetime import datetime

import config
from fetcher import fetch_candles, fetch_jci_summary, fetch_foreign_flow
from indicators import calculate_indicators
from forecaster import forecast_5d, warmup_models
from news_fetcher import score_news_sentiment
from scorer import score_stock, should_signal
import firestore_client as db
from scheduler import build_scheduler, is_market_open, is_trading_day, WIB
import telegram_bot as tg

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("main")

# Latest scores kept in memory for Telegram commands
_latest_scores: list[dict] = []


# ── Core scan ─────────────────────────────────────────────────────────────────

def scan_stocks():
    global _latest_scores
    now_wib = datetime.now(WIB).strftime("%H:%M WIB")
    logger.info(f"── Scan @ {now_wib} ──────────────────────────")

    scores = []
    for symbol in config.STOCKS:
        try:
            df = fetch_candles(symbol, interval=config.CANDLE_INTERVAL, period="5d")
            if df is None:
                logger.warning(f"{symbol}: no candle data, skipping")
                continue

            tech     = calculate_indicators(df)
            if tech is None:
                logger.warning(f"{symbol}: indicator calc failed, skipping")
                continue

            forecast = forecast_5d(symbol)
            flow     = fetch_foreign_flow(symbol)
            news     = score_news_sentiment(symbol)

            result   = score_stock(symbol, tech, forecast, flow, news)
            result["symbol"] = symbol
            scores.append(result)

            s = result["scores"]
            logger.info(
                f"{symbol:10s} | {result['price']:>8.2f}"
                f" | T={s['technical']:>2d} P={s['prophet']:>2d}"
                f" F={s['foreign']:>2d} N={s['news']:>2d}"
                f" | total={result['total_score']:>3d}/100 | {result['verdict']}"
            )

            # Save snapshot
            db.save_snapshot(symbol, {
                "price":               result["price"],
                "technical_score":     s["technical"],
                "prophet_score":       s["prophet"],
                "foreign_score":       s["foreign"],
                "news_score":          s["news"],
                "total_score":         result["total_score"],
                "verdict":             result["verdict"],
                "rsi":                 result["rsi"],
                "ma_trend":            result["ma_trend"],
                "volume_ratio":        result["volume_ratio"],
                "candle_pattern":      result["candle_pattern"],
                "forecast_5d":         result["forecast_5d"],
                "trend_pct":           result["trend_pct"],
                "trend":               result["trend"],
                "net_foreign_buy_idr": result["net_foreign_buy_idr"],
                "days_consecutive":    result["days_consecutive"],
                "news_sentiment":      result["news_sentiment"],
                "news_headline":       result["news_headline"],
                "reasons":             result["reasons"],
                "phase":               4,
            })

            # Fire Telegram signal if score qualifies
            if should_signal(result):
                logger.info(f"{symbol}: signal fired (score={result['total_score']})")
                msg = tg.format_signal(symbol, {**result, **result["scores"],
                                                "technical_score": s["technical"],
                                                "prophet_score":   s["prophet"],
                                                "foreign_score":   s["foreign"],
                                                "news_score":      s["news"]})
                tg.send_message(msg)
                db.save_signal(symbol, result)

        except Exception as e:
            logger.error(f"{symbol}: unexpected error — {e}")

    _latest_scores = scores


# ── Scheduled jobs ────────────────────────────────────────────────────────────

def presession1():
    logger.info("Running pre-session 1 briefing")
    _send_briefing(session=1)


def presession2():
    logger.info("Running pre-session 2 briefing")
    _send_briefing(session=2)


def _send_briefing(session: int):
    jci    = fetch_jci_summary()
    scores = _latest_scores or []
    msg    = tg.format_presession_briefing(
        session=session,
        date_str=datetime.now(WIB).strftime("%a %d %b %Y"),
        jci=jci,
        stock_scores=scores,
    )
    tg.send_message(msg)


def eod_summary():
    logger.info("Running end-of-day summary")
    signals = db.get_today_signals()
    msg     = tg.format_eod_summary(_latest_scores, signals)
    tg.send_message(msg)


# ── Command handlers ──────────────────────────────────────────────────────────

def cmd_status(_args):
    msg = tg.format_status(is_market_open(), is_trading_day(), len(config.STOCKS))
    tg.send_message(msg)


def cmd_stocks(_args):
    if not _latest_scores:
        tg.send_message("No scores yet — waiting for first scan.")
        return
    msg = tg.format_stocks_list(_latest_scores)
    tg.send_message(msg)


def cmd_briefing(_args):
    now = datetime.now(WIB)
    session = 1 if now.hour < 12 else 2
    _send_briefing(session)


def cmd_forecast(_args):
    if not _latest_scores:
        tg.send_message("No forecast data yet.")
        return
    lines = ["📈 <b>5-Day Forecasts</b>", ""]
    for s in _latest_scores:
        ticker  = s["symbol"].replace(".JK", "")
        pct     = s.get("trend_pct")
        trend   = s.get("trend", "—")
        f5d     = s.get("forecast_5d")
        price   = s.get("price", 0)
        t_emoji = tg.TREND_EMOJI.get(trend, "⚪")
        if pct is not None:
            lines.append(f"{t_emoji} <b>{ticker}</b>  {price:,.0f} → {f5d:,.0f}  ({pct:+.1f}%)")
        else:
            lines.append(f"⚪ <b>{ticker}</b>  No forecast")
    tg.send_message("\n".join(lines))


def cmd_flow(_args):
    lines = ["🌊 <b>Foreign Flow Today</b>", ""]
    for s in _latest_scores:
        ticker = s["symbol"].replace(".JK", "")
        days   = s.get("days_consecutive", 0) or 0
        net    = s.get("net_foreign_buy_idr", 0) or 0
        net_b  = net / 1_000_000_000
        if days > 0:
            emoji = "🟢"
        elif days < 0:
            emoji = "🔴"
        else:
            emoji = "⚪"
        lines.append(f"{emoji} <b>{ticker}</b>  {net_b:+.2f}B IDR  ({days:+d}d)")
    tg.send_message("\n".join(lines) if len(lines) > 2 else "No flow data yet.")


def cmd_news(_args):
    lines = ["📰 <b>Latest News</b>", ""]
    for s in _latest_scores:
        ticker   = s["symbol"].replace(".JK", "")
        headline = s.get("news_headline", "")
        sent     = s.get("news_sentiment", "NEUTRAL")
        emoji    = tg.SENTIMENT_EMOJI.get(sent, "➖")
        if headline:
            lines.append(f"{emoji} <b>{ticker}</b>: {headline[:80]}")
    tg.send_message("\n".join(lines) if len(lines) > 2 else "No news data yet.")


def cmd_help(_args):
    tg.send_message(
        "🤖 <b>IDX Bot Commands</b>\n\n"
        "/status   — bot + market status\n"
        "/stocks   — current scores for all stocks\n"
        "/briefing — trigger pre-session briefing now\n"
        "/forecast — 5-day price forecasts\n"
        "/flow     — today's foreign flow\n"
        "/news     — latest news headlines\n"
        "/help     — this message"
    )


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    now = datetime.now(WIB)
    logger.info("IDX Stock Bot — Phase 4 starting")
    logger.info(f"Time: {now.strftime('%A %d %b %Y %H:%M WIB')}")
    logger.info(f"Trading day: {is_trading_day()} | Market open: {is_market_open()}")
    logger.info(f"Watchlist: {', '.join(config.STOCKS)}")

    jci = fetch_jci_summary()
    if jci:
        logger.info(
            f"JCI: {jci['jci_close']} ({jci['jci_change_pct']:+.2f}%)"
            f" | Foreign net: {jci['total_foreign_net_idr']}"
        )

    # Pre-compute forecasts
    warmup_models(config.STOCKS)

    # Start Telegram command poller in background
    poller = tg.CommandPoller({
        "/status":   cmd_status,
        "/stocks":   cmd_stocks,
        "/briefing": cmd_briefing,
        "/forecast": cmd_forecast,
        "/flow":     cmd_flow,
        "/news":     cmd_news,
        "/help":     cmd_help,
    })
    poller.start()

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
        poller.stop()
        logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
