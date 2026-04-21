"""
IDX Stock Bot — Phase 5
Full pipeline: 4-layer scoring → AI verdict → Telegram signals + briefings + commands.
Includes: dynamic watchlist (/add /remove) + whale scanner.
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
from ai_agent import get_ai_verdict
from analyzer import analyze_stock
from risk_manager import check_risk_gates, calc_lot_size, record_entry, get_pnl_summary
from scalper import (
    scalp_scan, update_scalp_positions, close_scalp_positions,
    get_scalp_summary, manual_add_scalp, format_scalp_watchlist,
)
from screener import (
    find_top2_scalp_candidates, save_daily_scalp_watchlist,
    find_top2_s1_candidates,    save_s1_watchlist,
)
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

# ── Runtime state ─────────────────────────────────────────────────────────────
# Watchlist is dynamic — persisted in Firestore, mutated by /add /remove + whale scanner
_watchlist: list[str] = []
_latest_scores: list[dict] = []


# ── Watchlist helpers ─────────────────────────────────────────────────────────

def get_watchlist() -> list[str]:
    return _watchlist


def add_to_watchlist(symbol: str) -> bool:
    """Add symbol to watchlist. Returns False if already present."""
    sym = symbol.upper()
    if not sym.endswith(".JK"):
        sym += ".JK"
    if sym in _watchlist:
        return False
    _watchlist.append(sym)
    db.save_watchlist(_watchlist)
    return True


def remove_from_watchlist(symbol: str) -> bool:
    """Remove symbol from watchlist. Returns False if not found."""
    sym = symbol.upper()
    if not sym.endswith(".JK"):
        sym += ".JK"
    if sym not in _watchlist:
        return False
    _watchlist.remove(sym)
    db.save_watchlist(_watchlist)
    return True


# ── Core scan ─────────────────────────────────────────────────────────────────

def _scan_symbol(symbol: str) -> dict | None:
    """Fetch + score a single symbol. Returns result dict or None."""
    df = fetch_candles(symbol, interval=config.CANDLE_INTERVAL, period="5d")
    if df is None:
        logger.warning(f"{symbol}: no candle data, skipping")
        return None

    tech = calculate_indicators(df)
    if tech is None:
        logger.warning(f"{symbol}: indicator calc failed, skipping")
        return None

    forecast = forecast_5d(symbol)
    flow     = fetch_foreign_flow(symbol)
    news     = score_news_sentiment(symbol)
    result   = score_stock(symbol, tech, forecast, flow, news)
    result["symbol"] = symbol
    return result


def scan_stocks():
    global _latest_scores
    now_wib = datetime.now(WIB).strftime("%H:%M WIB")
    logger.info(f"── Scan @ {now_wib} ──────────────────────────")

    scores = []
    for symbol in list(_watchlist):
        try:
            result = _scan_symbol(symbol)
            if result is None:
                continue

            scores.append(result)
            s = result["scores"]
            logger.info(
                f"{symbol:10s} | {result['price']:>8.2f}"
                f" | T={s['technical']:>2d} P={s['prophet']:>2d}"
                f" F={s['foreign']:>2d} N={s['news']:>2d}"
                f" | total={result['total_score']:>3d}/100 | {result['verdict']}"
            )

            # Save snapshot
            _save_snapshot(symbol, result)

            # Fire signal + AI verdict if score qualifies
            if should_signal(result):
                allowed, reason = check_risk_gates(symbol)
                if not allowed:
                    logger.info(f"{symbol}: signal suppressed by risk gate — {reason}")
                else:
                    snapshot_for_ai = {
                        **result,
                        "technical_score": s["technical"],
                        "prophet_score":   s["prophet"],
                        "foreign_score":   s["foreign"],
                        "news_score":      s["news"],
                    }
                    ai   = get_ai_verdict(symbol, snapshot_for_ai)
                    lots = ai.get("lots", 1) if ai else calc_lot_size(result["price"], 50)
                    result["ai_verdict"] = ai

                    logger.info(
                        f"{symbol}: signal fired score={result['total_score']}"
                        + (f" AI={ai['action']} conf={ai['confidence']}% lots={lots}" if ai else "")
                    )
                    msg = tg.format_signal_with_ai(symbol, snapshot_for_ai, ai)
                    tg.send_message(msg)
                    db.save_signal(symbol, {**result, "ai_verdict": ai})
                    record_entry(symbol, result["price"], lots)

        except Exception as e:
            logger.error(f"{symbol}: unexpected error — {e}")

    _latest_scores = scores


def _save_snapshot(symbol: str, result: dict):
    s = result["scores"]
    db.save_snapshot(symbol, {
        "price":               result["price"],
        "technical_score":     s["technical"],
        "prophet_score":       s["prophet"],
        "foreign_score":       s["foreign"],
        "news_score":          s["news"],
        "total_score":         result["total_score"],
        "verdict":             result["verdict"],
        "rsi":                 result.get("rsi"),
        "ma_trend":            result.get("ma_trend"),
        "volume_ratio":        result.get("volume_ratio"),
        "candle_pattern":      result.get("candle_pattern"),
        "forecast_5d":         result.get("forecast_5d"),
        "trend_pct":           result.get("trend_pct"),
        "trend":               result.get("trend"),
        "net_foreign_buy_idr": result.get("net_foreign_buy_idr"),
        "days_consecutive":    result.get("days_consecutive"),
        "news_sentiment":      result.get("news_sentiment"),
        "news_headline":       result.get("news_headline"),
        "reasons":             result.get("reasons", []),
        "phase":               5,
    })


# ── Whale scanner ─────────────────────────────────────────────────────────────

def _handle_whale_callback(data: str, callback_query_id: str, message_id: int, chat_id: int):
    """Called when user taps ✅ Add or ❌ Skip on a whale confirm prompt."""
    if data.startswith("whale_add_"):
        symbol = data[len("whale_add_"):]
        ticker = symbol.replace(".JK", "")
        added  = add_to_watchlist(symbol)
        if added:
            tg.answer_callback_query(callback_query_id, f"✅ {ticker} added!")
            tg.edit_message_text(
                chat_id, message_id,
                f"🐋 <b>WHALE SURGE — {ticker}.JK</b>\n✅ Added to watchlist — tracking started.",
            )
            logger.info(f"Whale confirm: {symbol} manually added to watchlist")
        else:
            tg.answer_callback_query(callback_query_id, f"{ticker} already in watchlist.")
            tg.edit_message_text(
                chat_id, message_id,
                f"🐋 <b>WHALE SURGE — {ticker}.JK</b>\n⚠️ Already in watchlist.",
            )

    elif data.startswith("whale_skip_"):
        symbol = data[len("whale_skip_"):]
        ticker = symbol.replace(".JK", "")
        tg.answer_callback_query(callback_query_id, f"Skipped {ticker}.")
        tg.edit_message_text(
            chat_id, message_id,
            f"🐋 <b>WHALE SURGE — {ticker}.JK</b>\n❌ Skipped.",
        )
        logger.info(f"Whale confirm: {symbol} skipped by user")


def whale_scan():
    """
    Scan the broader UNIVERSE for abnormal volume surges.
    vol >= WHALE_CONFIRM_THRESHOLD (5×): sends a Yes/No Telegram prompt for manual decision.
    vol >= WHALE_VOL_THRESHOLD (3×):    auto-adds if WHALE_AUTO_ADD and tech score qualifies.
    """
    candidates = [s for s in config.UNIVERSE if s not in _watchlist]
    if not candidates:
        return

    logger.info(f"Whale scan — checking {len(candidates)} universe stocks for volume surges")
    surges = []

    for symbol in candidates:
        try:
            df = fetch_candles(symbol, interval=config.CANDLE_INTERVAL, period="5d")
            if df is None:
                continue
            tech = calculate_indicators(df)
            if tech is None:
                continue
            if tech["volume_ratio"] >= config.WHALE_VOL_THRESHOLD:
                surges.append((symbol, tech["volume_ratio"], tech["score"], tech))
        except Exception as e:
            logger.warning(f"Whale scan {symbol}: {e}")

    if not surges:
        logger.info("Whale scan complete — no surges found")
        return

    # Sort by volume ratio, take top N
    surges.sort(key=lambda x: x[1], reverse=True)
    top = surges[:config.WHALE_TOP_N]

    logger.info(f"Whale scan — {len(top)} surge(s) found: {[s[0] for s in top]}")

    for symbol, vol_ratio, tech_score, tech in top:
        if vol_ratio >= config.WHALE_CONFIRM_THRESHOLD:
            # Extreme surge — send a manual confirm prompt
            text, keyboard = tg.format_whale_confirm(symbol, vol_ratio, tech_score)
            tg.send_message_with_keyboard(text, keyboard)
            logger.info(f"Whale: confirm prompt sent for {symbol} (vol={vol_ratio:.1f}x)")
        else:
            # Normal surge — auto-add if score qualifies, else just notify
            auto_added = False
            if config.WHALE_AUTO_ADD and tech_score >= config.WHALE_MIN_SCORE:
                added = add_to_watchlist(symbol)
                if added:
                    auto_added = True
                    logger.info(f"Whale: auto-added {symbol} (vol={vol_ratio:.1f}x score={tech_score})")
            msg = tg.format_whale_alert(symbol, vol_ratio, tech_score, auto_added)
            tg.send_message(msg)


# ── Scheduled jobs ────────────────────────────────────────────────────────────

def presession1():
    logger.info("Running pre-session 1 briefing")
    _send_briefing(session=1)


def presession2():
    logger.info("Running pre-session 2 briefing")
    _send_briefing(session=2)


def _send_briefing(session: int, chat_id=None):
    jci           = fetch_jci_summary()
    scalp_summary = get_scalp_summary()

    if session == 2:
        # Session 2 (midday break): use today's Session 1 intraday data
        candidates = find_top2_s1_candidates(_watchlist)
        if candidates:
            save_s1_watchlist(candidates)
    else:
        # Session 1 (pre-open): use yesterday's daily (D-1) data
        candidates = find_top2_scalp_candidates(_watchlist)
        if candidates:
            save_daily_scalp_watchlist(candidates)

    msg = tg.format_presession_briefing(
        session=session,
        date_str=datetime.now(WIB).strftime("%a %d %b %Y"),
        jci=jci,
        stock_scores=_latest_scores,
        scalp_summary=scalp_summary,
        scalp_candidates=candidates,
    )
    tg.send_message(msg, chat_id=chat_id)


def eod_summary():
    logger.info("Running end-of-day summary")
    # Close all scalp positions at market close
    close_scalp_positions()
    # Send scalping EOD report
    scalp_summary = get_scalp_summary()
    if scalp_summary["total"] > 0:
        tg.send_message(format_scalp_watchlist(scalp_summary))
    # Send main EOD report
    signals = db.get_today_signals()
    msg = tg.format_eod_summary(_latest_scores, signals)
    tg.send_message(msg)


# ── Command handlers ──────────────────────────────────────────────────────────

def cmd_status(_args, chat_id=None):
    tg.send_message(tg.format_status(is_market_open(), is_trading_day(), len(_watchlist)), chat_id=chat_id)


def cmd_stocks(_args, chat_id=None):
    if not _latest_scores:
        tg.send_message("No scores yet — waiting for first scan.", chat_id=chat_id)
        return
    tg.send_message(tg.format_stocks_list(_latest_scores), chat_id=chat_id)


def cmd_briefing(_args, chat_id=None):
    session = 1 if datetime.now(WIB).hour < 12 else 2
    _send_briefing(session, chat_id=chat_id)


def cmd_forecast(_args, chat_id=None):
    if not _latest_scores:
        tg.send_message("No forecast data yet.", chat_id=chat_id)
        return
    lines = ["📈 <b>5-Day Forecasts</b>", ""]
    for s in _latest_scores:
        ticker = s["symbol"].replace(".JK", "")
        pct    = s.get("trend_pct")
        trend  = s.get("trend", "—")
        f5d    = s.get("forecast_5d")
        price  = s.get("price", 0)
        t_e    = tg.TREND_EMOJI.get(trend, "⚪")
        if pct is not None:
            lines.append(f"{t_e} <b>{ticker}</b>  {price:,.0f} → {f5d:,.0f}  ({pct:+.1f}%)")
        else:
            lines.append(f"⚪ <b>{ticker}</b>  No forecast")
    tg.send_message("\n".join(lines), chat_id=chat_id)


def cmd_flow(_args, chat_id=None):
    if not _latest_scores:
        tg.send_message("No flow data yet.", chat_id=chat_id)
        return
    lines = ["🌊 <b>Foreign Flow Today</b>", ""]
    for s in _latest_scores:
        ticker = s["symbol"].replace(".JK", "")
        days   = s.get("days_consecutive", 0) or 0
        net    = s.get("net_foreign_buy_idr", 0) or 0
        net_b  = net / 1_000_000_000
        emoji  = "🟢" if days > 0 else ("🔴" if days < 0 else "⚪")
        lines.append(f"{emoji} <b>{ticker}</b>  {net_b:+.2f}B IDR  ({days:+d}d)")
    tg.send_message("\n".join(lines), chat_id=chat_id)


def cmd_news(_args, chat_id=None):
    if not _latest_scores:
        tg.send_message("No news data yet.", chat_id=chat_id)
        return
    lines = ["📰 <b>Latest News</b>", ""]
    for s in _latest_scores:
        ticker   = s["symbol"].replace(".JK", "")
        headline = s.get("news_headline", "")
        sent     = s.get("news_sentiment", "NEUTRAL")
        emoji    = tg.SENTIMENT_EMOJI.get(sent, "➖")
        if headline:
            lines.append(f"{emoji} <b>{ticker}</b>: {headline[:80]}")
    tg.send_message("\n".join(lines) if len(lines) > 2 else "No news data yet.", chat_id=chat_id)


def cmd_add(args, chat_id=None):
    if not args:
        tg.send_message("Usage: /add BBCA", chat_id=chat_id)
        return
    symbol = args[0].upper()
    if add_to_watchlist(symbol):
        tg.send_message(tg.format_watchlist_change(symbol, "add", _watchlist), chat_id=chat_id)
        logger.info(f"Watchlist: added {symbol}")
    else:
        tg.send_message(f"{symbol}.JK is already in the watchlist.", chat_id=chat_id)


def cmd_remove(args, chat_id=None):
    if not args:
        tg.send_message("Usage: /remove BBCA", chat_id=chat_id)
        return
    symbol = args[0].upper()
    if remove_from_watchlist(symbol):
        tg.send_message(tg.format_watchlist_change(symbol, "remove", _watchlist), chat_id=chat_id)
        logger.info(f"Watchlist: removed {symbol}")
    else:
        tg.send_message(f"{symbol}.JK is not in the watchlist.", chat_id=chat_id)


def cmd_pnl(_args, chat_id=None):
    pnl = get_pnl_summary()
    tg.send_message(tg.format_pnl(pnl), chat_id=chat_id)


def cmd_scalps(_args, chat_id=None):
    summary = get_scalp_summary()
    tg.send_message(format_scalp_watchlist(summary), chat_id=chat_id)


def cmd_scalpadd(args, chat_id=None):
    if not args:
        tg.send_message("Usage: /scalpadd BBCA", chat_id=chat_id)
        return
    symbol = args[0].upper()
    pos = manual_add_scalp(symbol)
    if pos:
        ticker = symbol if symbol.endswith(".JK") else symbol + ".JK"
        tg.send_message(
            f"⚡ <b>{ticker} added to scalping watchlist</b>\n"
            f"Entry price: <b>{pos['entry_price']:,.0f}</b>\n"
            f"Open price:  {pos['open_price']:,.0f}\n"
            f"Drop from open: {pos['drop_pct']:+.1f}%\n"
            f"Monitoring until 15:49 WIB — P&L at EOD.",
            chat_id=chat_id,
        )
    else:
        tg.send_message(f"❌ Could not fetch data for {symbol}.", chat_id=chat_id)


def cmd_analyze(args, chat_id=None):
    if not args:
        tg.send_message("Usage: /analyze BBCA", chat_id=chat_id)
        return
    symbol = args[0].upper()
    tg.send_message(
        f"🔍 Memulai analisis mendalam untuk <b>{symbol}.JK</b>...\n(~30–60 detik)",
        chat_id=chat_id,
    )
    logger.info(f"/analyze {symbol} triggered")
    parts = analyze_stock(symbol)
    tg.send_long_message(parts, chat_id=chat_id)


def cmd_help(_args, chat_id=None):
    tg.send_message(
        "🤖 <b>IDX Bot Commands</b>\n\n"
        "/status        — bot + market status\n"
        "/stocks        — current scores for all stocks\n"
        "/briefing      — trigger pre-session briefing now\n"
        "/forecast      — 5-day price forecasts\n"
        "/flow          — today's foreign flow\n"
        "/news          — latest news headlines\n"
        "/add BBCA      — add stock to watchlist\n"
        "/remove BBCA   — remove stock from watchlist\n"
        "/pnl           — today's signal P&L\n"
        "/scalps        — scalping watchlist + live P&L\n"
        "/scalpadd BBCA — manually add to scalping watchlist\n"
        "/analyze BBCA  — full fundamental + technical report\n"
        "/help          — this message",
        chat_id=chat_id,
    )


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    global _watchlist

    now = datetime.now(WIB)
    logger.info("IDX Stock Bot — Phase 5 starting")
    logger.info(f"Time: {now.strftime('%A %d %b %Y %H:%M WIB')}")
    logger.info(f"Trading day: {is_trading_day()} | Market open: {is_market_open()}")

    # Load persisted watchlist (falls back to config.STOCKS)
    _watchlist = db.get_watchlist()
    logger.info(f"Watchlist ({len(_watchlist)}): {', '.join(_watchlist)}")

    jci = fetch_jci_summary()
    if jci:
        logger.info(
            f"JCI: {jci['jci_close']} ({jci['jci_change_pct']:+.2f}%)"
            f" | Foreign net: {jci['total_foreign_net_idr']}"
        )

    # Pre-compute forecasts for current watchlist
    warmup_models(_watchlist)

    # Start Telegram command poller in background
    poller = tg.CommandPoller(
        handlers={
            "/status":   cmd_status,
            "/stocks":   cmd_stocks,
            "/briefing": cmd_briefing,
            "/forecast": cmd_forecast,
            "/flow":     cmd_flow,
            "/news":     cmd_news,
            "/add":      cmd_add,
            "/remove":   cmd_remove,
            "/pnl":      cmd_pnl,
            "/scalps":   cmd_scalps,
            "/scalpadd": cmd_scalpadd,
            "/analyze":  cmd_analyze,
            "/help":     cmd_help,
        },
        callback_handlers={
            "whale_": _handle_whale_callback,
        },
    )
    poller.start()

    # Immediate scan
    scan_stocks()

    # Combined scan: watchlist + whale + scalper
    def scan_and_whale():
        scan_stocks()
        if is_market_open():
            whale_scan()
            scalp_scan(notify_fn=tg.send_message)
            update_scalp_positions(notify_fn=tg.send_message)

    logger.info(f"Scheduler started — scan every {config.SCAN_INTERVAL_SEC}s during market hours")
    scheduler = build_scheduler(
        scan_fn=scan_and_whale,
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
