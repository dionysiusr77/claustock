"""
Intraday mean-reversion scalping scanner.

Logic: finds stocks that dropped fast from their opening price with
high volume and oversold RSI — candidates to bounce back toward open.

Separate from whale scanner (which tracks volume surges in any direction).
Scalping watchlist resets each trading day.
Entry price is recorded when added; P&L calculated vs closing price at EOD.
"""

import logging
from datetime import datetime, timezone, date

import pandas as pd

import config
from fetcher import fetch_candles
from indicators import calculate_indicators
import firestore_client as db

logger = logging.getLogger(__name__)

# ── Settings ──────────────────────────────────────────────────────────────────
MIN_DROP_PCT   = 1.5    # minimum intraday drop from open to qualify
MAX_DROP_PCT   = 6.0    # maximum — beyond this is a crash, not a bounce
MIN_RSI        = 20     # too oversold = potential circuit breaker
MAX_RSI        = 48     # must be in oversold territory
MIN_VOL_RATIO  = 1.5    # must have above-average volume
MIN_SCORE      = 55     # minimum scalp score (0–100) to add to watchlist

# ── In-memory scalping watchlist ─────────────────────────────────────────────
# { symbol: { entry_price, entry_time, drop_pct, score, status, exit_price, pnl_pct } }
_scalp_positions: dict = {}
_watchlist_date: str   = ""   # YYYY-MM-DD — reset trigger


def _maybe_reset():
    global _scalp_positions, _watchlist_date
    today = date.today().isoformat()
    if today != _watchlist_date:
        if _scalp_positions:
            logger.info(f"Scalper: daily reset — clearing {len(_scalp_positions)} positions")
        _scalp_positions  = {}
        _watchlist_date   = today


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_scalp_candidate(
    drop_pct: float,
    rsi: float,
    vol_ratio: float,
    ma_trend: str,
) -> int:
    """
    Score a bounce candidate 0–100.
    Sweet spot: drop 1.5–3%, RSI 30–42, volume 2x+, MA not DOWN.
    """
    score = 0

    # Drop score (0–35): sweet spot is -1.5% to -3%
    if 1.5 <= drop_pct < 2.0:
        score += 25
    elif 2.0 <= drop_pct < 3.0:
        score += 35   # sweet spot
    elif 3.0 <= drop_pct < 4.5:
        score += 20
    elif 4.5 <= drop_pct <= MAX_DROP_PCT:
        score += 10

    # RSI score (0–30): lower = more oversold = better bounce potential
    if rsi < 30:
        score += 30
    elif rsi < 35:
        score += 25
    elif rsi < 40:
        score += 20
    elif rsi < 45:
        score += 12
    elif rsi <= MAX_RSI:
        score += 5

    # Volume score (0–25): high volume = real selling, real bounce
    if vol_ratio >= 3.0:
        score += 25
    elif vol_ratio >= 2.0:
        score += 20
    elif vol_ratio >= 1.5:
        score += 12

    # MA trend (0–10): avoid death spirals
    if ma_trend == "UP":
        score += 10     # dropped into an uptrend = strong bounce candidate
    elif ma_trend == "FLAT":
        score += 5
    else:
        score += 0      # downtrend = skip

    return min(score, 100)


# ── Open price fetcher ────────────────────────────────────────────────────────

def _get_open_price(df: pd.DataFrame) -> float | None:
    """
    Get today's opening price from intraday candles.
    Returns the open of the first candle of today's session.
    """
    from scheduler import WIB
    import pytz

    if df is None or df.empty:
        return None

    try:
        today_str = datetime.now(WIB).strftime("%Y-%m-%d")
        idx = df.index
        if idx.tzinfo is None:
            idx = idx.tz_localize("UTC").tz_convert(WIB)
        else:
            idx = idx.tz_convert(WIB)

        today_mask = idx.strftime("%Y-%m-%d") == today_str
        today_df   = df[today_mask]

        if today_df.empty:
            return None

        return float(today_df["open"].iloc[0])
    except Exception as e:
        logger.warning(f"_get_open_price: {e}")
        return None


# ── Main scanner ──────────────────────────────────────────────────────────────

def scalp_scan(notify_fn=None) -> list[dict]:
    """
    Scan the full UNIVERSE for bounce candidates.
    Auto-adds qualifying stocks to the scalping watchlist.

    notify_fn: optional callable(msg: str) to send Telegram alerts.
    Returns list of newly added candidates.
    """
    _maybe_reset()

    candidates = []
    already_watching = set(_scalp_positions.keys())

    for symbol in config.UNIVERSE:
        if symbol in already_watching:
            continue  # already tracking

        try:
            df = fetch_candles(symbol, interval=config.CANDLE_INTERVAL, period="5d")
            if df is None:
                continue

            tech = calculate_indicators(df)
            if tech is None:
                continue

            open_price = _get_open_price(df)
            if not open_price or open_price <= 0:
                continue

            current = tech["price"]
            drop_pct = (open_price - current) / open_price * 100

            # Filter: must be a meaningful drop (not a gap-up then bounce)
            if drop_pct < MIN_DROP_PCT or drop_pct > MAX_DROP_PCT:
                continue

            # Filter: RSI must be in oversold range
            if not (MIN_RSI <= tech["rsi"] <= MAX_RSI):
                continue

            # Filter: volume must be active
            if tech["volume_ratio"] < MIN_VOL_RATIO:
                continue

            score = _score_scalp_candidate(
                drop_pct, tech["rsi"], tech["volume_ratio"], tech["ma_trend"]
            )

            if score < MIN_SCORE:
                continue

            # Qualifies — add to scalping watchlist
            _scalp_positions[symbol] = {
                "entry_price": current,
                "open_price":  open_price,
                "entry_time":  datetime.now(timezone.utc).isoformat(),
                "drop_pct":    round(drop_pct, 2),
                "rsi":         tech["rsi"],
                "vol_ratio":   tech["volume_ratio"],
                "ma_trend":    tech["ma_trend"],
                "score":       score,
                "status":      "watching",
                "exit_price":  None,
                "pnl_pct":     None,
            }

            entry = {
                "symbol":      symbol,
                "entry_price": current,
                "open_price":  open_price,
                "drop_pct":    round(drop_pct, 2),
                "rsi":         tech["rsi"],
                "vol_ratio":   tech["volume_ratio"],
                "score":       score,
            }
            candidates.append(entry)
            logger.info(
                f"Scalp candidate: {symbol} drop={drop_pct:.1f}% "
                f"RSI={tech['rsi']:.0f} vol={tech['volume_ratio']:.1f}x score={score}"
            )

            if notify_fn:
                notify_fn(_format_scalp_alert(entry))

        except Exception as e:
            logger.warning(f"scalp_scan({symbol}): {e}")

    if candidates:
        logger.info(f"Scalper: {len(candidates)} new candidate(s) added to watchlist")

    return candidates


# ── Price monitoring ──────────────────────────────────────────────────────────

def update_scalp_positions(notify_fn=None):
    """
    Refresh current prices for all watched positions.
    Fires a Telegram alert if a position recovers >= 1% toward open.
    """
    _maybe_reset()
    if not _scalp_positions:
        return

    for symbol, pos in list(_scalp_positions.items()):
        if pos["status"] != "watching":
            continue
        try:
            df = fetch_candles(symbol, interval=config.CANDLE_INTERVAL, period="1d")
            if df is None:
                continue

            current = float(df["close"].iloc[-1])
            entry   = pos["entry_price"]
            move_pct = (current - entry) / entry * 100
            pos["current_price"] = current
            pos["move_pct"]      = round(move_pct, 2)

            # Alert if bouncing nicely
            prev_move = pos.get("last_alerted_move", -999)
            if move_pct >= 1.0 and move_pct > prev_move + 0.5:
                pos["last_alerted_move"] = move_pct
                if notify_fn:
                    notify_fn(_format_scalp_bounce_alert(symbol, pos))

        except Exception as e:
            logger.warning(f"update_scalp_positions({symbol}): {e}")


# ── EOD close-out ─────────────────────────────────────────────────────────────

def close_scalp_positions():
    """
    Called at 15:49 (market close) — records final P&L for all open positions.
    Uses last known price as exit price.
    """
    _maybe_reset()
    if not _scalp_positions:
        return

    for symbol, pos in _scalp_positions.items():
        if pos["status"] != "watching":
            continue
        try:
            df = fetch_candles(symbol, interval=config.CANDLE_INTERVAL, period="1d")
            exit_price = float(df["close"].iloc[-1]) if df is not None else pos["entry_price"]
        except Exception:
            exit_price = pos["entry_price"]

        entry    = pos["entry_price"]
        gross    = (exit_price - entry) / entry * 100
        fees     = (config.BUY_FEE_PCT + config.SELL_FEE_PCT) * 100  # % form
        net_pct  = gross - fees

        pos["exit_price"] = exit_price
        pos["pnl_pct"]    = round(net_pct, 2)
        pos["status"]     = "closed"

        logger.info(
            f"Scalp closed: {symbol} entry={entry:,.0f} exit={exit_price:,.0f} "
            f"P&L={net_pct:+.2f}%"
        )


# ── P&L summary ───────────────────────────────────────────────────────────────

def get_scalp_summary() -> dict:
    """Returns today's scalping watchlist with per-position status."""
    _maybe_reset()

    rows = []
    total_pct = 0.0
    closed_count = 0

    for symbol, pos in _scalp_positions.items():
        entry   = pos["entry_price"]
        current = pos.get("current_price", entry)
        exit_p  = pos.get("exit_price")

        if pos["status"] == "closed" and exit_p:
            pnl_pct = pos.get("pnl_pct", 0)
        else:
            gross   = (current - entry) / entry * 100
            fees    = (config.BUY_FEE_PCT + config.SELL_FEE_PCT) * 100
            pnl_pct = round(gross - fees, 2)

        if pos["status"] == "closed":
            closed_count += 1
            total_pct += pnl_pct

        rows.append({
            "symbol":      symbol,
            "entry_price": entry,
            "open_price":  pos["open_price"],
            "drop_pct":    pos["drop_pct"],
            "current":     current,
            "pnl_pct":     pnl_pct,
            "status":      pos["status"],
            "score":       pos["score"],
            "rsi":         pos["rsi"],
        })

    rows.sort(key=lambda x: x["pnl_pct"], reverse=True)

    return {
        "date":          _watchlist_date,
        "total":         len(rows),
        "closed":        closed_count,
        "rows":          rows,
        "avg_pnl_pct":   round(total_pct / closed_count, 2) if closed_count else None,
    }


# ── Manual add ────────────────────────────────────────────────────────────────

def manual_add_scalp(symbol: str) -> dict | None:
    """
    Manually add a stock to the scalping watchlist via /scalpadd command.
    Records current price as entry. Returns position dict or None if fetch fails.
    """
    _maybe_reset()
    sym = symbol.upper()
    if not sym.endswith(".JK"):
        sym += ".JK"

    try:
        df = fetch_candles(sym, interval=config.CANDLE_INTERVAL, period="5d")
        if df is None:
            return None
        tech       = calculate_indicators(df)
        open_price = _get_open_price(df) or float(df["close"].iloc[-1])
        current    = tech["price"] if tech else float(df["close"].iloc[-1])
        drop_pct   = (open_price - current) / open_price * 100 if open_price else 0

        _scalp_positions[sym] = {
            "entry_price": current,
            "open_price":  open_price,
            "entry_time":  datetime.now(timezone.utc).isoformat(),
            "drop_pct":    round(drop_pct, 2),
            "rsi":         tech["rsi"]        if tech else None,
            "vol_ratio":   tech["volume_ratio"] if tech else None,
            "ma_trend":    tech["ma_trend"]   if tech else None,
            "score":       0,
            "status":      "watching",
            "exit_price":  None,
            "pnl_pct":     None,
        }
        logger.info(f"Manual scalp add: {sym} @ {current:,.0f}")
        return _scalp_positions[sym]
    except Exception as e:
        logger.error(f"manual_add_scalp({symbol}): {e}")
        return None


# ── Telegram formatters ───────────────────────────────────────────────────────

def _format_scalp_alert(entry: dict) -> str:
    ticker = entry["symbol"].replace(".JK", "")
    return (
        f"⚡ <b>SCALP CANDIDATE — {ticker}.JK</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Open:    <b>{entry['open_price']:,.0f}</b>\n"
        f"Current: <b>{entry['entry_price']:,.0f}</b>  "
        f"(<b>-{entry['drop_pct']:.1f}%</b> from open)\n"
        f"RSI:     {entry['rsi']:.0f} (oversold)\n"
        f"Volume:  {entry['vol_ratio']:.1f}x average\n"
        f"Score:   {entry['score']}/100\n"
        f"\n"
        f"🎯 Target: {entry['open_price']:,.0f}  (+{entry['drop_pct']:.1f}%)\n"
        f"🛑 SL: {entry['entry_price'] * 0.988:,.0f}  (-1.2%)\n"
        f"⚡ Hold: intraday only — close before 15:45 WIB"
    )


def _format_scalp_bounce_alert(symbol: str, pos: dict) -> str:
    ticker   = symbol.replace(".JK", "")
    move_pct = pos.get("move_pct", 0)
    current  = pos.get("current_price", pos["entry_price"])
    return (
        f"🔄 <b>SCALP BOUNCING — {ticker}.JK</b>\n"
        f"Entry:   {pos['entry_price']:,.0f}\n"
        f"Current: <b>{current:,.0f}</b>  ({move_pct:+.1f}% from entry)\n"
        f"Open:    {pos['open_price']:,.0f}  (target)"
    )


def format_scalp_watchlist(summary: dict) -> str:
    rows  = summary.get("rows", [])
    today = summary.get("date", "—")
    lines = [
        f"⚡ <b>SCALPING WATCHLIST — {today}</b>",
        f"Positions: {summary['total']} | "
        + (f"Avg P&L: {summary['avg_pnl_pct']:+.2f}%" if summary.get('avg_pnl_pct') is not None else "Open"),
        "━━━━━━━━━━━━━━━━━━━━",
        "",
    ]

    if not rows:
        lines.append("Belum ada kandidat scalp hari ini.")
        return "\n".join(lines)

    for r in rows:
        ticker  = r["symbol"].replace(".JK", "")
        pnl     = r["pnl_pct"]
        emoji   = "🟢" if pnl >= 0 else "🔴"
        status  = "✅" if r["status"] == "closed" else "👀"
        lines.append(
            f"{emoji}{status} <b>{ticker}</b>"
            f"  Entry: {r['entry_price']:,.0f}"
            f"  Now: {r.get('current', r['entry_price']):,.0f}"
            f"  <b>{pnl:+.2f}%</b>"
        )
        lines.append(
            f"       Drop from open: -{r['drop_pct']:.1f}%"
            f"  RSI: {r['rsi']:.0f}"
            f"  Score: {r['score']}"
        )

    if summary.get("avg_pnl_pct") is not None:
        lines += ["", f"📊 Rata-rata P&L: <b>{summary['avg_pnl_pct']:+.2f}%</b>"]

    return "\n".join(lines)
