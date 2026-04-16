"""
Intraday scalping + multi-day momentum scanner.

Two separate watchlists:
  SCALP      — mean-reversion plays: dropped from open, RSI oversold, bounce back to open.
               Closed at session end (15:49 WIB) each day. Sector-correlated entries capped.
  MOMENTUM   — trend-continuation plays: strong score, RSI bullish, positive news.
               Carried overnight for up to MOMENTUM_HOLD_DAYS (default 3) trading days.
               Exits at target (+3–5%), stop-loss (-1.5%), or after hold_days.

Scalping watchlist resets each trading day.
Entry price is recorded when added; P&L calculated vs closing/current price.
"""

import logging
from datetime import datetime, timezone, date, timedelta

import pandas as pd

import config
from fetcher import fetch_candles
from indicators import calculate_indicators
import firestore_client as db

logger = logging.getLogger(__name__)

# ── Legacy filter constants (kept for _score_scalp_candidate) ─────────────────
MIN_DROP_PCT  = config.SCALP_MIN_DROP_FROM_OPEN
MAX_DROP_PCT  = config.SCALP_MAX_DROP_FROM_OPEN
MIN_RSI       = 20
MAX_RSI       = config.SCALP_MAX_RSI
MIN_VOL_RATIO = 1.5
MIN_SCORE     = config.SCALP_MIN_SCORE

# ── In-memory watchlists ──────────────────────────────────────────────────────
# Scalp:    { symbol: { entry_price, open_price, entry_time, drop_pct, rsi,
#                       vol_ratio, ma_trend, score, status, exit_price, pnl_pct } }
# Momentum: { symbol: { entry_price, entry_date, score, rsi, news_score,
#                       news_headline, target_pct, stop_loss_pct, hold_days,
#                       status, exit_price, pnl_pct, current_price } }
_scalp_positions: dict    = {}
_momentum_positions: dict = {}
_watchlist_date: str      = ""   # YYYY-MM-DD — scalp reset trigger


def _maybe_reset():
    """Reset scalp watchlist at the start of each new trading day."""
    global _scalp_positions, _watchlist_date
    today = date.today().isoformat()
    if today != _watchlist_date:
        if _scalp_positions:
            logger.info(f"Scalper: daily reset — clearing {len(_scalp_positions)} scalp positions")
        _scalp_positions = {}
        _watchlist_date  = today


# ── Signal classifier ─────────────────────────────────────────────────────────

def classify_signal(result: dict) -> str:
    """
    Classify a scanner result as "SCALP", "MOMENTUM", or "SKIP".

    SCALP criteria (intraday mean-reversion):
      - RSI < SCALP_MAX_RSI (default 50)
      - intraday drop from open between SCALP_MIN and SCALP_MAX %
      - volume_ratio >= 1.5

    MOMENTUM criteria (multi-day trend continuation):
      - RSI >= MOMENTUM_MIN_RSI (default 50)
      - news_score >= MOMENTUM_MIN_NEWS_SCORE (default 12)
      - total_score >= MOMENTUM_MIN_SCORE (default 60)

    Anything else → SKIP.
    """
    rsi          = result.get("rsi") or 0
    news_score   = result.get("scores", {}).get("news", 0) if "scores" in result else result.get("news_score", 0)
    total_score  = result.get("total_score", 0)
    volume_ratio = result.get("volume_ratio") or 0
    drop_pct     = result.get("drop_pct", 0)   # positive = dropped from open

    # Scalp: oversold + intraday drop + active volume
    if (
        rsi < config.SCALP_MAX_RSI
        and MIN_DROP_PCT <= drop_pct <= MAX_DROP_PCT
        and volume_ratio >= MIN_VOL_RATIO
    ):
        return "SCALP"

    # Momentum: bullish RSI + strong news + high score
    if (
        rsi >= config.MOMENTUM_MIN_RSI
        and news_score >= config.MOMENTUM_MIN_NEWS_SCORE
        and total_score >= config.MOMENTUM_MIN_SCORE
    ):
        return "MOMENTUM"

    return "SKIP"


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

    return min(score, 100)


# ── Open price fetcher ────────────────────────────────────────────────────────

def _get_open_price(df: pd.DataFrame) -> float | None:
    """
    Get today's opening price from intraday candles.
    Returns the open of the first candle of today's session.
    """
    from scheduler import WIB

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


# ── Sector guard ──────────────────────────────────────────────────────────────

def _sector_allows(symbol: str) -> bool:
    """
    Returns True if adding `symbol` to the scalp watchlist is allowed
    under the MAX_SAME_SECTOR cap.
    """
    sector = config.SECTORS.get(symbol)
    if sector is None:
        return True   # unknown sector — no restriction

    count = sum(
        1 for sym in _scalp_positions
        if config.SECTORS.get(sym) == sector
    )
    return count < config.MAX_SAME_SECTOR


# ── Fresh price fetcher ───────────────────────────────────────────────────────

def _fetch_current_price(symbol: str) -> float | None:
    """Fetch the latest close price for a symbol (used for live P&L display)."""
    try:
        df = fetch_candles(symbol, interval=config.CANDLE_INTERVAL, period="1d")
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
    except Exception as e:
        logger.warning(f"_fetch_current_price({symbol}): {e}")
    return None


# ── Main scanner ──────────────────────────────────────────────────────────────

def scalp_scan(notify_fn=None) -> list[dict]:
    """
    Scan the full UNIVERSE for bounce candidates (SCALP) and momentum plays.
    Auto-adds qualifying stocks to the appropriate watchlist.

    notify_fn: optional callable(msg: str) to send Telegram alerts.
    Returns list of newly added candidates (both scalp and momentum).
    """
    _maybe_reset()

    candidates = []
    already_scalp     = set(_scalp_positions.keys())
    already_momentum  = set(_momentum_positions.keys())

    for symbol in config.UNIVERSE:
        if symbol in already_scalp and symbol in already_momentum:
            continue  # already tracking in both

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

            current  = tech["price"]
            drop_pct = (open_price - current) / open_price * 100

            # Build a result dict for classify_signal
            result_ctx = {
                "rsi":          tech["rsi"],
                "volume_ratio": tech["volume_ratio"],
                "drop_pct":     drop_pct,
                "total_score":  tech["score"],   # tech-only at scan time
                "news_score":   0,               # not available at scan time
            }

            signal_type = classify_signal(result_ctx)

            if signal_type == "SCALP" and symbol not in already_scalp:
                score = _score_scalp_candidate(
                    drop_pct, tech["rsi"], tech["volume_ratio"], tech["ma_trend"]
                )
                if score < MIN_SCORE:
                    continue
                if not _sector_allows(symbol):
                    logger.info(
                        f"Scalp sector cap: {symbol} skipped "
                        f"(sector={config.SECTORS.get(symbol)}, "
                        f"max={config.MAX_SAME_SECTOR})"
                    )
                    continue

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
                    "type":        "SCALP",
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

            elif signal_type == "MOMENTUM" and symbol not in already_momentum:
                # Momentum uses the scan's tech score; proper full score would need
                # all 4 layers but we accept the tech score as a conservative gate.
                _momentum_positions[symbol] = {
                    "entry_price":    current,
                    "entry_date":     date.today().isoformat(),
                    "score":          tech["score"],
                    "rsi":            tech["rsi"],
                    "news_score":     0,
                    "news_headline":  "",
                    "target_pct":     config.MOMENTUM_TARGET_PCT,
                    "stop_loss_pct":  config.MOMENTUM_STOP_LOSS_PCT,
                    "hold_days":      config.MOMENTUM_HOLD_DAYS,
                    "status":         "watching",
                    "exit_price":     None,
                    "pnl_pct":        None,
                    "current_price":  current,
                }

                entry = {
                    "type":        "MOMENTUM",
                    "symbol":      symbol,
                    "entry_price": current,
                    "rsi":         tech["rsi"],
                    "score":       tech["score"],
                    "target_pct":  config.MOMENTUM_TARGET_PCT,
                    "sl_pct":      config.MOMENTUM_STOP_LOSS_PCT,
                }
                candidates.append(entry)
                logger.info(
                    f"Momentum candidate: {symbol} "
                    f"RSI={tech['rsi']:.0f} score={tech['score']}"
                )
                if notify_fn:
                    notify_fn(_format_momentum_alert(entry))

        except Exception as e:
            logger.warning(f"scalp_scan({symbol}): {e}")

    if candidates:
        logger.info(
            f"Scanner: {sum(1 for c in candidates if c['type']=='SCALP')} scalp, "
            f"{sum(1 for c in candidates if c['type']=='MOMENTUM')} momentum candidates added"
        )

    return candidates


# ── Price monitoring ──────────────────────────────────────────────────────────

def update_scalp_positions(notify_fn=None):
    """
    Refresh current prices for all watched scalp positions.
    Fires a Telegram alert if a position recovers >= 1% toward open.
    Also checks momentum positions for target/stop-loss hits.
    """
    _maybe_reset()

    # Update scalp positions
    for symbol, pos in list(_scalp_positions.items()):
        if pos["status"] != "watching":
            continue
        try:
            current = _fetch_current_price(symbol)
            if current is None:
                continue

            entry    = pos["entry_price"]
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

    # Update momentum positions — check target/SL hits
    today = date.today().isoformat()
    for symbol, pos in list(_momentum_positions.items()):
        if pos["status"] != "watching":
            continue
        try:
            current = _fetch_current_price(symbol)
            if current is None:
                continue

            entry    = pos["entry_price"]
            move_pct = (current - entry) / entry * 100
            pos["current_price"] = current

            # Check target hit
            if move_pct >= pos["target_pct"]:
                _close_momentum(symbol, pos, current, "TARGET_HIT")
                if notify_fn:
                    notify_fn(_format_momentum_exit_alert(symbol, pos, "TARGET HIT"))

            # Check stop-loss hit
            elif move_pct <= pos["stop_loss_pct"]:
                _close_momentum(symbol, pos, current, "STOP_LOSS")
                if notify_fn:
                    notify_fn(_format_momentum_exit_alert(symbol, pos, "STOP LOSS"))

            # Check hold_days expiry
            else:
                entry_date = date.fromisoformat(pos["entry_date"])
                days_held  = (date.today() - entry_date).days
                if days_held >= pos["hold_days"]:
                    _close_momentum(symbol, pos, current, "EXPIRED")
                    if notify_fn:
                        notify_fn(_format_momentum_exit_alert(symbol, pos, "HOLD EXPIRED"))

        except Exception as e:
            logger.warning(f"update_momentum_positions({symbol}): {e}")


def _close_momentum(symbol: str, pos: dict, exit_price: float, reason: str):
    """Mark a momentum position as closed and record P&L."""
    entry   = pos["entry_price"]
    gross   = (exit_price - entry) / entry * 100
    fees    = (config.BUY_FEE_PCT + config.SELL_FEE_PCT) * 100
    net_pct = round(gross - fees, 2)

    pos["exit_price"] = exit_price
    pos["pnl_pct"]    = net_pct
    pos["status"]     = "closed"
    pos["exit_reason"]= reason

    logger.info(
        f"Momentum closed ({reason}): {symbol} "
        f"entry={entry:,.0f} exit={exit_price:,.0f} P&L={net_pct:+.2f}%"
    )


# ── EOD close-out ─────────────────────────────────────────────────────────────

def close_scalp_positions():
    """
    Called at 15:49 (market close) — records final P&L for all open SCALP positions.
    Momentum positions are NOT closed here (they carry overnight).
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
        fees     = (config.BUY_FEE_PCT + config.SELL_FEE_PCT) * 100
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
    """Returns today's scalp + momentum watchlists with live P&L."""
    _maybe_reset()

    scalp_rows    = []
    momentum_rows = []
    scalp_pnl     = 0.0
    scalp_closed  = 0
    mom_pnl       = 0.0
    mom_closed    = 0

    # ── Scalp positions ───────────────────────────────────────────────────
    for symbol, pos in _scalp_positions.items():
        entry    = pos["entry_price"]
        exit_p   = pos.get("exit_price")

        if pos["status"] == "closed" and exit_p:
            pnl_pct = pos.get("pnl_pct", 0)
            current = exit_p
        else:
            # Fetch fresh price for live P&L — never use stale cached value
            fresh = _fetch_current_price(symbol)
            current = fresh if fresh is not None else entry
            gross   = (current - entry) / entry * 100
            fees    = (config.BUY_FEE_PCT + config.SELL_FEE_PCT) * 100
            pnl_pct = round(gross - fees, 2)

        if pos["status"] == "closed":
            scalp_closed += 1
            scalp_pnl    += pnl_pct

        scalp_rows.append({
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

    scalp_rows.sort(key=lambda x: x["pnl_pct"], reverse=True)

    # ── Momentum positions ────────────────────────────────────────────────
    for symbol, pos in _momentum_positions.items():
        entry  = pos["entry_price"]
        exit_p = pos.get("exit_price")

        if pos["status"] == "closed" and exit_p:
            pnl_pct = pos.get("pnl_pct", 0)
            current = exit_p
        else:
            fresh = _fetch_current_price(symbol)
            current = fresh if fresh is not None else entry
            gross   = (current - entry) / entry * 100
            fees    = (config.BUY_FEE_PCT + config.SELL_FEE_PCT) * 100
            pnl_pct = round(gross - fees, 2)

        if pos["status"] == "closed":
            mom_closed += 1
            mom_pnl    += pnl_pct

        entry_date = pos.get("entry_date", "")
        days_held  = (date.today() - date.fromisoformat(entry_date)).days if entry_date else 0

        momentum_rows.append({
            "symbol":       symbol,
            "entry_price":  entry,
            "current":      current,
            "pnl_pct":      pnl_pct,
            "status":       pos["status"],
            "score":        pos["score"],
            "rsi":          pos["rsi"],
            "days_held":    days_held,
            "hold_days":    pos["hold_days"],
            "target_pct":   pos["target_pct"],
            "sl_pct":       pos["stop_loss_pct"],
            "exit_reason":  pos.get("exit_reason", ""),
        })

    momentum_rows.sort(key=lambda x: x["pnl_pct"], reverse=True)

    return {
        "date":             _watchlist_date,
        # Scalp
        "scalp_total":      len(scalp_rows),
        "scalp_closed":     scalp_closed,
        "scalp_rows":       scalp_rows,
        "scalp_avg_pnl":    round(scalp_pnl / scalp_closed, 2) if scalp_closed else None,
        # Momentum
        "momentum_total":   len(momentum_rows),
        "momentum_closed":  mom_closed,
        "momentum_rows":    momentum_rows,
        "momentum_avg_pnl": round(mom_pnl / mom_closed, 2) if mom_closed else None,
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
            "rsi":         tech["rsi"]          if tech else None,
            "vol_ratio":   tech["volume_ratio"] if tech else None,
            "ma_trend":    tech["ma_trend"]     if tech else None,
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


def _format_momentum_alert(entry: dict) -> str:
    ticker = entry["symbol"].replace(".JK", "")
    target = entry["entry_price"] * (1 + entry["target_pct"] / 100)
    sl     = entry["entry_price"] * (1 + entry["sl_pct"] / 100)
    return (
        f"📈 <b>MOMENTUM CANDIDATE — {ticker}.JK</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Entry:   <b>{entry['entry_price']:,.0f}</b>\n"
        f"RSI:     {entry['rsi']:.0f} (bullish)\n"
        f"Score:   {entry['score']}/100\n"
        f"\n"
        f"🎯 Target: {target:,.0f}  (+{entry['target_pct']:.1f}%)\n"
        f"🛑 SL: {sl:,.0f}  ({entry['sl_pct']:+.1f}%)\n"
        f"📅 Hold: up to {config.MOMENTUM_HOLD_DAYS} trading days"
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


def _format_momentum_exit_alert(symbol: str, pos: dict, reason: str) -> str:
    ticker  = symbol.replace(".JK", "")
    pnl     = pos.get("pnl_pct", 0)
    emoji   = "🟢" if pnl >= 0 else "🔴"
    return (
        f"{emoji} <b>MOMENTUM EXIT — {ticker}.JK</b>  [{reason}]\n"
        f"Entry:  {pos['entry_price']:,.0f}\n"
        f"Exit:   {pos.get('exit_price', 0):,.0f}\n"
        f"P&L:    <b>{pnl:+.2f}%</b>"
    )


def format_scalp_watchlist(summary: dict) -> str:
    today = summary.get("date", "—")
    lines = [f"⚡📈 <b>WATCHLIST — {today}</b>", ""]

    # ── Scalp section ─────────────────────────────────────────────────────
    scalp_rows = summary.get("scalp_rows", [])
    avg_scalp  = summary.get("scalp_avg_pnl")
    lines.append(
        f"⚡ <b>SCALP</b>  ({summary.get('scalp_total', 0)} positions"
        + (f"  |  Avg P&L: {avg_scalp:+.2f}%" if avg_scalp is not None else "")
        + ")"
    )
    lines.append("━━━━━━━━━━━━━━━━━━━━")

    if not scalp_rows:
        lines.append("  Belum ada kandidat scalp hari ini.")
    else:
        for r in scalp_rows:
            ticker = r["symbol"].replace(".JK", "")
            pnl    = r["pnl_pct"]
            emoji  = "🟢" if pnl >= 0 else "🔴"
            status = "✅" if r["status"] == "closed" else "👀"
            lines.append(
                f"{emoji}{status} <b>{ticker}</b>"
                f"  Entry: {r['entry_price']:,.0f}"
                f"  Now: {r['current']:,.0f}"
                f"  <b>{pnl:+.2f}%</b>"
            )
            lines.append(
                f"       Drop: -{r['drop_pct']:.1f}%"
                f"  RSI: {r['rsi']:.0f}"
                f"  Score: {r['score']}"
            )

    lines.append("")

    # ── Momentum section ──────────────────────────────────────────────────
    mom_rows  = summary.get("momentum_rows", [])
    avg_mom   = summary.get("momentum_avg_pnl")
    lines.append(
        f"📈 <b>MOMENTUM</b>  ({summary.get('momentum_total', 0)} positions"
        + (f"  |  Avg P&L: {avg_mom:+.2f}%" if avg_mom is not None else "")
        + ")"
    )
    lines.append("━━━━━━━━━━━━━━━━━━━━")

    if not mom_rows:
        lines.append("  Belum ada kandidat momentum hari ini.")
    else:
        for r in mom_rows:
            ticker = r["symbol"].replace(".JK", "")
            pnl    = r["pnl_pct"]
            emoji  = "🟢" if pnl >= 0 else "🔴"
            status = "✅" if r["status"] == "closed" else "👀"
            reason = f"  [{r['exit_reason']}]" if r.get("exit_reason") else ""
            lines.append(
                f"{emoji}{status} <b>{ticker}</b>"
                f"  Entry: {r['entry_price']:,.0f}"
                f"  Now: {r['current']:,.0f}"
                f"  <b>{pnl:+.2f}%</b>{reason}"
            )
            lines.append(
                f"       Day {r['days_held']}/{r['hold_days']}"
                f"  RSI: {r['rsi']:.0f}"
                f"  🎯{r['target_pct']:+.1f}%  🛑{r['sl_pct']:+.1f}%"
            )

    return "\n".join(lines)
