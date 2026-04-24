"""
Telegram integration — raw requests pattern (no library dependency).
Handles:
  - Sending messages / alerts
  - Pre-session briefings
  - Live signal alerts
  - Command polling (/status, /stocks, /briefing, /forecast, /flow, /help)
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

import requests

import config
from scheduler import WIB

logger = logging.getLogger(__name__)

TELEGRAM_API = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}"

# Emoji map
VERDICT_EMOJI = {
    "STRONG_BUY": "🚀",
    "BUY":        "📈",
    "WATCH":      "👀",
    "SKIP":       "⛔",
}
TREND_EMOJI = {"UP": "🟢", "DOWN": "🔴", "FLAT": "⚪"}
SENTIMENT_EMOJI = {"POSITIVE": "✅", "NEUTRAL": "➖", "NEGATIVE": "⚠️"}


# ── Core send ─────────────────────────────────────────────────────────────────

def send_long_message(
    parts: list[str],
    parse_mode: str = "HTML",
    chat_id: str | int | None = None,
) -> None:
    """
    Send a multi-part message with retry + delay between parts.
    Telegram rate-limits rapid sequential messages — 1.5s gap prevents drops.
    chat_id: override destination (defaults to config.TELEGRAM_CHAT_ID).
    """
    import time
    for i, part in enumerate(parts, 1):
        prefix = f"<i>({i}/{len(parts)})</i>\n" if len(parts) > 1 else ""
        text   = prefix + part

        for attempt in range(1, 4):
            ok = send_message(text, parse_mode=parse_mode, chat_id=chat_id)
            if ok:
                break
            logger.warning(f"send_long_message part {i}/{len(parts)} attempt {attempt} failed, retrying...")
            time.sleep(2 * attempt)
        else:
            logger.error(f"send_long_message: part {i}/{len(parts)} failed after 3 attempts")

        if i < len(parts):
            time.sleep(1.5)


def send_message_with_keyboard(
    text: str,
    keyboard: list[list[dict]],
    parse_mode: str = "HTML",
    chat_id: str | int | None = None,
) -> bool:
    """Send a message with an inline keyboard (buttons)."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return False
    target = chat_id if chat_id is not None else config.TELEGRAM_CHAT_ID
    try:
        resp = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={
                "chat_id":      target,
                "text":         text,
                "parse_mode":   parse_mode,
                "reply_markup": {"inline_keyboard": keyboard},
            },
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"send_message_with_keyboard failed: {e}")
        return False


def answer_callback_query(callback_query_id: str, text: str = "") -> None:
    """Acknowledge a button press (removes the loading spinner)."""
    try:
        requests.post(
            f"{TELEGRAM_API}/answerCallbackQuery",
            json={"callback_query_id": callback_query_id, "text": text},
            timeout=5,
        )
    except Exception as e:
        logger.warning(f"answer_callback_query failed: {e}")


def edit_message_text(
    chat_id: str | int,
    message_id: int,
    text: str,
    parse_mode: str = "HTML",
) -> None:
    """Replace the text of an existing message (used to update confirm prompts)."""
    try:
        requests.post(
            f"{TELEGRAM_API}/editMessageText",
            json={
                "chat_id":    chat_id,
                "message_id": message_id,
                "text":       text,
                "parse_mode": parse_mode,
            },
            timeout=5,
        )
    except Exception as e:
        logger.warning(f"edit_message_text failed: {e}")


def send_message(
    text: str,
    parse_mode: str = "HTML",
    chat_id: str | int | None = None,
) -> bool:
    """
    Send a message to chat_id (defaults to config.TELEGRAM_CHAT_ID).
    Pass chat_id explicitly to reply to a specific user or group.
    """
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured — skipping send")
        return False
    target = chat_id if chat_id is not None else config.TELEGRAM_CHAT_ID
    try:
        resp = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={
                "chat_id":    target,
                "text":       text,
                "parse_mode": parse_mode,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"send_message failed: {e}")
        return False


# ── Message formatters ────────────────────────────────────────────────────────

def format_signal_with_ai(symbol: str, snapshot: dict, ai: dict | None) -> str:
    """Format a live signal alert matching the spec in BRIEF.md."""
    ticker  = symbol.replace(".JK", "")
    price   = snapshot.get("price", 0)
    score   = snapshot.get("total_score", 0)
    verdict = snapshot.get("verdict", "")
    emoji   = VERDICT_EMOJI.get(verdict, "📊")

    t_score = snapshot.get("technical_score", 0)
    p_score = snapshot.get("prophet_score", 0)
    f_score = snapshot.get("foreign_score", 0)
    n_score = snapshot.get("news_score", 0)

    rsi       = snapshot.get("rsi", 0)
    ma_trend  = snapshot.get("ma_trend", "")
    vol_ratio = snapshot.get("volume_ratio", 0)
    trend_pct = snapshot.get("trend_pct")
    flow_days = snapshot.get("days_consecutive")
    headline  = snapshot.get("news_headline", "")
    sentiment = snapshot.get("news_sentiment", "NEUTRAL")

    # Entry / target / SL — simple % based calculation
    target_pct = max(config.MIN_TARGET_PCT, 2.0)
    sl_pct     = 1.2
    target     = round(price * (1 + target_pct / 100))
    sl         = round(price * (1 - sl_pct / 100))
    rr         = round(target_pct / sl_pct, 1)

    # Use AI verdict if available, otherwise fall back to rule-based
    if ai and ai.get("action") == "ENTER":
        entry  = ai.get("entry_price", price)
        target = ai.get("target_price", round(price * 1.02))
        sl     = ai.get("stop_loss",    round(price * 0.988))
        t_pct  = ai.get("target_pct",   2.0)
        sl_pct = ai.get("stop_loss_pct", 1.2)
        rr     = ai.get("risk_reward",   round(t_pct / sl_pct, 1))
        lots   = ai.get("lots", 1)
        cap    = ai.get("capital_idr", lots * 100 * price)
        conf   = ai.get("confidence", 0)
        hold   = ai.get("hold_duration", "intraday")
        reason = ai.get("reasoning", "")
        ai_block = [
            "",
            f"🤖 <b>AI Verdict: {ai['action']}</b> ({conf}% confidence)",
            f"🚀 Entry:   <b>{entry:,.0f}</b>",
            f"✅ Target:  <b>{target:,.0f}</b>  (+{t_pct:.1f}%)",
            f"🛑 SL:      <b>{sl:,.0f}</b>  (-{sl_pct:.1f}%)",
            f"⚖️ R/R:     1:{rr}",
            f"💼 Lots:    {lots} (~Rp{cap/1_000_000:.1f}M)",
            f"⏱ Hold:    {hold}",
            f"💬 {reason}",
            f"⚠️ T+2 settlement — capital locked 2 days after sell",
        ]
    else:
        ai_action = ai.get("action", "WAIT") if ai else "—"
        ai_block = [
            "",
            f"🤖 <b>AI Verdict: {ai_action}</b>",
            f"🚀 Entry:   <b>{price:,.0f}</b>",
            f"✅ Target:  <b>{round(price * (1 + target_pct / 100)):,.0f}</b>  (+{target_pct:.1f}%)",
            f"🛑 SL:      <b>{round(price * (1 - sl_pct / 100)):,.0f}</b>  (-{sl_pct:.1f}%)",
            f"⚖️ R/R:     1:{rr}",
            f"⚠️ T+2 settlement — capital locked 2 days after sell",
        ]

    vwap      = snapshot.get("vwap")
    vwap_pct  = snapshot.get("vwap_pct")
    vwap_line = ""
    if vwap and vwap_pct is not None:
        vwap_e    = "🔴" if vwap_pct < -1 else ("🟡" if vwap_pct < 0 else "🟢")
        vwap_line = f"  VWAP {vwap:,.0f} {vwap_e}{vwap_pct:+.1f}%"

    lines = [
        f"📡 <b>SIGNAL — {ticker}.JK</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"💰 Price: <b>{price:,.0f}</b>{vwap_line}",
        f"📊 Score: <b>{score}/100 → {verdict}</b>",
        "",
        f"Technical:    <b>{t_score}/40</b>  RSI {rsi:.0f}, MA {ma_trend}, vol {vol_ratio:.1f}x",
        f"Forecast:     <b>{p_score}/25</b>"
        + (f"  {trend_pct:+.1f}% trend next 5d" if trend_pct is not None else ""),
        f"Foreign flow: <b>{f_score}/20</b>"
        + (f"  Net buy {flow_days}d in a row" if flow_days and flow_days > 0 else ""),
        f"News:         <b>{n_score}/20</b>  {SENTIMENT_EMOJI.get(sentiment, '')} {headline[:60] if headline else 'No major news'}",
    ] + ai_block
    return "\n".join(lines)


def format_whale_alert(symbol: str, vol_ratio: float, score: int, auto_added: bool) -> str:
    ticker = symbol.replace(".JK", "")
    action = "✅ Added to watchlist" if auto_added else "⏭ Not added (below score threshold)"
    return (
        f"🐋 <b>WHALE DETECTED — {ticker}.JK</b>\n"
        f"Volume surge: <b>{vol_ratio:.1f}×</b> average\n"
        f"Tech score: {score}/35\n"
        f"{action}"
    )


def format_whale_confirm(symbol: str, vol_ratio: float, score: int) -> tuple[str, list]:
    """Returns (message_text, inline_keyboard) for a manual-confirm whale prompt."""
    ticker = symbol.replace(".JK", "")
    text = (
        f"🐋 <b>WHALE SURGE — {ticker}.JK</b>\n"
        f"Volume: <b>{vol_ratio:.1f}×</b> above average\n"
        f"Tech score: {score}/35\n\n"
        f"Add <b>{ticker}</b> to the intraday watchlist?"
    )
    keyboard = [[
        {"text": "✅ Add to watchlist", "callback_data": f"whale_add_{symbol}"},
        {"text": "❌ Skip",             "callback_data": f"whale_skip_{symbol}"},
    ]]
    return text, keyboard


def format_pnl(pnl: dict) -> str:
    rows      = pnl.get("rows", [])
    realized  = pnl.get("realized_pnl_idr", 0)
    unrealized = pnl.get("unrealized_pnl_idr", 0)
    total     = pnl.get("total_pnl_idr", 0)
    capital   = pnl.get("capital", 1)
    count     = pnl.get("signals_count", 0)
    today     = datetime.now(WIB).strftime("%a %d %b %Y")

    total_pct = total / capital * 100

    lines = [
        f"📊 <b>P&L HARI INI — {today}</b>",
        f"Sinyal fired: <b>{count}</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    if rows:
        lines.append("")
        for r in rows:
            ticker = r["symbol"].replace(".JK", "")
            pct    = r["pnl_pct"]
            pnl_k  = r["pnl_idr"] / 1000
            emoji  = "🟢" if pct >= 0 else "🔴"
            lines.append(
                f"{emoji} <b>{ticker}</b>  {r['entry']:,.0f} → {r['current']:,.0f}"
                f"  ({pct:+.1f}%)  {pnl_k:+.0f}K IDR  x{r['lots']} lot"
            )

    lines += [
        "",
        f"Realized:   <b>{realized/1000:+.0f}K IDR</b>",
        f"Unrealized: <b>{unrealized/1000:+.0f}K IDR</b>",
        f"Total:      <b>{total/1000:+.0f}K IDR</b>  ({total_pct:+.2f}%)",
    ]

    if total < 0 and abs(total_pct) >= 2.0:
        lines.append("⚠️ Mendekati daily loss limit!")

    return "\n".join(lines)


def format_watchlist_change(symbol: str, action: str, watchlist: list[str]) -> str:
    ticker = symbol.replace(".JK", "")
    verb   = "Added" if action == "add" else "Removed"
    stocks = ", ".join(s.replace(".JK", "") for s in watchlist)
    return (
        f"✅ <b>{verb}: {ticker}.JK</b>\n"
        f"Watchlist ({len(watchlist)}): {stocks}"
    )


def format_presession_briefing(
    session: int,
    date_str: str,
    jci: dict | None,
    stock_scores: list[dict],
    scalp_summary: dict | None = None,
    scalp_candidates: list[dict] | None = None,
) -> str:
    """
    Pre-session briefing.

    The D-1 screened candidates are the single source of truth for "today's watchlist".
    5-day forecast data is pulled from stock_scores for those same stocks.
    The old general forecast / top-picks sections are removed.
    """
    now_wib = datetime.now(WIB).strftime("%a %d %b %Y")

    lines = [
        f"📋 <b>SESSION {session} BRIEFING — {now_wib}</b>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    # ── Market context ────────────────────────────────────────────────────
    if jci:
        chg    = jci.get("jci_change_pct", 0) or 0
        chg_e  = "🟢" if chg > 0 else ("🔴" if chg < 0 else "⚪")
        net_b  = (jci.get("total_foreign_net_idr") or 0) / 1_000_000_000
        flow_e = "🟢" if net_b > 0 else ("🔴" if net_b < 0 else "⚪")
        lines += [
            f"🌍 <b>Market</b>  JCI {jci.get('jci_close', 'N/A')} {chg_e}{chg:+.1f}%"
            f"  |  Foreign net {flow_e}{net_b:+.2f}B IDR",
            "",
        ]

    # ── Today's watchlist (D-1 qualified stocks) ──────────────────────────
    # Build a symbol → stock_score lookup for forecast enrichment
    score_map = {s["symbol"]: s for s in stock_scores}

    lines.append(f"📋 <b>Today's Watchlist — Session {session}</b>")
    lines.append("━━━━━━━━━━━━━━━━━━━━")

    _MACD_LABEL = {
        "CROSS":       "Bullish cross ✅",
        "APPROACHING": "Approaching cross ✅",
        "BULLISH":     "Bullish ✅",
        "BEARISH":     "Bearish ❌",
    }

    if scalp_candidates:
        for c in scalp_candidates:
            ticker  = c["symbol"].replace(".JK", "")
            rsi_dir = "↑" if c["rsi_today"] > c["rsi_7d_avg"] else "↓"
            is_s1   = c.get("is_s1", False)

            # 5-day forecast from live scan data (may be None before first scan)
            sc      = score_map.get(c["symbol"], {})
            cur     = sc.get("price") or c["close_d1"]
            f5d     = sc.get("forecast_5d")
            t_pct   = sc.get("trend_pct")
            trend_e = TREND_EMOJI.get(sc.get("trend", "FLAT"), "⚪")

            if f5d and t_pct is not None:
                forecast_line = (
                    f"   📈 5D Forecast: {cur:,.0f} → {f5d:,.0f}"
                    f"  ({t_pct:+.1f}%) {trend_e}"
                )
            else:
                forecast_line = "   📈 5D Forecast: pending first scan"

            if is_s1:
                drop_e     = "🔴" if c["drop_from_open_pct"] < 0 else "🟢"
                price_line = (
                    f"   S1 Close:     {c['close_d1']:,.0f}"
                    f"  (Open: {c['open_price']:,.0f},"
                    f" {drop_e}{c['drop_from_open_pct']:+.1f}%)"
                )
                score_label = "S1 Score"
                rsi_ref     = f"avg last 5 S1: {c['rsi_7d_avg']}"
                vol_label   = "Volume S1:"
                vol_ref     = "avg S1 last week"
            else:
                price_line  = f"   Close D-1:    {c['close_d1']:,.0f}"
                score_label = "D-1 Score"
                rsi_ref     = f"avg 7d: {c['rsi_7d_avg']}"
                vol_label   = "Volume:    "
                vol_ref     = "avg 7d"

            lines += [
                f"📌 <b>{ticker}</b>  |  {score_label}: {c['total_score']}/100",
                price_line,
                forecast_line,
                f"   RSI:          {c['rsi_today']}  ({rsi_ref}) ✅ {rsi_dir}",
                f"   {vol_label}   {c['vol_ratio']}x {vol_ref} ✅",
                f"   MACD:         {_MACD_LABEL.get(c['macd_status'], c['macd_status'])}",
                f"   BB Position:  {c['bb_position_pct']}%"
                + (" ✅" if c.get("bb_pass") else " ➖ <i>(optional)</i>"),
                f"",
                f"   🎯 Target:  {c['target_price']:,.0f} ({c['target_pct']:+.1f}%) → {c['target_label']}",
                f"   🛑 SL:      {c['stop_loss_price']:,.0f} ({c['stop_loss_pct']:+.1f}%)",
                f"   💬 {c['note']}",
                f"",
            ]
    else:
        lines += [
            "⚠️ No stocks passed all 4 criteria today.",
            "<i>Check back at Session 2 briefing.</i>",
            "",
        ]

    # ── Active scalp / momentum positions ─────────────────────────────────
    if scalp_summary:
        s_total = scalp_summary.get("scalp_total", 0)
        m_total = scalp_summary.get("momentum_total", 0)
        if s_total or m_total:
            lines += [
                f"⚡ <b>Active positions</b>  Scalp: {s_total}  |  Momentum: {m_total}"
                f"  → /scalps",
                "",
            ]

    return "\n".join(lines)


def format_eod_summary(stock_scores: list[dict], signals_fired: list[dict]) -> str:
    now_wib = datetime.now(WIB).strftime("%a %d %b %Y")
    lines = [
        f"🏁 <b>END OF DAY — {now_wib}</b>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"Signals fired today: <b>{len(signals_fired)}</b>",
        "",
        "📊 <b>Final Scores</b>",
    ]
    for s in sorted(stock_scores, key=lambda x: x.get("total_score", 0), reverse=True):
        ticker  = s["symbol"].replace(".JK", "")
        score   = s.get("total_score", 0)
        verdict = s.get("verdict", "")
        emoji   = VERDICT_EMOJI.get(verdict, "➖")
        lines.append(f"  {ticker:5s}  {score:>3d}/100  {emoji} {verdict}")

    lines += ["", "See you tomorrow 👋"]
    return "\n".join(lines)


def format_status(is_open: bool, is_trading: bool, stock_count: int) -> str:
    now_wib = datetime.now(WIB).strftime("%H:%M WIB")
    status  = "🟢 OPEN" if is_open else "🔴 CLOSED"
    return (
        f"🤖 <b>IDX Bot Status</b>\n"
        f"Time: {now_wib}\n"
        f"Market: {status}\n"
        f"Trading day: {'Yes' if is_trading else 'No'}\n"
        f"Watchlist: {stock_count} stocks"
    )


def format_stocks_list(stock_scores: list[dict]) -> str:
    lines = ["📋 <b>Current Watchlist Scores</b>", ""]
    for s in sorted(stock_scores, key=lambda x: x.get("total_score", 0), reverse=True):
        ticker  = s["symbol"].replace(".JK", "")
        score   = s.get("total_score", 0)
        price   = s.get("price", 0)
        verdict = s.get("verdict", "—")
        emoji   = VERDICT_EMOJI.get(verdict, "➖")
        lines.append(f"{emoji} <b>{ticker}</b>  {score}/100  {price:,.0f}")
    return "\n".join(lines)


# ── Command polling ───────────────────────────────────────────────────────────

class CommandPoller:
    """
    Long-polls Telegram for commands and inline keyboard callbacks in a background thread.

    handlers:          { "/command": fn(args, chat_id) }
    callback_handlers: { "prefix_": fn(data, callback_query_id, message_id, chat_id) }
                       Matched by prefix — first match wins.
    """

    def __init__(self, handlers: dict, callback_handlers: dict | None = None):
        self.handlers          = handlers
        self.callback_handlers = callback_handlers or {}
        self.offset            = 0
        self._stop             = threading.Event()
        self._thread           = None

    def start(self):
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("Telegram command poller started")

    def stop(self):
        self._stop.set()

    def _poll_loop(self):
        while not self._stop.is_set():
            try:
                resp = requests.get(
                    f"{TELEGRAM_API}/getUpdates",
                    params={"offset": self.offset, "timeout": 30},
                    timeout=35,
                )
                updates = resp.json().get("result", [])
                for update in updates:
                    self.offset = update["update_id"] + 1
                    self._handle_update(update)
            except Exception as e:
                logger.warning(f"Telegram poll error: {e}")
                time.sleep(5)

    def _handle_update(self, update: dict):
        # Inline keyboard button press
        if "callback_query" in update:
            self._handle_callback(update["callback_query"])
            return

        msg = update.get("message") or update.get("edited_message")
        if not msg:
            return

        chat_id = msg["chat"]["id"]
        user_id = msg.get("from", {}).get("id")

        # Allow if sender's user ID or the chat itself is in the allowlist
        allowed = config.TELEGRAM_ALLOWED_USER_IDS
        if allowed and user_id not in allowed and chat_id not in allowed:
            logger.info(f"Blocked: user_id={user_id} chat_id={chat_id} — not in allowlist {allowed}")
            return

        text = (msg.get("text") or "").strip()
        if not text.startswith("/"):
            return

        # Strip @BotName suffix (required in groups: /cmd@BotName)
        cmd  = text.split()[0].lower().split("@")[0]
        args = text.split()[1:]

        handler = self.handlers.get(cmd)
        if handler:
            try:
                handler(args, chat_id=chat_id)
            except Exception as e:
                logger.error(f"Command handler {cmd} error: {e}")
                send_message(f"❌ Error handling {cmd}: {e}", chat_id=chat_id)
        else:
            send_message(
                f"Unknown command: {cmd}\nType /help for available commands.",
                chat_id=chat_id,
            )

    def _handle_callback(self, cq: dict):
        user_id    = cq.get("from", {}).get("id")
        cq_msg     = cq.get("message", {})
        chat_id    = cq_msg.get("chat", {}).get("id")
        message_id = cq_msg.get("message_id")
        data       = cq.get("data", "")

        allowed = config.TELEGRAM_ALLOWED_USER_IDS
        if allowed and user_id not in allowed and chat_id not in allowed:
            answer_callback_query(cq["id"], "Not authorized.")
            return

        for prefix, handler in self.callback_handlers.items():
            if data.startswith(prefix):
                try:
                    handler(data, cq["id"], message_id, chat_id)
                except Exception as e:
                    logger.error(f"Callback handler '{prefix}' error: {e}")
                    answer_callback_query(cq["id"], "Error processing request.")
                return

        answer_callback_query(cq["id"])
