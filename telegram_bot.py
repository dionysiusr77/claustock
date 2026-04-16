"""
Telegram integration — raw requests pattern (no library dependency).
Handles:
  - Sending messages / alerts
  - Pre-session briefings
  - Live signal alerts
  - Command polling (/status, /stocks, /briefing, /forecast, /flow, /help)
"""

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

def send_long_message(parts: list[str], parse_mode: str = "HTML") -> None:
    """
    Send a multi-part message with retry + delay between parts.
    Telegram rate-limits rapid sequential messages — 1.5s gap prevents drops.
    """
    import time
    for i, part in enumerate(parts, 1):
        prefix = f"<i>({i}/{len(parts)})</i>\n" if len(parts) > 1 else ""
        text   = prefix + part

        # Retry up to 3 times per part
        for attempt in range(1, 4):
            ok = send_message(text, parse_mode=parse_mode)
            if ok:
                break
            logger.warning(f"send_long_message part {i}/{len(parts)} attempt {attempt} failed, retrying...")
            time.sleep(2 * attempt)
        else:
            logger.error(f"send_long_message: part {i}/{len(parts)} failed after 3 attempts")

        # Delay between parts to avoid Telegram rate limit (30 msg/sec per chat)
        if i < len(parts):
            time.sleep(1.5)


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured — skipping send")
        return False
    try:
        resp = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={
                "chat_id":    config.TELEGRAM_CHAT_ID,
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

    lines = [
        f"📡 <b>SIGNAL — {ticker}.JK</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"💰 Price: <b>{price:,.0f}</b>",
        f"📊 Score: <b>{score}/100 → {verdict}</b>",
        "",
        f"Technical:    <b>{t_score}/35</b>  RSI {rsi:.0f}, MA {ma_trend}, vol {vol_ratio:.1f}x",
        f"Forecast:     <b>{p_score}/25</b>"
        + (f"  {trend_pct:+.1f}% trend next 5d" if trend_pct is not None else ""),
        f"Foreign flow: <b>{f_score}/20</b>"
        + (f"  Net buy {flow_days}d in a row" if flow_days and flow_days > 0 else ""),
        f"News:         <b>{n_score}/20</b>  {SENTIMENT_EMOJI.get(sentiment, '')} {headline[:60] if headline else 'No major news'}",
    ] + ai_block
    return "\n".join(lines)


def format_whale_alert(symbol: str, vol_ratio: float, score: int, auto_added: bool) -> str:
    ticker = symbol.replace(".JK", "")
    action = "Added to watchlist" if auto_added else "Not added (below score threshold)"
    return (
        f"🐋 <b>WHALE DETECTED — {ticker}.JK</b>\n"
        f"Volume surge: <b>{vol_ratio:.1f}x</b> average\n"
        f"Tech score: {score}/35\n"
        f"{action}"
    )


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
) -> str:
    """Format the pre-session briefing message."""
    session_label = f"SESSION {session} BRIEFING"
    now_wib = datetime.now(WIB).strftime("%a %d %b %Y")

    lines = [
        f"📋 <b>{session_label} — {now_wib}</b>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    # Market context
    if jci:
        chg = jci.get("jci_change_pct", 0)
        chg_str = f"{chg:+.1f}%" if chg else "N/A"
        lines += [
            "🌍 <b>Market Context</b>",
            f"  JCI: {jci.get('jci_close', 'N/A')} ({chg_str})",
            "",
        ]

    # Forecast summary
    forecasts = [s for s in stock_scores if s.get("trend_pct") is not None]
    if forecasts:
        lines.append("📈 <b>5-Day Forecast</b>")
        for s in forecasts[:5]:
            ticker    = s["symbol"].replace(".JK", "")
            cur       = s.get("price", 0)
            f5d       = s.get("forecast_5d", cur)
            pct       = s.get("trend_pct", 0)
            trend_e   = TREND_EMOJI.get(s.get("trend", "FLAT"), "⚪")
            lines.append(f"  {ticker:5s}  {cur:>8,.0f} → {f5d:>8,.0f}  ({pct:+.1f}%)  {trend_e}")
        lines.append("")

    # Top picks
    picks = [s for s in stock_scores if s.get("total_score", 0) >= config.IDX_MIN_SCORE]
    picks.sort(key=lambda x: x.get("total_score", 0), reverse=True)

    if picks:
        lines.append(f"🎯 <b>Top Picks — Session {session}</b>")
        for i, s in enumerate(picks[:3], 1):
            ticker  = s["symbol"].replace(".JK", "")
            price   = s.get("price", 0)
            score   = s.get("total_score", 0)
            verdict = s.get("verdict", "")
            t_pct   = config.MIN_TARGET_PCT
            target  = round(price * (1 + t_pct / 100))
            sl      = round(price * (1 - 1.2 / 100))
            headline = s.get("news_headline", "")
            lines.append(
                f"  {i}. <b>{ticker}</b> — Entry {price:,.0f} | Target {target:,.0f} | SL {sl:,.0f}"
            )
            lines.append(f"     Score: {score}/100 | {verdict}")
            if headline:
                lines.append(f"     📰 {headline[:70]}")
        lines.append("")

    # Avoid list
    avoids = [s["symbol"].replace(".JK", "") for s in stock_scores
              if s.get("verdict") in ("SKIP",) and s.get("total_score", 100) < 40]
    if avoids:
        lines.append(f"⚠️ <b>Avoid:</b> {', '.join(avoids)}")

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
    Long-polls Telegram for commands in a background thread.
    Calls handler functions injected at construction time.
    """

    def __init__(self, handlers: dict):
        """
        handlers: dict mapping command string → callable
        e.g. { "/status": fn, "/stocks": fn }
        """
        self.handlers  = handlers
        self.offset    = 0
        self._stop     = threading.Event()
        self._thread   = None

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
        msg = update.get("message") or update.get("edited_message")
        if not msg:
            return
        text = (msg.get("text") or "").strip()
        if not text.startswith("/"):
            return

        # Extract base command (ignore args for routing)
        cmd   = text.split()[0].lower().split("@")[0]
        args  = text.split()[1:]

        handler = self.handlers.get(cmd)
        if handler:
            try:
                handler(args)
            except Exception as e:
                logger.error(f"Command handler {cmd} error: {e}")
                send_message(f"❌ Error handling {cmd}: {e}")
        else:
            send_message(f"Unknown command: {cmd}\nType /help for available commands.")
