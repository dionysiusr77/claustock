"""
Telegram bot — commands and broadcaster.
Uses python-telegram-bot v20 async architecture.

Commands:
  /start    — welcome
  /help     — command list
  /briefing — send latest saved briefing (or trigger fresh if none today)
  /scan     — trigger live D-1 scan now (admin only)
  /pick BBCA — deep-dive score for one stock
  /status   — bot status + market hours check
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytz
from telegram import Bot, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import TelegramError

import config

logger = logging.getLogger(__name__)

_WIB      = pytz.timezone(config.MARKET_TZ)
_executor = ThreadPoolExecutor(max_workers=2)   # for blocking scan calls


# ── Auth guard ────────────────────────────────────────────────────────────────

def _allowed(update: Update) -> bool:
    """Only respond to configured chat IDs."""
    if not config.TELEGRAM_CHAT_IDS:
        return True   # open if no list configured
    return str(update.effective_chat.id) in config.TELEGRAM_CHAT_IDS


# ── Broadcaster ───────────────────────────────────────────────────────────────

async def broadcast(bot: Bot, text: str, parse_mode: str = ParseMode.HTML) -> None:
    """Send a message to all configured chat IDs. Silently skips failed sends."""
    if not config.TELEGRAM_CHAT_IDS:
        logger.warning("TELEGRAM_CHAT_IDS not configured — nowhere to broadcast")
        return

    for chat_id in config.TELEGRAM_CHAT_IDS:
        try:
            # Telegram max message length is 4096 chars
            if len(text) <= 4096:
                await bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
            else:
                # Split on double-newline boundaries
                chunks = _split_message(text, 4000)
                for chunk in chunks:
                    await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=parse_mode)
        except TelegramError as e:
            logger.error("Failed to send to %s: %s", chat_id, e)


def _split_message(text: str, limit: int) -> list[str]:
    """Split long text into chunks, preserving paragraph breaks."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > limit:
            if current:
                chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para
    if current:
        chunks.append(current.strip())
    return chunks


async def _reply_long(update: Update, text: str, parse_mode: str = ParseMode.HTML) -> None:
    """reply_text with automatic splitting for messages > 4096 chars."""
    if len(text) <= 4096:
        await update.message.reply_text(text, parse_mode=parse_mode)
    else:
        for chunk in _split_message(text, 4000):
            await update.message.reply_text(chunk, parse_mode=parse_mode)


# ── Market hours helper ───────────────────────────────────────────────────────

def _market_status() -> str:
    now = datetime.now(_WIB)
    if now.weekday() >= 5:
        return "CLOSED (weekend)"
    h, m = now.hour, now.minute
    s1 = config.SESSION1_START[0] * 60 + config.SESSION1_START[1]
    s1e = config.SESSION1_END[0] * 60 + config.SESSION1_END[1]
    s2 = config.SESSION2_START[0] * 60 + config.SESSION2_START[1]
    s2e = config.SESSION2_END[0] * 60 + config.SESSION2_END[1]
    cur = h * 60 + m
    if s1 <= cur < s1e:
        return "OPEN — Sesi 1"
    if s2 <= cur < s2e:
        return "OPEN — Sesi 2"
    if s1e <= cur < s2:
        return "JEDA siang"
    if cur >= s2e:
        return "CLOSED (after market)"
    return "PRE-MARKET"


# ── Command handlers ──────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        return
    await update.message.reply_text(
        "<b>📊 Claustock IDX v2</b>\n\n"
        "Bot analisa saham IDX berbasis D-1 scan.\n\n"
        "Ketik /help untuk daftar perintah.",
        parse_mode=ParseMode.HTML,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        return
    text = (
        "<b>Perintah tersedia:</b>\n\n"
        "/briefing — kirim morning briefing hari ini\n"
        "/midday   — briefing pre-Sesi 2 (data Sesi 1)\n"
        "/scan     — jalankan D-1 scan sekarang\n"
        "/pick BBCA — analisa satu saham\n"
        "/status   — status bot dan market\n"
        "/help     — daftar perintah ini"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        return
    now    = datetime.now(_WIB).strftime("%H:%M WIB")
    mstat  = _market_status()
    text   = (
        f"<b>Status Bot</b>\n\n"
        f"🕐 Waktu: {now}\n"
        f"📈 Market: {mstat}\n"
        f"🌐 Universe: {config.UNIVERSE}\n"
        f"💰 Capital: Rp {config.CAPITAL_IDR:,.0f}\n"
        f"📊 Min score: {config.MIN_SCORE}"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def cmd_briefing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send latest briefing from Firestore, or trigger a fresh scan if nothing saved today."""
    if not _allowed(update):
        return

    await update.message.reply_text("⏳ Mengambil briefing...")

    from firestore_client import load_latest_briefing
    text = load_latest_briefing()

    if text:
        await _reply_long(update, text)
    else:
        await update.message.reply_text(
            "Belum ada briefing hari ini. Gunakan /scan untuk generate baru."
        )


async def cmd_midday(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send pre-Sesi 2 midday briefing on demand."""
    if not _allowed(update):
        return

    await update.message.reply_text("⏳ Menyiapkan briefing Sesi 2...")

    loop = asyncio.get_running_loop()
    try:
        text = await loop.run_in_executor(_executor, _build_midday_briefing)
        if text:
            await _reply_long(update, text)
        else:
            await update.message.reply_text(
                "Belum ada data untuk midday briefing. Pastikan morning scan sudah dijalankan."
            )
    except Exception as e:
        logger.exception("cmd_midday error")
        await update.message.reply_text(f"❌ Midday briefing gagal: {e}")


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Trigger a live D-1 scan. Runs in a thread to avoid blocking the event loop."""
    if not _allowed(update):
        return

    await update.message.reply_text("🔍 Scan dimulai, tunggu sebentar...")

    loop = asyncio.get_running_loop()
    try:
        scan_data = await loop.run_in_executor(_executor, _run_full_scan)
        briefing_text = await loop.run_in_executor(_executor, _build_briefing, scan_data)
        await _reply_long(update, briefing_text)
    except Exception as e:
        logger.exception("cmd_scan error")
        await update.message.reply_text(f"❌ Scan gagal: {e}")


async def cmd_pick(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Deep-dive score for a single stock. Usage: /pick BBCA"""
    if not _allowed(update):
        return

    if not context.args:
        await update.message.reply_text("Usage: /pick BBCA")
        return

    raw    = context.args[0].upper().strip()
    symbol = raw if raw.endswith(".JK") else f"{raw}.JK"
    await update.message.reply_text(f"🔍 Menganalisa {symbol}...")

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(_executor, _run_single_pick, symbol)
        if result is None:
            await update.message.reply_text(f"❌ Tidak ada data untuk {symbol}")
            return
        await _reply_long(update, _format_single_pick(result))
    except Exception as e:
        logger.exception("cmd_pick error for %s", symbol)
        await update.message.reply_text(f"❌ Error: {e}")


# ── Blocking helpers (run in executor) ───────────────────────────────────────

def _run_full_scan() -> dict:
    from screener import run_scan
    return run_scan()


def _build_briefing(scan_data: dict) -> str:
    from market_breadth import build_breadth_summary
    from ai_briefing import build_briefing
    from agents import run_briefing_pipeline
    from firestore_client import save_scan, save_briefing

    breadth_summary = build_breadth_summary(
        scan_data.get("market", {}),
        scan_data.get("all_scored") or scan_data.get("candidates", []),
        scan_data.get("foreign_market"),
    )

    # Build market context from available breadth data
    ihsg_data  = (scan_data.get("market") or {}).get("IHSG") or {}
    market_ctx = {
        "ihsg_change":    ihsg_data.get("change_pct", 0),
        "wall_st_change": 0,
        "usd_idr":        "—",
        "macro_note":     "",
    }

    # Run 3-agent pipeline for each candidate, attach results, re-rank
    pipeline_map: dict[str, dict] = {}
    candidates = scan_data.get("candidates", [])
    for cand in candidates:
        sym = cand["symbol"]
        try:
            pr = run_briefing_pipeline(
                symbol     = sym,
                score_data = cand,
                news_raw   = [],
                market_ctx = market_ctx,
            )
            cand["pipeline_result"] = pr
            pipeline_map[sym]                    = pr
            pipeline_map[sym.replace(".JK", "")] = pr
            if pr["pipeline_ok"] and pr["conclusion"]:
                cand["final_score"] = pr["conclusion"]["final_score"]
        except Exception:
            logger.exception("Pipeline failed for %s — skipping", sym)

    # Re-sort by pipeline final_score when available, fall back to total_score
    candidates.sort(
        key=lambda c: c.get("final_score") or c.get("total_score", 0),
        reverse=True,
    )

    text = build_briefing(scan_data, breadth_summary, pipeline_map)
    save_scan(scan_data)
    save_briefing(text)
    return text


def _build_midday_briefing() -> str | None:
    from firestore_client import load_latest_scan, save_midday_briefing
    from invezgo_client import fetch_intraday_batch, fetch_intraday_sesi1
    from ai_briefing import build_midday_briefing

    scan_data = load_latest_scan()
    if not scan_data:
        logger.warning("midday: no saved scan found")
        return None

    candidates = scan_data.get("candidates", [])
    if not candidates:
        logger.warning("midday: saved scan has no candidates")
        return None

    symbols     = [c["symbol"] for c in candidates]
    prev_closes = {c["symbol"]: (c.get("snapshot") or {}).get("close") for c in candidates}
    sesi1_map   = fetch_intraday_batch(symbols, prev_closes)

    # IHSG Sesi 1 — uses /analysis/intraday-index/COMPOSITE endpoint
    ihsg_sesi1 = fetch_intraday_sesi1("COMPOSITE", prev_close=None, kind="index")

    text = build_midday_briefing(candidates, sesi1_map, ihsg_sesi1)
    if text:
        save_midday_briefing(text)
    return text


def _run_single_pick(symbol: str) -> dict | None:
    from screener import scan_single
    return scan_single(symbol)


def _format_single_pick(result: dict) -> str:
    sym    = result["symbol"]
    score  = result["total_score"]
    verdict = result["verdict"]
    setup  = result["setup"]
    div    = result.get("divergence", "NONE")
    snap   = result.get("snapshot", {})
    levels = result.get("trade_levels") or {}
    warns  = result.get("bearish_warnings", [])

    rsi    = snap.get("rsi")
    vol_r  = snap.get("vol_ratio")
    close  = snap.get("close")
    atr_p  = snap.get("atr_pct")

    layers = result.get("layer_scores", {})
    reasons = result.get("reasons", {})

    lines = [
        f"<b>🔍 {sym} — {verdict}</b>  <i>(skor {score}/100)</i>",
        f"Setup: {setup} | Divergensi: {div}",
        "",
        f"<b>Harga:</b> {close:,.0f}  |  RSI: {rsi:.1f}  |  Vol ratio: {vol_r:.1f}×  |  ATR: {atr_p:.1f}%" if all(v is not None for v in (close, rsi, vol_r, atr_p)) else "",
        "",
        "<b>Layer scores:</b>",
        f"  Trend {layers.get('trend',0)}/25 · Momentum {layers.get('momentum',0)}/20 · "
        f"Volume {layers.get('volume',0)}/20",
        f"  Pattern {layers.get('pattern',0)}/15 · Foreign {layers.get('foreign',0)}/15 · "
        f"Breadth {layers.get('breadth',0)}/5",
        "",
    ]

    if levels:
        lines += [
            "<b>Trade levels:</b>",
            f"  Entry  : <code>{levels.get('entry'):,.0f}</code>",
            f"  Target : <code>{levels.get('target'):,.0f}</code> (+{levels.get('target_pct')}%)",
            f"  Inv    : <code>{levels.get('stop_loss'):,.0f}</code> (-{levels.get('sl_pct')}%)  R:R {levels.get('rr')}",
            "",
        ]

    if warns:
        lines.append(f"⚠️ <i>{' | '.join(warns)}</i>")

    # Top reasons per layer
    for layer, layer_reasons in reasons.items():
        if layer_reasons:
            lines.append(f"• <i>{layer_reasons[0]}</i>")

    return "\n".join(l for l in lines if l is not None)


# ── Scheduled job callbacks ───────────────────────────────────────────────────

async def job_eod_scan(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Nightly D-1 scan at 16:30 WIB."""
    logger.info("EOD scan job triggered")
    loop = asyncio.get_running_loop()
    try:
        scan_data     = await loop.run_in_executor(_executor, _run_full_scan)
        briefing_text = await loop.run_in_executor(_executor, _build_briefing, scan_data)
        logger.info("EOD scan complete — briefing saved")
    except Exception:
        logger.exception("EOD scan job failed")


async def job_midday_briefing(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Pre-Sesi 2 briefing at 13:15 WIB."""
    logger.info("Midday briefing job triggered")
    loop = asyncio.get_running_loop()
    try:
        text = await loop.run_in_executor(_executor, _build_midday_briefing)
        if text:
            await broadcast(context.bot, text)
        else:
            logger.warning("Midday briefing: no text generated")
    except Exception:
        logger.exception("Midday briefing job failed")
        await broadcast(context.bot, "⚠️ Midday briefing gagal dibuat. Cek log.")


async def job_morning_briefing(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Morning briefing delivery at 08:30 WIB.
    If a fresh scan is triggered, _build_briefing() runs the 3-agent pipeline
    (agents.run_briefing_pipeline) for each candidate before formatting.
    """
    logger.info("Morning briefing job triggered")
    from firestore_client import load_latest_briefing
    text = load_latest_briefing()
    if text:
        await broadcast(context.bot, text)
    else:
        logger.warning("No saved briefing found — triggering fresh scan")
        loop = asyncio.get_running_loop()
        try:
            scan_data     = await loop.run_in_executor(_executor, _run_full_scan)
            briefing_text = await loop.run_in_executor(_executor, _build_briefing, scan_data)
            await broadcast(context.bot, briefing_text)
        except Exception:
            logger.exception("Morning briefing fallback scan failed")
            await broadcast(context.bot, "⚠️ Morning briefing gagal dibuat. Cek log.")


# ── Global error handler ─────────────────────────────────────────────────────

async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception (update=%s): %s", update, context.error, exc_info=context.error)


# ── Startup notification ──────────────────────────────────────────────────────

async def _on_startup(app: Application) -> None:
    now = datetime.now(_WIB).strftime("%d %b %Y %H:%M WIB")
    text = (
        f"🟢 <b>Claustock IDX v2 online</b>\n"
        f"<i>{now}</i>\n\n"
        f"Universe: {config.UNIVERSE} | Min score: {config.MIN_SCORE}\n"
        f"EOD scan: {config.EOD_SCAN_TIME[0]:02d}:{config.EOD_SCAN_TIME[1]:02d} WIB | "
        f"Briefing: {config.BRIEFING_TIME[0]:02d}:{config.BRIEFING_TIME[1]:02d} WIB"
    )
    await broadcast(app.bot, text)


# ── App builder ───────────────────────────────────────────────────────────────

def build_app() -> Application:
    app = (
        Application.builder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .post_init(_on_startup)
        .build()
    )

    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("help",     cmd_help))
    app.add_handler(CommandHandler("status",   cmd_status))
    app.add_handler(CommandHandler("briefing", cmd_briefing))
    app.add_handler(CommandHandler("midday",   cmd_midday))
    app.add_handler(CommandHandler("scan",     cmd_scan))
    app.add_handler(CommandHandler("pick",     cmd_pick))
    app.add_error_handler(_error_handler)

    return app
