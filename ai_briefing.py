"""
Claude API briefing synthesiser.
Takes the structured scan output and produces an Indonesian-language morning briefing.

Design:
  - System prompt is cached (static every call) — saves ~70% token cost on repeat calls
  - User prompt is a compact JSON payload (market context + top candidates)
  - Claude returns structured JSON; we format it into a Telegram-ready string
  - Falls back to a rule-based briefing if the API call fails
"""

import html
import json
import logging
from datetime import datetime

import anthropic
import pytz

import config

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
_WIB    = pytz.timezone(config.MARKET_TZ)

# ── System prompt (cached) ────────────────────────────────────────────────────

_SYSTEM_PROMPT = """Kamu adalah analis pasar IDX (Bursa Efek Indonesia) profesional yang menulis morning briefing harian untuk trader retail Indonesia.

KONTEKS TRADING IDX:
- Biaya beli 0,15% + jual 0,25% = 0,40% round-trip
- Minimum target viable: +1,5% setelah biaya
- Lot minimum: 1 lot = 100 lembar saham
- Settlement T+2: modal tertahan 2 hari setelah jual
- Jam market: Sesi 1 (09:00–12:00 WIB), Sesi 2 (13:30–15:49 WIB)

SETUP YANG DIKENAL:
- BREAKOUT: harga close di atas high N-hari dengan volume tinggi
- PULLBACK: harga mundur ke EMA20/50, RSI 45–55, bouncing
- OVERSOLD_BOUNCE: RSI < 30 dengan candle reversal (hammer/engulfing)
- MOMENTUM_CONTINUATION: RSI sweet spot 45–65, EMA stack bullish
- FOREIGN_ACCUMULATION: asing net buy berturut-turut 3+ hari

GAYA PENULISAN:
- Bahasa Indonesia profesional, singkat, actionable
- Fokus pada WHY (kenapa ini menarik sekarang) bukan WHAT (deskripsi teknikal kering)
- Jangan sebut "stop loss" — sebut "level invalidasi"
- Setiap pick harus punya satu alasan katalis yang konkret

OUTPUT FORMAT (kembalikan JSON valid, tidak ada teks lain):
{
  "market_header": "string — 1-2 kalimat ringkasan kondisi pasar kemarin",
  "sentiment": "FEAR|CAUTIOUS|NEUTRAL|OPTIMISTIC|GREED",
  "picks": [
    {
      "symbol": "BBRI",
      "setup": "BREAKOUT",
      "narrative": "string — 2-3 kalimat: kenapa sekarang, apa katalisnya, apa yang dikonfirmasi",
      "hold_duration": "intraday|1-2 hari|3-5 hari",
      "key_risk": "string — 1 kalimat: kondisi yang membatalkan setup ini"
    }
  ],
  "watchlist": [
    {"symbol": "TLKM", "note": "string — 1 kalimat kenapa belum masuk"}
  ],
  "market_risks": ["string", "string"]
}"""


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_user_prompt(
    scan_data:       dict,
    breadth_summary: dict,
    for_date:        datetime | None = None,
) -> str:
    """Compact JSON payload — only the fields Claude needs."""

    candidates = scan_data.get("candidates", [])
    picks_payload = []

    for r in candidates[:config.TOP_N_AI]:
        snap   = r.get("snapshot", {})
        levels = r.get("trade_levels") or {}
        picks_payload.append({
            "symbol":       r["symbol"],
            "score":        r["total_score"],
            "verdict":      r["verdict"],
            "setup":        r["setup"],
            "divergence":   r.get("divergence", "NONE"),
            "rsi":          snap.get("rsi"),
            "vol_ratio":    snap.get("vol_ratio"),
            "close":        snap.get("close"),
            "ema20":        snap.get("ema20"),
            "ema50":        snap.get("ema50"),
            "atr_pct":      snap.get("atr_pct"),
            "pct_from_52w_high": snap.get("pct_from_52w_high"),
            "entry":        levels.get("entry"),
            "target":       levels.get("target"),
            "stop_loss":    levels.get("stop_loss"),
            "target_pct":   levels.get("target_pct"),
            "sl_pct":       levels.get("sl_pct"),
            "rr":           levels.get("rr"),
            "layer_scores": r.get("layer_scores", {}),
            "top_reasons":  _top_reasons(r),
            "bearish_warnings": r.get("bearish_warnings", []),
            "foreign": {
                "direction":       (r.get("foreign") or {}).get("direction", "NEUTRAL"),
                "consecutive_buy": (r.get("foreign") or {}).get("consecutive_buy_days", 0),
            },
        })

    target_dt = for_date or datetime.now(_WIB)
    payload = {
        "date":    target_dt.strftime("%A, %d %b %Y"),
        "market":  breadth_summary,
        "candidates": picks_payload,
    }

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _top_reasons(result: dict) -> list[str]:
    """Flatten the top 2 reasons from each layer into a single list."""
    out = []
    for layer, reasons in result.get("reasons", {}).items():
        if reasons:
            out.append(reasons[0])
    return out[:6]


# ── JSON extraction helper ───────────────────────────────────────────────────

def _parse_claude_json(raw: str, label: str = "") -> dict | None:
    """
    Robustly extract and parse a JSON object from Claude's raw response.
    Handles: markdown fences, leading/trailing prose, trailing commas.
    """
    text = raw.strip()

    # Strip ```json ... ``` fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    # Find outermost { ... } block (handles prose before/after JSON)
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        logger.error("%s no JSON object found in response: %s", label, text[:200])
        return None
    text = text[start:end + 1]

    import re as _re

    # Fix 1: trailing commas before } or ] (Claude emits JS-style JSON)
    text = _re.sub(r",\s*([}\]])", r"\1", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix 2: literal newlines/tabs inside string values (must be escaped in JSON)
    # Replace any bare \n or \t that sits inside a quoted string with a space.
    # Safe to apply globally — whitespace between tokens is irrelevant to json.loads.
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    # Re-apply trailing comma fix after whitespace collapse
    text = _re.sub(r",\s*([}\]])", r"\1", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        pos  = e.pos
        snip = text[max(0, pos - 60):pos + 60]
        logger.error("%s invalid JSON at char %d — …%s… | %s", label, pos, snip, e)
        return None


# ── API call ──────────────────────────────────────────────────────────────────

def generate_briefing(
    scan_data:       dict,
    breadth_summary: dict,
    for_date:        datetime | None = None,
) -> dict | None:
    """
    Call Claude and return parsed briefing JSON.
    Returns None on failure (caller falls back to rule-based format).
    """
    user_prompt = _build_user_prompt(scan_data, breadth_summary, for_date)

    try:
        response = _client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw      = response.content[0].text
        briefing = _parse_claude_json(raw, label="Morning briefing:")
        if briefing:
            logger.info(
                "Briefing generated: %d picks, %d watchlist, cache_tokens=%s",
                len(briefing.get("picks", [])),
                len(briefing.get("watchlist", [])),
                getattr(response.usage, "cache_read_input_tokens", "n/a"),
            )
        return briefing

    except Exception as e:
        logger.error("Claude API call failed: %s", e)
        return None


# ── Fallback: rule-based briefing ─────────────────────────────────────────────

def _rule_based_briefing(scan_data: dict, breadth_summary: dict) -> dict:
    """Used when Claude API is unavailable. Produces a mechanical briefing."""
    candidates = scan_data.get("candidates", [])

    picks = []
    for r in candidates[:5]:
        levels = r.get("trade_levels") or {}
        picks.append({
            "symbol":        r["symbol"],
            "setup":         r["setup"],
            "narrative":     (
                                 f"Skor {r['total_score']}/100. Setup: {r['setup']}. "
                                 + (f"RSI {r['snapshot']['rsi']:.1f}. " if r['snapshot'].get('rsi') is not None else "")
                                 + f"Volume ratio {r['snapshot'].get('vol_ratio') or 0:.1f}×."
                             ),
            "hold_duration": "1-2 hari",
            "key_risk":      "Level invalidasi di bawah stop loss.",
        })

    ihsg_chg = breadth_summary.get("ihsg_change_pct", 0) or 0
    direction = "naik" if ihsg_chg > 0 else "turun"
    return {
        "market_header": f"IHSG kemarin {direction} {abs(ihsg_chg):.2f}%. "
                         f"Asing {breadth_summary.get('foreign_market_direction', 'NEUTRAL')}.",
        "sentiment":     breadth_summary.get("fear_greed_label", "NEUTRAL"),
        "picks":         picks,
        "watchlist":     [],
        "market_risks":  ["(briefing otomatis — Claude API tidak tersedia)"],
    }


# ── Telegram formatter ────────────────────────────────────────────────────────

_SETUP_EMOJI = {
    "BREAKOUT":              "🚀",
    "OVERSOLD_BOUNCE":       "🔄",
    "MOMENTUM_CONTINUATION": "📈",
    "PULLBACK":              "🎯",
    "FOREIGN_ACCUMULATION":  "🏦",
    "WATCH":                 "👀",
}

_SENTIMENT_EMOJI = {
    "GREED":      "🤑",
    "OPTIMISTIC": "😊",
    "NEUTRAL":    "😐",
    "CAUTIOUS":   "😟",
    "FEAR":       "😱",
}


def format_telegram(
    briefing:        dict,
    scan_data:       dict,
    breadth_summary: dict,
    pipeline_map:    dict | None = None,
    for_date:        datetime | None = None,
) -> str:
    """
    Format briefing JSON into a Telegram-ready HTML string.
    Telegram HTML supports: <b>, <i>, <code>, <pre>
    """
    target_dt = for_date or datetime.now(_WIB)
    now       = target_dt.strftime("%A, %d %b %Y")
    sentiment = briefing.get("sentiment", "NEUTRAL")
    fg_emoji  = _SENTIMENT_EMOJI.get(sentiment, "😐")
    stats     = scan_data.get("stats", {})

    ihsg_chg  = breadth_summary.get("ihsg_change_pct", 0) or 0
    ihsg_dir  = "▲" if ihsg_chg > 0 else ("▼" if ihsg_chg < 0 else "—")
    fm_net    = breadth_summary.get("foreign_market_net_idr", 0) or 0
    fm_dir    = "Net Buy" if fm_net > 0 else ("Net Sell" if fm_net < 0 else "Netral")

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        f"<b>📊 MORNING BRIEFING IDX</b>",
        f"<i>{now}</i>",
        "",
        f"IHSG: {breadth_summary.get('ihsg_close') or '—'} "
        f"({ihsg_dir}{abs(ihsg_chg):.2f}%) | "
        f"Asing: {fm_dir} {abs(fm_net):.1f}B IDR",
        f"A/D: {breadth_summary.get('advance','—')}/{breadth_summary.get('decline','—')} | "
        f"Sentimen: {fg_emoji} {sentiment}",
        "",
        html.escape(briefing.get("market_header", "")),
        "",
    ]

    # ── Sector heat ───────────────────────────────────────────────────────────
    hot  = breadth_summary.get("sectors_hot", [])
    cold = breadth_summary.get("sectors_cold", [])
    if hot or cold:
        if hot:
            lines.append(f"🟢 <b>Sektor kuat:</b> {', '.join(hot[:3])}")
        if cold:
            lines.append(f"🔴 <b>Sektor lemah:</b> {', '.join(cold[:3])}")
        lines.append("")

    # ── Picks ─────────────────────────────────────────────────────────────────
    picks = briefing.get("picks", [])
    if picks:
        lines.append(f"<b>━━━ TOP PICKS ({len(picks)}) ━━━</b>")
        lines.append("")

    # Merge Claude narratives with computed trade levels.
    # Claude returns symbols without .JK; candidates use XXXX.JK — index both.
    candidates_map = {}
    for r in scan_data.get("candidates", []):
        candidates_map[r["symbol"]] = r
        candidates_map[r["symbol"].replace(".JK", "")] = r

    for i, pick in enumerate(picks, 1):
        sym    = pick["symbol"]
        setup  = pick.get("setup", "WATCH")
        emoji  = _SETUP_EMOJI.get(setup, "📌")
        cand   = candidates_map.get(sym, {})
        levels = cand.get("trade_levels") or {}
        score  = cand.get("total_score", "—")
        rsi    = cand.get("snapshot", {}).get("rsi")

        entry  = levels.get("entry", "—")
        target = levels.get("target", "—")
        sl     = levels.get("stop_loss", "—")
        t_pct  = levels.get("target_pct", "—")
        sl_pct = levels.get("sl_pct", "—")
        rr     = levels.get("rr", "—")

        lines += [
            f"<b>{i}. {sym}</b> — {emoji} {setup}  <i>(skor {score})</i>",
            f"   {html.escape(pick.get('narrative', ''))}",
            f"   <code>Entry  : {entry:,.0f}</code>" if isinstance(entry, (int, float)) else f"   Entry  : {entry}",
            f"   <code>Target : {target:,.0f} (+{t_pct}%)</code>" if isinstance(target, (int, float)) else f"   Target : {target}",
            f"   <code>Inv    : {sl:,.0f} (-{sl_pct}%)  R:R {rr}</code>" if isinstance(sl, (int, float)) else f"   Inv    : {sl}",
            f"   Hold: <i>{html.escape(str(pick.get('hold_duration', '—')))}</i>"
            + (f" | RSI {rsi:.1f}" if rsi else ""),
            f"   ⚠️ <i>{html.escape(pick.get('key_risk', ''))}</i>",
        ]

        # Inject pipeline block if available for this symbol
        if pipeline_map:
            from agents import format_pipeline_block
            pr = pipeline_map.get(sym) or pipeline_map.get(sym.replace(".JK", ""))
            if pr:
                block = format_pipeline_block(pr)
                if block:
                    lines.append(block)

        lines.append("")

    # ── Watchlist ─────────────────────────────────────────────────────────────
    watchlist = briefing.get("watchlist", [])
    if watchlist:
        lines.append("<b>━━━ WATCHLIST ━━━</b>")
        for w in watchlist:
            lines.append(f"• <b>{html.escape(w['symbol'])}</b> — {html.escape(w.get('note', ''))}")
        lines.append("")

    # ── Risks ─────────────────────────────────────────────────────────────────
    risks = briefing.get("market_risks", [])
    if risks:
        lines.append("<b>⚠️ RISIKO HARI INI</b>")
        for r in risks:
            lines.append(f"• {html.escape(r)}")
        lines.append("")

    # ── Footer ────────────────────────────────────────────────────────────────
    lines.append(
        f"<i>Scan: {stats.get('liquid_size','—')} saham liquid dari "
        f"{stats.get('universe_size','—')} universe → "
        f"{stats.get('candidate_size','—')} kandidat</i>"
    )

    return "\n".join(lines)


# ── Midday / pre-Sesi 2 briefing ─────────────────────────────────────────────

_MIDDAY_SYSTEM_PROMPT = """Kamu adalah trader IDX profesional yang menulis update singkat sebelum Sesi 2 (13:30 WIB).

Kamu sudah punya morning picks dari D-1 scan. Sekarang Sesi 1 (09:00–12:00) sudah selesai dan kamu punya data intraday-nya.

TUGASMU:
Evaluasi setiap morning pick berdasarkan aksi harga Sesi 1, lalu berikan rekomendasi untuk Sesi 2.

STATUS YANG BISA DIBERIKAN:
- MASIH_VALID : Setup D-1 intact, harga di/dekat zona entry, volume Sesi 1 konfirmasi
- TUNGGU      : Setup intact tapi harga belum di zona entry — tunggu pullback di Sesi 2
- SUDAH_JALAN : Harga sudah naik melewati entry/target — terlambat masuk, skip
- BATAL       : Support pecah atau setup rusak di Sesi 1 — jangan masuk

FOKUS ANALISA:
- Apakah harga Sesi 1 sudah menyentuh zona entry D-1?
- Volume Sesi 1 tinggi (konfirmasi) atau rendah (waspadai)?
- IHSG Sesi 1 naik atau turun? Apakah mempengaruhi setup?
- Jika ada setup baru yang muncul dari aksi Sesi 1, sebutkan sebagai BARU

GAYA PENULISAN:
- Bahasa Indonesia, singkat, langsung ke angka dan aksi
- Sertakan zona entry baru jika berubah dari morning pick
- Maksimal 2 kalimat per pick

OUTPUT FORMAT (JSON valid, tidak ada teks lain):
{
  "sesi1_summary": "string — 1 kalimat ringkasan IHSG dan market Sesi 1",
  "ihsg_sesi1_pct": number,
  "picks": [
    {
      "symbol": "BBRI",
      "status": "MASIH_VALID|TUNGGU|SUDAH_JALAN|BATAL",
      "entry_zone": "string — zona entry untuk Sesi 2 (e.g. '4250–4300')",
      "note": "string — 1-2 kalimat analisa Sesi 1 dan rekomendasi Sesi 2",
      "hold_duration": "intraday|overnight"
    }
  ],
  "new_picks": [
    {
      "symbol": "TLKM",
      "setup": "string",
      "note": "string — kenapa ini menarik setelah Sesi 1"
    }
  ],
  "risks": ["string"]
}"""


_MIDDAY_MAX_PICKS = 8   # cap to avoid truncating the JSON at max_tokens


def _build_midday_prompt(
    morning_candidates: list[dict],
    sesi1_map:          dict[str, dict | None],
    ihsg_sesi1:         dict | None,
) -> str:
    """Build compact JSON payload for the midday Claude call."""
    picks_payload = []
    for r in morning_candidates[:_MIDDAY_MAX_PICKS]:
        sym    = r["symbol"]
        snap   = r.get("snapshot", {})
        levels = r.get("trade_levels") or {}
        s1     = sesi1_map.get(sym) or sesi1_map.get(sym.replace(".JK", ""))

        picks_payload.append({
            "symbol":       sym.replace(".JK", ""),
            "d1_score":     r.get("total_score"),
            "d1_setup":     r.get("setup"),
            "d1_entry":     levels.get("entry"),
            "d1_target":    levels.get("target"),
            "d1_stop":      levels.get("stop_loss"),
            "d1_rsi":       snap.get("rsi"),
            "d1_close":     snap.get("close"),
            "sesi1": {
                "open":       s1.get("open")       if s1 else None,
                "high":       s1.get("high")       if s1 else None,
                "low":        s1.get("low")        if s1 else None,
                "close":      s1.get("close")      if s1 else None,
                "pct_change": s1.get("pct_change") if s1 else None,
                "volume":     s1.get("volume")     if s1 else None,
                "candles":    s1.get("candles")    if s1 else None,
                "data":       s1 is not None,
            },
        })

    payload = {
        "time":  datetime.now(_WIB).strftime("%H:%M WIB"),
        "ihsg_sesi1": ihsg_sesi1,
        "morning_picks": picks_payload,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def generate_midday_briefing(
    morning_candidates: list[dict],
    sesi1_map:          dict[str, dict | None],
    ihsg_sesi1:         dict | None,
) -> dict | None:
    """Call Claude for the pre-Sesi 2 briefing. Returns parsed JSON or None."""
    user_prompt = _build_midday_prompt(morning_candidates, sesi1_map, ihsg_sesi1)
    try:
        response = _client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=2500,
            system=[
                {
                    "type": "text",
                    "text": _MIDDAY_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text
        return _parse_claude_json(raw, label="Midday briefing:")
    except Exception as e:
        logger.error("Midday briefing: Claude API call failed: %s", e)
        return None


_STATUS_EMOJI = {
    "MASIH_VALID":  "✅",
    "TUNGGU":       "⏳",
    "SUDAH_JALAN":  "🚀",
    "BATAL":        "❌",
}


def format_midday_telegram(
    briefing:           dict,
    morning_candidates: list[dict],
    sesi1_map:          dict[str, dict | None],
) -> str:
    """Format midday briefing JSON into Telegram HTML."""
    now  = datetime.now(_WIB).strftime("%H:%M WIB")
    cmap = {}
    for r in morning_candidates:
        cmap[r["symbol"]] = r
        cmap[r["symbol"].replace(".JK", "")] = r

    lines = [
        "<b>📊 PRE-SESI 2 UPDATE</b>",
        f"<i>{now} — evaluasi morning picks setelah Sesi 1</i>",
        "",
        html.escape(briefing.get("sesi1_summary", "")),
        "",
    ]

    picks = briefing.get("picks", [])
    if picks:
        lines.append("<b>━━━ STATUS MORNING PICKS ━━━</b>")
        lines.append("")

    for pick in picks:
        sym    = pick["symbol"]
        status = pick.get("status", "TUNGGU")
        emoji  = _STATUS_EMOJI.get(status, "⏳")
        cand   = cmap.get(sym, cmap.get(f"{sym}.JK", {}))
        levels = cand.get("trade_levels") or {}
        s1     = sesi1_map.get(f"{sym}.JK") or sesi1_map.get(sym)

        s1_line = ""
        if s1:
            pct = s1.get("pct_change")
            pct_str = f"{pct:+.2f}%" if pct is not None else "—"
            s1_line = (
                f"   Sesi 1: H <code>{s1['high']:,.0f}</code> "
                f"L <code>{s1['low']:,.0f}</code> "
                f"tutup <code>{s1['close']:,.0f}</code> ({pct_str})"
            )

        target  = levels.get("target", "—")
        sl      = levels.get("stop_loss", "—")

        lines += [
            f"<b>{sym}</b> {emoji} <b>{status}</b>",
            f"   {html.escape(pick.get('note', ''))}",
            s1_line,
            f"   Entry Sesi 2: <code>{html.escape(pick.get('entry_zone', '—'))}</code>"
            f"  |  Target: <code>{target:,.0f}</code>"
            f"  Inv: <code>{sl:,.0f}</code>"
            if isinstance(target, (int, float)) else
            f"   Entry Sesi 2: <code>{html.escape(pick.get('entry_zone', '—'))}</code>",
            "",
        ]

    new_picks = briefing.get("new_picks", [])
    if new_picks:
        lines.append("<b>━━━ SETUP BARU SESI 2 ━━━</b>")
        for p in new_picks:
            lines.append(
                f"• <b>{html.escape(p['symbol'])}</b> — {html.escape(p.get('note', ''))}"
            )
        lines.append("")

    risks = briefing.get("risks", [])
    if risks:
        lines.append("<b>⚠️ RISIKO</b>")
        for r in risks:
            lines.append(f"• {html.escape(r)}")

    return "\n".join(l for l in lines if l is not None)


def build_midday_briefing(
    morning_candidates: list[dict],
    sesi1_map:          dict[str, dict | None],
    ihsg_sesi1:         dict | None,
) -> str:
    """Full midday pipeline: generate → format → return Telegram string."""
    now = datetime.now(_WIB).strftime("%H:%M WIB")

    # Guard: if no Sesi 1 data at all, return a clear message instead of empty output
    has_sesi1 = any(v is not None for v in sesi1_map.values())
    if not has_sesi1:
        logger.warning("build_midday_briefing: no Sesi 1 intraday data available")
        return (
            "<b>📊 PRE-SESI 2 UPDATE</b>\n"
            f"<i>{now}</i>\n\n"
            "⚠️ Data intraday Sesi 1 belum tersedia.\n"
            "Pastikan dipanggil setelah Sesi 1 selesai (≥12:00 WIB)."
        )

    briefing = generate_midday_briefing(morning_candidates, sesi1_map, ihsg_sesi1)
    if briefing is None:
        # Rule-based fallback — Claude API unavailable
        logger.warning("Midday briefing: falling back to rule-based output")
        lines = [
            "<b>📊 PRE-SESI 2 UPDATE</b>",
            f"<i>{now} — evaluasi morning picks setelah Sesi 1</i>",
            "<i>(Claude API tidak tersedia — data mentah)</i>",
            "",
        ]
        for r in morning_candidates[:8]:
            sym = r["symbol"].replace(".JK", "")
            s1  = sesi1_map.get(r["symbol"]) or sesi1_map.get(sym)
            pct = s1.get("pct_change") if s1 else None
            close = s1.get("close") if s1 else None
            pct_str   = f"{pct:+.2f}%" if pct is not None else "—"
            close_str = f"<code>{close:,.0f}</code>" if close is not None else "no data"
            lines.append(f"• <b>{sym}</b> — Sesi 1 tutup: {close_str} ({pct_str})")
        return "\n".join(lines)

    return format_midday_telegram(briefing, morning_candidates, sesi1_map)


# ── Main entry point ──────────────────────────────────────────────────────────

def build_briefing(
    scan_data:       dict,
    breadth_summary: dict,
    pipeline_map:    dict | None = None,
    for_date:        datetime | None = None,
) -> str:
    """
    Full pipeline: generate → format → return Telegram string.
    Falls back to rule-based if Claude API fails.
    Pass for_date to stamp the briefing with a specific date (e.g. next trading day
    when building from the EOD scan at 16:30).
    """
    briefing = generate_briefing(scan_data, breadth_summary, for_date)
    if briefing is None:
        logger.warning("Falling back to rule-based briefing")
        briefing = _rule_based_briefing(scan_data, breadth_summary)

    return format_telegram(briefing, scan_data, breadth_summary, pipeline_map, for_date)
