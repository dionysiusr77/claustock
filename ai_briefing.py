"""
Claude API briefing synthesiser.
Takes the structured scan output and produces an Indonesian-language morning briefing.

Design:
  - System prompt is cached (static every call) — saves ~70% token cost on repeat calls
  - User prompt is a compact JSON payload (market context + top candidates)
  - Claude returns structured JSON; we format it into a Telegram-ready string
  - Falls back to a rule-based briefing if the API call fails
"""

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

def _build_user_prompt(scan_data: dict, breadth_summary: dict) -> str:
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
                "direction":         r.get("snapshot", {}).get("_ff_direction"),
                "consecutive_buy":   r.get("snapshot", {}).get("_ff_consec_buy", 0),
            },
        })

    payload = {
        "date":    datetime.now(_WIB).strftime("%A, %d %b %Y"),
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


# ── API call ──────────────────────────────────────────────────────────────────

def generate_briefing(scan_data: dict, breadth_summary: dict) -> dict | None:
    """
    Call Claude and return parsed briefing JSON.
    Returns None on failure (caller falls back to rule-based format).
    """
    user_prompt = _build_user_prompt(scan_data, breadth_summary)

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
        raw = response.content[0].text.strip()

        # Strip markdown code fences if Claude wraps with ```json
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        briefing = json.loads(raw)
        logger.info(
            "Briefing generated: %d picks, %d watchlist, cache_tokens=%s",
            len(briefing.get("picks", [])),
            len(briefing.get("watchlist", [])),
            getattr(response.usage, "cache_read_input_tokens", "n/a"),
        )
        return briefing

    except json.JSONDecodeError as e:
        logger.error("Claude returned invalid JSON: %s", e)
        return None
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
    briefing:       dict,
    scan_data:      dict,
    breadth_summary: dict,
) -> str:
    """
    Format briefing JSON into a Telegram-ready HTML string.
    Telegram HTML supports: <b>, <i>, <code>, <pre>
    """
    now       = datetime.now(_WIB).strftime("%A, %d %b %Y")
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
        f"IHSG: {breadth_summary.get('ihsg_close', '—')} "
        f"({ihsg_dir}{abs(ihsg_chg):.2f}%) | "
        f"Asing: {fm_dir} {abs(fm_net):.1f}B IDR",
        f"A/D: {breadth_summary.get('advance','—')}/{breadth_summary.get('decline','—')} | "
        f"Sentimen: {fg_emoji} {sentiment}",
        "",
        briefing.get("market_header", ""),
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

    # Merge Claude narratives with computed trade levels
    candidates_map = {r["symbol"]: r for r in scan_data.get("candidates", [])}

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
            f"   {pick.get('narrative', '')}",
            f"   <code>Entry  : {entry:,.0f}</code>" if isinstance(entry, (int, float)) else f"   Entry  : {entry}",
            f"   <code>Target : {target:,.0f} (+{t_pct}%)</code>" if isinstance(target, (int, float)) else f"   Target : {target}",
            f"   <code>Inv    : {sl:,.0f} (-{sl_pct}%)  R:R {rr}</code>" if isinstance(sl, (int, float)) else f"   Inv    : {sl}",
            f"   Hold: <i>{pick.get('hold_duration', '—')}</i>"
            + (f" | RSI {rsi:.1f}" if rsi else ""),
            f"   ⚠️ <i>{pick.get('key_risk', '')}</i>",
            "",
        ]

    # ── Watchlist ─────────────────────────────────────────────────────────────
    watchlist = briefing.get("watchlist", [])
    if watchlist:
        lines.append("<b>━━━ WATCHLIST ━━━</b>")
        for w in watchlist:
            lines.append(f"• <b>{w['symbol']}</b> — {w['note']}")
        lines.append("")

    # ── Risks ─────────────────────────────────────────────────────────────────
    risks = briefing.get("market_risks", [])
    if risks:
        lines.append("<b>⚠️ RISIKO HARI INI</b>")
        for r in risks:
            lines.append(f"• {r}")
        lines.append("")

    # ── Footer ────────────────────────────────────────────────────────────────
    lines.append(
        f"<i>Scan: {stats.get('liquid_size','—')} saham liquid dari "
        f"{stats.get('universe_size','—')} universe → "
        f"{stats.get('candidate_size','—')} kandidat</i>"
    )

    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────────────

def build_briefing(scan_data: dict, breadth_summary: dict) -> str:
    """
    Full pipeline: generate → format → return Telegram string.
    Falls back to rule-based if Claude API fails.
    """
    briefing = generate_briefing(scan_data, breadth_summary)
    if briefing is None:
        logger.warning("Falling back to rule-based briefing")
        briefing = _rule_based_briefing(scan_data, breadth_summary)

    return format_telegram(briefing, scan_data, breadth_summary)
