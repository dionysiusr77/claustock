"""
3-Agent briefing pipeline for IDX pre-session analysis.

Pipeline:
  Agent 1 (claude-sonnet-4-20250514) — Deep technical + fundamental analysis
  Agent 2 (claude-haiku-4-5-20251001) — Market participant simulation (5 personas)
  Agent 3 (claude-haiku-4-5-20251001) — Final recommendation synthesis
"""

import html
import json
import logging
import re
import time
from datetime import datetime

import anthropic
import pytz

import config

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

_ANALYSIS_MODEL   = "claude-sonnet-4-20250514"
_SIMULATION_MODEL = "claude-haiku-4-5-20251001"
_CONCLUSION_MODEL = "claude-haiku-4-5-20251001"


# ── JSON extraction helper ────────────────────────────────────────────────────

def _parse_json(raw: str, label: str = "") -> dict | None:
    """Extract and parse JSON from Claude response. Handles fences and trailing commas."""
    text = raw.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        logger.error("%s no JSON found in response: %s", label, text[:200])
        return None
    text = text[start:end + 1]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    text = (text.replace("\r\n", " ").replace("\r", " ")
                .replace("\n", " ").replace("\t", " "))
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        pos  = e.pos
        snip = text[max(0, pos - 60):pos + 60]
        logger.error("%s invalid JSON at char %d — …%s… | %s", label, pos, snip, e)
        return None


# ── Technical signal helpers ──────────────────────────────────────────────────

def _macd_status(snap: dict) -> str:
    macd = snap.get("macd")
    sig  = snap.get("macd_signal")
    hist = snap.get("macd_hist")
    if macd is None or sig is None:
        return "N/A"
    if macd > sig and hist is not None and hist > 0:
        return "BULLISH (above signal, histogram expanding)"
    if macd > sig:
        return "BULLISH (above signal)"
    return "BEARISH (below signal)"


def _bb_position(snap: dict) -> str:
    bp = snap.get("bb_pct")
    if bp is None:
        return "N/A"
    pct = bp * 100
    if pct >= 80:
        return f"Near upper band ({pct:.0f}%ile)"
    if pct <= 20:
        return f"Near lower band ({pct:.0f}%ile)"
    return f"Mid-band ({pct:.0f}%ile)"


def _ma_trend(snap: dict) -> str:
    price = snap.get("close")
    e20   = snap.get("ema20")
    e50   = snap.get("ema50")
    e200  = snap.get("ema200")
    if not all(v is not None for v in (price, e20, e50, e200)):
        return "N/A"
    met = sum([price > e20, e20 > e50, e50 > e200, price > e200])
    return {
        4: "Full bull stack (price>EMA20>EMA50>EMA200)",
        3: "Strong uptrend (3/4 conditions met)",
        2: "Mixed trend",
        1: "Weak/downtrend",
        0: "Full bear stack",
    }.get(met, "Mixed")


def _candle_pattern(score_data: dict) -> str:
    for r in score_data.get("reasons", {}).get("pattern", []):
        if r and "WARNING" not in r:
            return r
    return "none"


# ── Agent 1: Deep Analysis (Sonnet) ──────────────────────────────────────────

_ANALYSIS_SYSTEM = (
    "You are a senior IDX (Indonesia Stock Exchange) equity analyst. "
    "Analyze all provided data layers for an Indonesian stock and "
    "produce a structured deep analysis. Be specific, reference actual "
    "numbers. Respond only in valid JSON."
)


def run_analysis_agent(
    symbol:       str,
    technical:    dict,
    forecast:     dict,
    foreign_flow: dict,
    news:         dict,
    market_ctx:   dict,
) -> dict | None:
    price    = technical.get("price") or 0
    per_pos  = config.CAPITAL_IDR // max(getattr(config, "MAX_POSITIONS", 5), 1)
    max_lots = getattr(config, "IDX_MAX_LOTS", 5)

    ff_idr = abs(foreign_flow.get("net_buy_idr", 0)) / 1e9

    user_prompt = (
        f"Analyze this IDX stock for the pre-session briefing.\n\n"
        f"SYMBOL: {symbol}\n"
        f"CURRENT PRICE: Rp {price:,.0f}\n\n"
        f"TECHNICAL DATA (score: {technical.get('score', 0)}/80):\n"
        f"- RSI: {technical.get('rsi', 'N/A')}\n"
        f"- MACD: {technical.get('macd_status', 'N/A')}\n"
        f"- BB Position: {technical.get('bb_position', 'N/A')}\n"
        f"- MA Trend: {technical.get('ma_trend', 'N/A')}\n"
        f"- Volume Ratio: {technical.get('vol_ratio', 'N/A')}x\n"
        f"- Candle Pattern: {technical.get('candle_pattern', 'none')}\n"
        f"- Divergence: {technical.get('divergence', 'NONE')}\n\n"
        f"FORECAST (score: {forecast.get('score', 0)}/25):\n"
        f"- 5-day trend: {forecast.get('direction', 'NEUTRAL')} "
        f"{forecast.get('trend_pct', 0):.1f}%\n"
        f"- Confidence: {forecast.get('confidence', 50)}%\n\n"
        f"FOREIGN FLOW (score: {foreign_flow.get('score', 0)}/20):\n"
        f"- Net position: {foreign_flow.get('direction', 'NEUTRAL')} "
        f"Rp {ff_idr:.1f}B IDR\n"
        f"- Consecutive days: {foreign_flow.get('consecutive_days', 0)}\n\n"
        f"NEWS (score: {news.get('score', 0)}/20):\n"
        f"- Headline: {news.get('headline', 'Tidak ada berita terbaru')}\n"
        f"- Sentiment: {news.get('sentiment', 'NEUTRAL')}\n\n"
        f"TOTAL SCORE: {technical.get('total_score', 0)}/100 — "
        f"Verdict: {technical.get('verdict', 'WATCH')}\n\n"
        f"MARKET CONTEXT:\n"
        f"- IHSG: {market_ctx.get('ihsg_change', 0):+.2f}%\n"
        f"- Wall St (S&P 500): {market_ctx.get('wall_st_change', 0):+.2f}%\n"
        f"- USD/IDR: {market_ctx.get('usd_idr', '—')}\n"
        f"- Macro: {market_ctx.get('macro_note', 'none')}\n\n"
        f"CAPITAL CONTEXT: Rp {per_pos:,.0f} per position, "
        f"max {max_lots} lots (1 lot = 100 shares)\n\n"
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "technical_insight": "2 sentences on technical setup",\n'
        '  "fundamental_insight": "2 sentences on fundamental context",\n'
        '  "catalyst": "main price catalyst for today",\n'
        '  "risk": "main risk for today",\n'
        '  "confidence": <int 0-100>,\n'
        '  "price_thesis": "BULLISH|NEUTRAL|BEARISH",\n'
        '  "entry_zone": "price range or N/A",\n'
        '  "target_1": <float>,\n'
        '  "target_2": <float>,\n'
        '  "stop_loss": <float>,\n'
        '  "hold_duration": "intraday|1-2 days|3-5 days",\n'
        f'  "lot_suggestion": <int max {max_lots}>,\n'
        '  "analysis_score": <int: confidence>=90->20, >=75->15, >=60->10, else 0>\n'
        "}"
    )

    def _call():
        return _client.messages.create(
            model=_ANALYSIS_MODEL,
            max_tokens=800,
            system=_ANALYSIS_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )

    try:
        resp = _call()
        return _parse_json(resp.content[0].text, label=f"Agent1/{symbol}")
    except Exception as e:
        if any(k in str(e).lower() for k in ("timeout", "overloaded", "529")):
            logger.warning("Agent1/%s timeout — retrying once", symbol)
            try:
                time.sleep(3)
                resp = _call()
                return _parse_json(resp.content[0].text, label=f"Agent1/{symbol}")
            except Exception as e2:
                logger.error("Agent1/%s retry failed: %s", symbol, e2)
                return None
        logger.error("Agent1/%s failed: %s", symbol, e)
        return None


# ── Agent 2: Market Simulation (Haiku) ───────────────────────────────────────

_SIMULATION_SYSTEM = (
    "You are simulating how 5 types of Indonesian stock market "
    "participants react to information. You deeply understand: "
    "retail investors on Stockbit (momentum, emotional, FOMO-driven), "
    "foreign institutional funds (GIC, Vanguard, macro-sensitive), "
    "domestic fund managers (Schroder, Manulife, fundamental-focused), "
    "short-term scalpers (technical, RSI/volume, intraday), "
    "and long-term value investors (dividend-aware, patient). "
    "Respond only in valid JSON."
)


def run_simulation_agent(
    symbol:     str,
    analysis:   dict,
    news:       dict,
    technical:  dict,
    foreign:    dict,
    market_ctx: dict,
) -> dict | None:
    thesis     = analysis.get("price_thesis", "NEUTRAL") if analysis else "NEUTRAL"
    catalyst   = analysis.get("catalyst", "—")           if analysis else "—"
    risk_text  = analysis.get("risk", "—")               if analysis else "—"
    confidence = analysis.get("confidence", 50)           if analysis else 50

    news_items = news.get("items", []) if isinstance(news.get("items"), list) else []
    if news_items:
        headlines = [n.get("headline", "") for n in news_items[:3]]
    else:
        hl = news.get("headline", "Tidak ada berita terbaru")
        headlines = [hl] if hl else ["Tidak ada berita terbaru"]
    headlines_str = "\n".join(f"- {h}" for h in headlines)

    user_prompt = (
        f"Simulate how 5 Indonesian market participant personas react to this stock.\n\n"
        f"SYMBOL: {symbol}\n\n"
        f"DEEP ANALYSIS SUMMARY:\n"
        f"- Thesis: {thesis}\n"
        f"- Catalyst: {catalyst}\n"
        f"- Risk: {risk_text}\n"
        f"- Confidence: {confidence}%\n\n"
        f"NEWS HEADLINES (max 3):\n{headlines_str}\n\n"
        f"TECHNICAL SIGNALS:\n"
        f"- RSI: {technical.get('rsi', 'N/A')}\n"
        f"- MACD: {technical.get('macd_status', 'N/A')}\n"
        f"- Volume Ratio: {technical.get('vol_ratio', 'N/A')}x\n"
        f"- Foreign Flow: {foreign.get('direction', 'NEUTRAL')}\n\n"
        f"MARKET CONTEXT:\n"
        f"- IHSG: {market_ctx.get('ihsg_change', 0):+.2f}%\n"
        f"- Wall St: {market_ctx.get('wall_st_change', 0):+.2f}%\n"
        f"- USD/IDR: {market_ctx.get('usd_idr', '—')}\n\n"
        "Simulate exactly 5 persona reactions IN THIS ORDER:\n"
        "1. Retail investor (Stockbit user)\n"
        "2. Foreign institutional fund\n"
        "3. Domestic fund manager\n"
        "4. Short-term scalper\n"
        "5. Long-term value investor\n\n"
        "bullish_pct + neutral_pct + bearish_pct must equal exactly 100.\n\n"
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "personas": [\n'
        "    {\n"
        '      "persona": "<name>",\n'
        '      "action": "BUY|HOLD|SELL|WATCH",\n'
        '      "confidence": <int 0-100>,\n'
        '      "reasoning": "<max 1 sentence in Bahasa Indonesia>"\n'
        "    }\n"
        "  ],\n"
        '  "aggregate": "BULLISH|NEUTRAL|BEARISH",\n'
        '  "bullish_pct": <int>,\n'
        '  "neutral_pct": <int>,\n'
        '  "bearish_pct": <int>,\n'
        '  "simulated_open_move": <float % change>,\n'
        '  "key_driver": "<1 sentence>",\n'
        '  "risk_factor": "<1 sentence>",\n'
        '  "sim_score": <int: BULLISH>=70%->20, BULLISH<70%->15, BULLISH<60%->10, NEUTRAL->5, BEARISH->0>\n'
        "}"
    )

    try:
        resp = _client.messages.create(
            model=_SIMULATION_MODEL,
            max_tokens=600,
            system=_SIMULATION_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return _parse_json(resp.content[0].text, label=f"Agent2/{symbol}")
    except Exception as e:
        logger.error("Agent2/%s failed: %s", symbol, e)
        return None


# ── Agent 3: Conclusion (Haiku) ───────────────────────────────────────────────

_CONCLUSION_SYSTEM = (
    "You are a decisive IDX trading advisor giving the final "
    "recommendation for a pre-session briefing. You receive a deep "
    "analysis and a market simulation. Synthesize both into a clear, "
    "actionable recommendation. Be direct. Respond only in valid JSON."
)


def run_conclusion_agent(
    symbol:     str,
    score_data: dict,
    analysis:   dict,
    simulation: dict,
) -> dict | None:
    snap   = score_data.get("snapshot", {})
    price  = snap.get("close", 0) or 0
    layers = score_data.get("layer_scores", {})
    levels = score_data.get("trade_levels") or {}

    thesis    = analysis.get("price_thesis", "NEUTRAL")                           if analysis else "NEUTRAL"
    catalyst  = analysis.get("catalyst", "—")                                     if analysis else "—"
    risk_text = analysis.get("risk", "—")                                         if analysis else "—"
    entry_z   = analysis.get("entry_zone", levels.get("entry", "N/A"))            if analysis else levels.get("entry", "N/A")
    target_1  = analysis.get("target_1", levels.get("target", 0))                 if analysis else levels.get("target", 0)
    target_2  = analysis.get("target_2", 0)                                       if analysis else 0
    sl        = analysis.get("stop_loss", levels.get("stop_loss", 0))             if analysis else levels.get("stop_loss", 0)

    agg_sent   = simulation.get("aggregate", "NEUTRAL")   if simulation else "NEUTRAL"
    sim_move   = simulation.get("simulated_open_move", 0) if simulation else 0
    key_driver = simulation.get("key_driver", "—")        if simulation else "—"

    analysis_score = analysis.get("analysis_score", 0)  if analysis  else 0
    sim_score      = simulation.get("sim_score", 0)      if simulation else 0
    expected_final = score_data.get("total_score", 0) + analysis_score + sim_score

    t2_str = f"{target_2:,.0f}" if isinstance(target_2, (int, float)) and target_2 else "—"

    user_prompt = (
        f"Provide the final IDX trading recommendation for the pre-session briefing.\n\n"
        f"SYMBOL: {symbol}\n"
        f"PRICE: Rp {price:,.0f}\n\n"
        f"SCORING SUMMARY:\n"
        f"- Total score: {score_data.get('total_score', 0)}/100 — "
        f"Verdict: {score_data.get('verdict', 'WATCH')}\n"
        f"- Trend: {layers.get('trend', 0)}/25 | Momentum: {layers.get('momentum', 0)}/20 | "
        f"Volume: {layers.get('volume', 0)}/20\n"
        f"- Pattern: {layers.get('pattern', 0)}/15 | Foreign: {layers.get('foreign', 0)}/15 | "
        f"Breadth: {layers.get('breadth', 0)}/5\n\n"
        f"AGENT 1 — DEEP ANALYSIS:\n"
        f"- Thesis: {thesis}\n"
        f"- Catalyst: {catalyst}\n"
        f"- Risk: {risk_text}\n"
        f"- Entry zone: {entry_z}\n"
        f"- Target 1: {target_1:,.0f} | Target 2: {t2_str}\n"
        f"- Stop Loss: {sl:,.0f}\n\n"
        f"AGENT 2 — MARKET SIMULATION:\n"
        f"- Aggregate sentiment: {agg_sent}\n"
        f"- Estimated open move: {sim_move:+.1f}%\n"
        f"- Key driver: {key_driver}\n\n"
        "Conclude with a final recommendation. "
        "Use Bahasa Indonesia for action, conviction, alasan, peringatan.\n\n"
        f"Return JSON with this exact schema (final_score should be ~{expected_final}):\n"
        "{\n"
        '  "action": "BELI|PERHATIKAN|LEWATI",\n'
        '  "conviction": "TINGGI|SEDANG|RENDAH",\n'
        '  "entry": <float>,\n'
        '  "target_1": <float>,\n'
        '  "target_2": <float>,\n'
        '  "stop_loss": <float>,\n'
        '  "lots": <int>,\n'
        '  "hold": "intraday|1-2 hari|3-5 hari",\n'
        '  "alasan": "<2-3 sentences in Bahasa Indonesia>",\n'
        '  "peringatan": "<1 sentence main risk in Bahasa Indonesia>",\n'
        f'  "final_score": <int: {score_data.get("total_score", 0)} + {analysis_score} + {sim_score}>\n'
        "}"
    )

    try:
        resp = _client.messages.create(
            model=_CONCLUSION_MODEL,
            max_tokens=400,
            system=_CONCLUSION_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        result = _parse_json(resp.content[0].text, label=f"Agent3/{symbol}")
        if result and "final_score" not in result:
            result["final_score"] = expected_final
        return result
    except Exception as e:
        logger.error("Agent3/%s failed: %s", symbol, e)
        return None


# ── Pipeline Orchestrator ─────────────────────────────────────────────────────

def run_briefing_pipeline(
    symbol:     str,
    score_data: dict,
    news_raw:   list,
    market_ctx: dict,
) -> dict:
    """
    Runs all 3 agents sequentially.
    Returns combined result or partial result if an agent fails.
    Agents 2 and 3 degrade gracefully if Agent 1 fails.
    """
    snap   = score_data.get("snapshot", {})
    layers = score_data.get("layer_scores", {})
    ff_raw = score_data.get("foreign") or {}

    technical = {
        "price":          snap.get("close", 0),
        "rsi":            snap.get("rsi"),
        "macd_status":    _macd_status(snap),
        "bb_position":    _bb_position(snap),
        "ma_trend":       _ma_trend(snap),
        "vol_ratio":      snap.get("vol_ratio"),
        "candle_pattern": _candle_pattern(score_data),
        "divergence":     snap.get("divergence", "NONE"),
        "score":          sum(layers.get(k, 0) for k in ("trend", "momentum", "volume", "pattern")),
        "total_score":    score_data.get("total_score", 0),
        "verdict":        score_data.get("verdict", "WATCH"),
    }

    forecast = {
        "direction":  "BULLISH" if score_data.get("total_score", 0) >= 65 else "NEUTRAL",
        "trend_pct":  0.0,
        "confidence": min(100, score_data.get("total_score", 50)),
        "score":      0,
    }

    foreign_flow = {
        "direction":        ff_raw.get("direction", "NEUTRAL"),
        "net_buy_idr":      ff_raw.get("net_val_idr", 0),
        "consecutive_days": ff_raw.get("consecutive_buy_days", 0),
        "score":            layers.get("foreign", 0),
    }

    news_item = news_raw[0] if news_raw else {}
    news = {
        "headline":  news_item.get("headline", "Tidak ada berita terbaru"),
        "sentiment": news_item.get("sentiment", "NEUTRAL"),
        "items":     news_raw,
        "score":     0,
    }

    # ── Agent 1 ──────────────────────────────────────────────────────────────
    logger.info("Pipeline %s: running Agent 1 (Analysis)", symbol)
    analysis = run_analysis_agent(symbol, technical, forecast, foreign_flow, news, market_ctx)
    if analysis is None:
        logger.warning("Pipeline %s: Agent 1 failed", symbol)

    time.sleep(0.5)

    # ── Agent 2 ──────────────────────────────────────────────────────────────
    logger.info("Pipeline %s: running Agent 2 (Simulation)", symbol)
    simulation = run_simulation_agent(
        symbol, analysis or {}, news, technical, foreign_flow, market_ctx
    )
    if simulation is None:
        logger.warning("Pipeline %s: Agent 2 failed", symbol)

    time.sleep(0.5)

    # ── Agent 3 ──────────────────────────────────────────────────────────────
    logger.info("Pipeline %s: running Agent 3 (Conclusion)", symbol)
    conclusion = run_conclusion_agent(symbol, score_data, analysis or {}, simulation or {})
    if conclusion is None:
        logger.warning("Pipeline %s: Agent 3 failed", symbol)

    pipeline_ok = all(x is not None for x in (analysis, simulation, conclusion))

    return {
        "symbol":      symbol,
        "analysis":    analysis,
        "simulation":  simulation,
        "conclusion":  conclusion,
        "pipeline_ok": pipeline_ok,
    }


# ── Telegram formatter ────────────────────────────────────────────────────────

_ACTION_EMOJI  = {"BUY": "✅", "HOLD": "🔵", "SELL": "❌", "WATCH": "👀"}
_AGG_EMOJI     = {"BULLISH": "🟢", "NEUTRAL": "🟡", "BEARISH": "🔴"}
_PERSONA_SHORT = ["Retail", "Asing", "Fund mgr", "Scalper", "LT value"]


def format_pipeline_block(result: dict) -> str:
    """
    Formats pipeline output as Telegram HTML for the briefing.
    Returns empty string if pipeline_ok is False.
    """
    if not result.get("pipeline_ok"):
        return ""

    analysis   = result["analysis"]
    simulation = result["simulation"]
    conclusion = result["conclusion"]

    lines: list[str] = []

    # ── Agent 1 block ─────────────────────────────────────────────────────────
    lines.append("  🤖 <b>Agent Analysis</b>")
    thesis     = html.escape(str(analysis.get("price_thesis", "NEUTRAL")))
    confidence = analysis.get("confidence", 0)
    catalyst   = html.escape(str(analysis.get("catalyst", "—")))
    risk_text  = html.escape(str(analysis.get("risk", "—")))
    lines.append(f"    Thesis: <b>{thesis}</b> | Confidence: {confidence}%")
    lines.append(f"    Catalyst: {catalyst}")
    lines.append(f"    Risk: {risk_text}")
    lines.append("")

    # ── Agent 2 block ─────────────────────────────────────────────────────────
    lines.append("  👥 <b>Simulasi Pasar (5 personas)</b>")
    agg       = simulation.get("aggregate", "NEUTRAL")
    agg_emoji = _AGG_EMOJI.get(agg, "🟡")
    bull_pct  = simulation.get("bullish_pct", 0)
    neut_pct  = simulation.get("neutral_pct", 0)
    bear_pct  = simulation.get("bearish_pct", 0)
    open_move = simulation.get("simulated_open_move", 0) or 0
    sign      = "+" if open_move >= 0 else ""
    lines.append(
        f"    {agg_emoji} {agg} — {bull_pct}% beli / {neut_pct}% netral / {bear_pct}% jual"
    )
    lines.append(f"    📊 Estimasi open: {sign}{open_move:.1f}%")

    for i, p in enumerate(simulation.get("personas", [])[:5]):
        action = p.get("action", "WATCH")
        conf   = p.get("confidence", 0)
        reason = html.escape(str(p.get("reasoning", "—")))
        emoji  = _ACTION_EMOJI.get(action, "👀")
        name   = (
            _PERSONA_SHORT[i] if i < len(_PERSONA_SHORT)
            else html.escape(str(p.get("persona", f"Persona {i + 1}")))
        )
        lines.append(f"    {emoji} {name}: {action} ({conf}%) — {reason}")
    lines.append("")

    # ── Agent 3 block ─────────────────────────────────────────────────────────
    action     = conclusion.get("action", "PERHATIKAN")
    conviction = conclusion.get("conviction", "SEDANG")
    entry      = conclusion.get("entry", 0)
    t1         = conclusion.get("target_1", 0)
    t2         = conclusion.get("target_2", 0)
    sl         = conclusion.get("stop_loss", 0)
    lots       = conclusion.get("lots", 1)
    hold       = html.escape(str(conclusion.get("hold", "—")))
    alasan     = html.escape(str(conclusion.get("alasan", "—")))
    peringatan = html.escape(str(conclusion.get("peringatan", "—")))

    action_emoji = {"BELI": "✅", "PERHATIKAN": "⏳", "LEWATI": "⛔"}.get(action, "📌")
    lines.append(
        f"  {action_emoji} <b>Kesimpulan: {action}</b> (Keyakinan: {conviction})"
    )

    def _fmt(v) -> str:
        return f"{v:,.0f}" if isinstance(v, (int, float)) and v else "—"

    lines.append(
        f"    Entry: {_fmt(entry)} | Target: {_fmt(t1)} / {_fmt(t2)} | SL: {_fmt(sl)}"
    )
    lines.append(f"    Lots: {lots} | Hold: {hold}")
    lines.append(f"    Alasan: {alasan}")
    lines.append(f"    ⚠️ Peringatan: {peringatan}")

    return "\n".join(lines)


# ── Standalone /analyze formatter ─────────────────────────────────────────────

def format_analyze_message(pipeline_result: dict, score_data: dict) -> str:
    """
    Full standalone Telegram HTML message for the /analyze command.
    Wraps format_pipeline_block() with a header row.
    """
    sym   = pipeline_result.get("symbol", "")
    code  = sym.replace(".JK", "")
    snap  = score_data.get("snapshot", {})
    close = snap.get("close")
    rsi   = snap.get("rsi")

    conclusion = pipeline_result.get("conclusion") or {}
    score = conclusion.get("final_score") or score_data.get("total_score", "—")

    close_str = f"<code>{close:,.0f}</code>  RSI: {rsi:.1f}" if close and rsi else ""

    header = [
        f"<b>🤖 ANALISA 3-AGEN: {code}</b>  <i>(skor {score})</i>",
        close_str,
        "",
    ]

    block = format_pipeline_block(pipeline_result)
    return "\n".join(l for l in header if l is not None) + "\n" + block


# ── Multi-timeframe context ───────────────────────────────────────────────────

def build_timeframe_ctx(df_ind) -> dict:
    """Compute weekly and monthly trend/range context from daily indicator DataFrame."""
    try:
        import pandas as pd

        ohlc = df_ind[["open", "high", "low", "close"]].copy()

        # Weekly (last 12 weeks)
        weekly = ohlc.resample("W").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna().tail(12)
        w_trend = "N/A"
        if len(weekly) >= 4:
            w_trend = (
                "UPTREND"   if weekly["close"].iloc[-1] > weekly["close"].iloc[-4]
                else "DOWNTREND" if weekly["close"].iloc[-1] < weekly["close"].iloc[-4]
                else "SIDEWAYS"
            )

        # Monthly (last 6 months) — try "ME" (pandas >=2.2) then fall back to "M"
        monthly = pd.DataFrame()
        for freq in ("ME", "MS", "M"):
            try:
                monthly = ohlc.resample(freq).agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last"}
                ).dropna().tail(6)
                break
            except Exception:
                continue
        m_trend = "N/A"
        if len(monthly) >= 3:
            m_trend = (
                "UPTREND"   if monthly["close"].iloc[-1] > monthly["close"].iloc[-3]
                else "DOWNTREND" if monthly["close"].iloc[-1] < monthly["close"].iloc[-3]
                else "SIDEWAYS"
            )

        return {
            "weekly_high":   float(weekly["high"].max())  if len(weekly) else None,
            "weekly_low":    float(weekly["low"].min())   if len(weekly) else None,
            "weekly_trend":  w_trend,
            "monthly_high":  float(monthly["high"].max()) if len(monthly) else None,
            "monthly_low":   float(monthly["low"].min())  if len(monthly) else None,
            "monthly_trend": m_trend,
        }
    except Exception as e:
        logger.debug("build_timeframe_ctx error: %s", e)
        return {}


def build_sr_levels(df_ind) -> dict:
    """Compute support/resistance from rolling-period highs and lows."""
    try:
        high  = df_ind["high"]
        low   = df_ind["low"]
        close = df_ind["close"]
        last  = float(close.iloc[-1])

        r20 = float(high.tail(20).max())
        s20 = float(low.tail(20).min())
        r60 = float(high.tail(60).max())
        s60 = float(low.tail(60).min())

        # Sort by proximity to current price; exclude levels equal to current price
        resistances = sorted(
            [x for x in (r20, r60) if x > last * 1.001],
            key=lambda x: x - last,
        )
        supports = sorted(
            [x for x in (s20, s60) if x < last * 0.999],
            key=lambda x: last - x,
        )

        return {
            "support_1":    supports[0]    if len(supports) > 0 else s20,
            "support_2":    supports[1]    if len(supports) > 1 else s60,
            "resistance_1": resistances[0] if len(resistances) > 0 else r20,
            "resistance_2": resistances[1] if len(resistances) > 1 else r60,
        }
    except Exception as e:
        logger.debug("build_sr_levels error: %s", e)
        return {}


# ── Research Agent (Sonnet) ───────────────────────────────────────────────────

_RESEARCH_SYSTEM = (
    "Kamu adalah analis saham profesional IDX Indonesia. "
    "Tugasmu: laporan riset komprehensif 5 bagian untuk satu emiten. "
    "Gabungkan data teknikal aktual yang diberikan dengan pengetahuanmu "
    "tentang fundamental, valuasi, dan sentimen emiten tersebut. "
    "Tandai jika data fundamental mungkin sudah kadaluarsa. "
    "Respond only in valid JSON."
)


def run_research_agent(
    symbol:     str,
    score_data: dict,
    tf_ctx:     dict,
    sr_levels:  dict,
    market_ctx: dict,
) -> dict | None:
    snap   = score_data.get("snapshot", {})
    layers = score_data.get("layer_scores", {})
    ff_raw = score_data.get("foreign") or {}
    levels = score_data.get("trade_levels") or {}
    price  = snap.get("close", 0) or 0
    code   = symbol.replace(".JK", "")

    def _p(v) -> str:
        return f"{v:,.0f}" if isinstance(v, (int, float)) and v else "N/A"

    user_prompt = (
        f"Buat laporan riset komprehensif untuk saham <b>{code}</b> (IDX).\n\n"
        f"== DATA AKTUAL (dari market data) ==\n\n"
        f"HARGA & TEKNIKAL HARIAN:\n"
        f"- Harga: Rp {price:,.0f}\n"
        f"- RSI (14): {snap.get('rsi', 'N/A')}\n"
        f"- MACD: {_macd_status(snap)}\n"
        f"- BB Position: {_bb_position(snap)}\n"
        f"- MA Trend: {_ma_trend(snap)}\n"
        f"- Volume Ratio: {snap.get('vol_ratio', 'N/A')}x\n"
        f"- Divergence: {snap.get('divergence', 'NONE')}\n"
        f"- 52W High: Rp {_p(snap.get('high_52w'))} | Low: Rp {_p(snap.get('low_52w'))}\n"
        f"- % dari 52W High: {snap.get('pct_from_52w_high', 'N/A')}%\n\n"
        f"MULTI-TIMEFRAME:\n"
        f"- Weekly (12W): Tren {tf_ctx.get('weekly_trend','N/A')} | "
        f"High Rp {_p(tf_ctx.get('weekly_high'))} | Low Rp {_p(tf_ctx.get('weekly_low'))}\n"
        f"- Monthly (6M): Tren {tf_ctx.get('monthly_trend','N/A')} | "
        f"High Rp {_p(tf_ctx.get('monthly_high'))} | Low Rp {_p(tf_ctx.get('monthly_low'))}\n\n"
        f"SUPPORT & RESISTANCE:\n"
        f"- Support 1: Rp {_p(sr_levels.get('support_1'))} | Support 2: Rp {_p(sr_levels.get('support_2'))}\n"
        f"- Resistance 1: Rp {_p(sr_levels.get('resistance_1'))} | Resistance 2: Rp {_p(sr_levels.get('resistance_2'))}\n\n"
        f"FOREIGN FLOW:\n"
        f"- Arah: {ff_raw.get('direction','NEUTRAL')} | "
        f"Net: Rp {abs(ff_raw.get('net_val_idr',0))/1e9:.1f}B IDR | "
        f"Konsekutif: {ff_raw.get('consecutive_buy_days',0)} hari\n\n"
        f"SCORE INTERNAL: {score_data.get('total_score',0)}/100 — {score_data.get('verdict','WATCH')}\n"
        f"(Trend {layers.get('trend',0)}/25 | Mom {layers.get('momentum',0)}/20 | "
        f"Vol {layers.get('volume',0)}/20 | Pat {layers.get('pattern',0)}/15 | "
        f"FF {layers.get('foreign',0)}/15)\n\n"
        f"KONTEKS PASAR: IHSG {market_ctx.get('ihsg_change',0):+.2f}% | "
        f"USD/IDR {market_ctx.get('usd_idr','—')}\n\n"
        f"== GUNAKAN PENGETAHUANMU UNTUK {code} ==\n\n"
        "Lengkapi analisis berikut (tandai jika data fundamental mungkin kadaluarsa):\n"
        "- Profil: sektor, subsektor, model bisnis, posisi kompetitif, estimasi market cap\n"
        "- Fundamental: kinerja keuangan (revenue, laba, margin, ROE, ROA, DER)\n"
        "- Valuasi: PER, PBV, EV/EBITDA, dividend yield vs rata-rata sektor\n"
        "- Sentimen: isu terkini, aksi korporasi, pandangan analis\n"
        "- Kompetitor: 2 kompetitor utama di sektor yang sama\n\n"
        "Return JSON dengan schema ini:\n"
        "{\n"
        '  "profil": {\n'
        '    "sektor": "...", "subsektor": "...", "market_cap_est": "...",\n'
        '    "bisnis": "2-3 kalimat model bisnis dan sumber pendapatan",\n'
        '    "kompetitif": "1-2 kalimat posisi vs pesaing"\n'
        "  },\n"
        '  "fundamental": {\n'
        '    "kinerja": "3-4 kalimat kinerja keuangan terbaru dengan angka nyata",\n'
        '    "valuasi": "2-3 kalimat PER/PBV/dividend vs sektor",\n'
        '    "prospek": "2-3 kalimat proyeksi industri dan strategi bisnis",\n'
        '    "kesimpulan": "1-2 kalimat kesimpulan fundamental",\n'
        '    "data_note": "tahun data atau catatan keakuratan"\n'
        "  },\n"
        '  "teknikal": {\n'
        '    "tren_harian": "UPTREND|DOWNTREND|SIDEWAYS",\n'
        '    "tren_mingguan": "UPTREND|DOWNTREND|SIDEWAYS",\n'
        '    "tren_bulanan": "UPTREND|DOWNTREND|SIDEWAYS",\n'
        '    "support_1": <float>, "support_2": <float>,\n'
        '    "resistance_1": <float>, "resistance_2": <float>,\n'
        '    "sinyal": "BUY|SELL|NEUTRAL",\n'
        '    "pattern": "nama pola atau none",\n'
        '    "target_pendek": <float>, "target_menengah": <float>,\n'
        '    "kesimpulan": "2-3 kalimat kesimpulan teknikal"\n'
        "  },\n"
        '  "sentimen": {\n'
        '    "asing": "1-2 kalimat aktivitas investor asing",\n'
        '    "katalis": "1-2 kalimat katalis utama",\n'
        '    "aksi_korporasi": "aksi korporasi terbaru atau N/A",\n'
        '    "risiko": "1-2 kalimat risiko utama",\n'
        '    "kesimpulan": "1-2 kalimat kesimpulan sentimen"\n'
        "  },\n"
        '  "rekomendasi": {\n'
        '    "action": "BUY|HOLD|SELL",\n'
        '    "conviction": "TINGGI|SEDANG|RENDAH",\n'
        '    "entry": <float>, "target_1": <float>, "target_2": <float>, "stop_loss": <float>,\n'
        '    "horizon": "jangka pendek|jangka menengah|jangka panjang",\n'
        '    "ringkasan": "2-3 kalimat Bahasa Indonesia",\n'
        '    "risiko_utama": "1 kalimat Bahasa Indonesia"\n'
        "  },\n"
        '  "kompetitor": "2-3 kalimat perbandingan dengan 2 kompetitor utama",\n'
        '  "disclaimer": "Data fundamental berdasarkan pengetahuan AI — verifikasi dengan laporan emiten terbaru."\n'
        "}"
    )

    try:
        resp = _client.messages.create(
            model=_ANALYSIS_MODEL,
            max_tokens=2000,
            system=_RESEARCH_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return _parse_json(resp.content[0].text, label=f"Research/{symbol}")
    except Exception as e:
        logger.error("Research agent/%s failed: %s", symbol, e)
        return None


# ── Research formatter ────────────────────────────────────────────────────────

_WIB = pytz.timezone("Asia/Jakarta")

_SINYAL_EMOJI  = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "🟡"}
_ACTION_REC_EMOJI = {"BUY": "✅", "HOLD": "⏳", "SELL": "❌"}
_TREN_ARROW    = {"UPTREND": "↑", "DOWNTREND": "↓", "SIDEWAYS": "→"}


def format_research_sections(result: dict, symbol: str) -> list[str]:
    """
    Format research output as a list of Telegram HTML strings — one per bagian.

    Each string is a complete, self-contained message guaranteed to be under
    Telegram's 4096-char limit.  Send them sequentially; do NOT concatenate.
    """
    now  = datetime.now(_WIB).strftime("%d %b %Y %H:%M WIB")
    code = symbol.replace(".JK", "")

    def _e(v) -> str:
        return html.escape(str(v)) if v else "—"

    def _p(v) -> str:
        return f"{v:,.0f}" if isinstance(v, (int, float)) and v else "—"

    def _tr(v) -> str:
        return f"{_TREN_ARROW.get(str(v), '?')} {_e(v)}"

    sections: list[str] = []

    # ── (1/5) Profil ──────────────────────────────────────────────────────────
    p = result.get("profil", {})
    sections.append("\n".join([
        f"<b>🔬 RISET: {code}</b>  <i>{now}</i>",
        "<b>(1/5) PROFIL EMITEN</b>",
        "",
        f"Sektor: <b>{_e(p.get('sektor'))}</b> — {_e(p.get('subsektor'))}",
        f"Est. Market Cap: {_e(p.get('market_cap_est'))}",
        "",
        f"<b>Bisnis:</b> {_e(p.get('bisnis'))}",
        f"<b>Kompetitif:</b> {_e(p.get('kompetitif'))}",
    ]))

    # ── (2/5) Fundamental ─────────────────────────────────────────────────────
    f2 = result.get("fundamental", {})
    sec2 = [
        f"<b>🔬 RISET: {code}  (2/5) FUNDAMENTAL</b>",
        "",
        f"📈 <b>Kinerja Keuangan</b>",
        _e(f2.get("kinerja")),
        "",
        f"💰 <b>Valuasi</b>",
        _e(f2.get("valuasi")),
        "",
        f"🔭 <b>Prospek</b>",
        _e(f2.get("prospek")),
        "",
        f"<i>Kesimpulan: {_e(f2.get('kesimpulan'))}</i>",
    ]
    if f2.get("data_note"):
        sec2.append(f"<i>⚠️ {_e(f2.get('data_note'))}</i>")
    sections.append("\n".join(sec2))

    # ── (3/5) Teknikal ────────────────────────────────────────────────────────
    t      = result.get("teknikal", {})
    sinyal = str(t.get("sinyal", "NEUTRAL"))
    sections.append("\n".join([
        f"<b>🔬 RISET: {code}  (3/5) TEKNIKAL</b>",
        "",
        f"Tren: Harian {_tr(t.get('tren_harian'))} | "
        f"Mingguan {_tr(t.get('tren_mingguan'))} | "
        f"Bulanan {_tr(t.get('tren_bulanan'))}",
        "",
        f"Support   : <code>{_p(t.get('support_1'))} / {_p(t.get('support_2'))}</code>",
        f"Resistansi: <code>{_p(t.get('resistance_1'))} / {_p(t.get('resistance_2'))}</code>",
        "",
        f"Pola: {_e(t.get('pattern', 'none'))}",
        f"Sinyal: {_SINYAL_EMOJI.get(sinyal, '🟡')} <b>{sinyal}</b>",
        f"Target Pendek  : <code>{_p(t.get('target_pendek'))}</code>",
        f"Target Menengah: <code>{_p(t.get('target_menengah'))}</code>",
        "",
        f"<i>Kesimpulan: {_e(t.get('kesimpulan'))}</i>",
    ]))

    # ── (4/5) Sentimen ────────────────────────────────────────────────────────
    s = result.get("sentimen", {})
    sections.append("\n".join([
        f"<b>🔬 RISET: {code}  (4/5) SENTIMEN &amp; KATALIS</b>",
        "",
        f"<b>Asing:</b> {_e(s.get('asing'))}",
        "",
        f"<b>Katalis:</b> {_e(s.get('katalis'))}",
        "",
        f"<b>Aksi Korporasi:</b> {_e(s.get('aksi_korporasi', 'N/A'))}",
        "",
        f"<b>Risiko:</b> {_e(s.get('risiko'))}",
        "",
        f"<i>Kesimpulan: {_e(s.get('kesimpulan'))}</i>",
    ]))

    # ── (5/5) Rekomendasi + Kompetitor + Disclaimer ───────────────────────────
    r          = result.get("rekomendasi", {})
    action     = str(r.get("action", "HOLD"))
    conviction = _e(r.get("conviction", "SEDANG"))
    a_emoji    = _ACTION_REC_EMOJI.get(action, "📌")
    sec5 = [
        f"<b>🔬 RISET: {code}  (5/5) REKOMENDASI</b>",
        "",
        f"{a_emoji} <b>{action}</b> (Keyakinan: {conviction})",
        f"Horizon: {_e(r.get('horizon'))}",
        "",
        f"Entry    : <code>{_p(r.get('entry'))}</code>",
        f"Target 1 : <code>{_p(r.get('target_1'))}</code>",
        f"Target 2 : <code>{_p(r.get('target_2'))}</code>",
        f"Stop Loss: <code>{_p(r.get('stop_loss'))}</code>",
        "",
        f"📝 {_e(r.get('ringkasan'))}",
        f"⚠️ {_e(r.get('risiko_utama'))}",
    ]
    komp = result.get("kompetitor", "")
    if komp:
        sec5 += ["", "<b>Kompetitor:</b>", _e(komp)]
    disc = result.get("disclaimer", "")
    if disc:
        sec5 += ["", f"<i>⚠️ {_e(disc)}</i>"]
    sections.append("\n".join(sec5))

    return sections


# ── EOD Report helpers ────────────────────────────────────────────────────────

def _pick_status_label(p: dict) -> str:
    today  = p.get("today_close") or 0
    target = p.get("target")      or 0
    sl     = p.get("stop_loss")   or 0
    entry  = p.get("entry")       or 0
    if target and today >= target:
        return "target ✅"
    if sl and today <= sl:
        return "SL kena 🛑"
    if entry and today >= entry:
        return "di atas entry"
    return "di bawah entry"


# ── EOD Report Agent (Haiku) ──────────────────────────────────────────────────

_EOD_SYSTEM = (
    "Kamu adalah analis pasar saham IDX Indonesia yang memberikan ringkasan "
    "akhir hari perdagangan. Berikan narasi singkat (3-4 kalimat), padat, "
    "dan insightful dalam Bahasa Indonesia. Respond only in valid JSON."
)


def run_eod_report_agent(
    picks_perf:     list[dict],
    watchlist_perf: list[dict],
    market_data:    dict,
) -> dict | None:
    ihsg_chg = market_data.get("ihsg_change_pct", 0) or 0
    ff_dir   = market_data.get("foreign_direction", "—")
    ff_net   = abs(market_data.get("foreign_net_idr", 0)) / 1e9

    picks_lines = []
    for p in picks_perf:
        sym    = p["symbol"].replace(".JK", "")
        pct    = p.get("pct_change", 0) or 0
        status = _pick_status_label(p)
        picks_lines.append(f"  {sym}: {pct:+.1f}% ({status})")

    watch_lines = []
    for p in watchlist_perf:
        sym = p["symbol"].replace(".JK", "")
        pct = p.get("pct_change", 0) or 0
        watch_lines.append(f"  {sym}: {pct:+.1f}%")

    picks_str = "\n".join(picks_lines) if picks_lines else "  (tidak ada)"
    watch_str = "\n".join(watch_lines) if watch_lines else "  (tidak ada)"

    top_syms = [p["symbol"].replace(".JK", "") for p in picks_perf[:3]]
    top_syms += [p["symbol"].replace(".JK", "") for p in watchlist_perf[:2]]

    user_prompt = (
        f"Buat ringkasan akhir hari perdagangan IDX.\n\n"
        f"DATA PASAR HARI INI:\n"
        f"- IHSG: {ihsg_chg:+.2f}%\n"
        f"- Asing: {ff_dir} Rp {ff_net:.1f}B IDR\n\n"
        f"PERFORMA PICKS:\n{picks_str}\n\n"
        f"WATCHLIST:\n{watch_str}\n\n"
        "Tulis ringkasan akhir hari dalam Bahasa Indonesia. "
        "Fokus pada: kondisi market hari ini, highlight picks yang menonjol, "
        "dan pandangan besok.\n\n"
        f"Saham relevan untuk besok: {', '.join(top_syms)}\n\n"
        "Return JSON:\n"
        "{\n"
        '  "narrative": "<3-4 kalimat ringkasan kondisi pasar dan picks>",\n'
        '  "tomorrow_watch": ["SYM1", "SYM2", "SYM3"],\n'
        '  "sentiment": "POSITIF|NEGATIF|MIXED"\n'
        "}"
    )

    try:
        resp = _client.messages.create(
            model=_CONCLUSION_MODEL,
            max_tokens=300,
            system=_EOD_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return _parse_json(resp.content[0].text, label="EOD/agent")
    except Exception as e:
        logger.error("EOD agent failed: %s", e)
        return None


def format_eod_report(
    picks_perf:     list[dict],
    watchlist_perf: list[dict],
    market_data:    dict,
    agent_result:   dict | None,
) -> str:
    now    = datetime.now(_WIB)
    today  = now.strftime("%d %b %Y")

    ihsg_close = market_data.get("ihsg_close", 0) or 0
    ihsg_chg   = market_data.get("ihsg_change_pct", 0) or 0
    ff_dir     = market_data.get("foreign_direction", "—")
    ff_net     = abs(market_data.get("foreign_net_idr", 0)) / 1e9

    def _pct_emoji(pct: float) -> str:
        if pct >= 1.0:   return "🟢"
        if pct <= -1.5:  return "🔴"
        return "🟡"

    def _fmt_px(v) -> str:
        return f"<code>{v:,.0f}</code>" if v else ""

    lines: list[str] = []
    lines.append(f"<b>📊 LAPORAN PASAR — {today}</b>")
    lines.append("")

    # IHSG + foreign
    ihsg_sign = "+" if ihsg_chg >= 0 else ""
    ihsg_str  = _fmt_px(ihsg_close) if ihsg_close else "<i>N/A</i>"
    ihsg_chg_str = f"<i>{ihsg_sign}{ihsg_chg:.2f}%</i>" if ihsg_chg else ""
    lines.append(f"🏛 <b>IHSG:</b> {ihsg_str}  {ihsg_chg_str}".strip())
    lines.append(f"💱 Asing: {html.escape(ff_dir)} Rp {ff_net:.1f}B")
    lines.append("")

    # Picks
    lines.append("<b>📋 PICKS HARI INI</b>")
    if picks_perf:
        for p in picks_perf:
            sym      = p["symbol"].replace(".JK", "")
            pct      = p.get("pct_change", 0) or 0
            today_px = p.get("today_close", 0) or 0
            emoji    = _pct_emoji(pct)
            sign     = "+" if pct >= 0 else ""
            status   = html.escape(_pick_status_label(p))
            lines.append(
                f"{emoji} <code>{sym:<5}</code> {_fmt_px(today_px)}  "
                f"<b>{sign}{pct:.1f}%</b>  {status}"
            )
    else:
        lines.append("<i>Tidak ada picks hari ini</i>")
    lines.append("")

    # Watchlist
    if watchlist_perf:
        lines.append("<b>👁 WATCHLIST</b>")
        for p in watchlist_perf:
            sym      = p["symbol"].replace(".JK", "")
            pct      = p.get("pct_change", 0) or 0
            today_px = p.get("today_close", 0) or 0
            emoji    = _pct_emoji(pct)
            sign     = "+" if pct >= 0 else ""
            lines.append(
                f"{emoji} <code>{sym:<5}</code> {_fmt_px(today_px)}  "
                f"<b>{sign}{pct:.1f}%</b>"
            )
        lines.append("")

    # AI narrative
    if agent_result:
        narrative = agent_result.get("narrative", "")
        if narrative:
            lines.append("<b>🤖 NARASI PASAR</b>")
            lines.append(html.escape(narrative))
            lines.append("")

        tomorrow = agent_result.get("tomorrow_watch") or []
        if tomorrow:
            watch_str = " · ".join(
                html.escape(s.replace(".JK", "")) for s in tomorrow[:5]
            )
            lines.append(f"📌 <b>Besok pantau:</b> {watch_str}")
            lines.append("")

    lines.append(
        f"<i>Data ~{now.strftime('%H:%M')} WIB · Bukan rekomendasi investasi</i>"
    )

    return "\n".join(lines)
