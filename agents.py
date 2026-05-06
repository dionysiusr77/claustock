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

import anthropic

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
