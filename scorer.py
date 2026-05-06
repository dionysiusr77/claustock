"""
Scoring engine — converts indicator snapshots into a 0-100 score.

Layer breakdown (weights must sum to 100):
  Trend     25 pts — EMA alignment
  Momentum  20 pts — RSI zone, MACD cross, Stochastic  (+divergence bonus/penalty)
  Volume    20 pts — Volume ratio, OBV trend
  Pattern   15 pts — Candlestick + breakout
  Foreign   15 pts — IDX net foreign flow
  Breadth    5 pts — IHSG + sector direction

Divergence modifiers (applied after all layers):
  BULLISH divergence        → +5 bonus
  HIDDEN_BULLISH divergence → +3 bonus
  BEARISH divergence        → -10 penalty  (hard disqualifier in screener)
  HIDDEN_BEARISH divergence → -5 penalty

Final score is clamped to [0, 100].
"""

import logging

import numpy as np

import config
from patterns import detect_patterns

logger = logging.getLogger(__name__)


# ── Layer 1: Trend ────────────────────────────────────────────────────────────

def _score_trend(snap: dict) -> tuple[int, list[str]]:
    """
    Scores EMA alignment. Max 25 pts.
    Full bull stack = price > EMA20 > EMA50 > EMA200.
    """
    price = snap.get("close")
    e20   = snap.get("ema20")
    e50   = snap.get("ema50")
    e200  = snap.get("ema200")

    if any(v is None for v in (price, e20, e50, e200)):
        return 5, ["EMA data incomplete — partial credit"]

    conditions = [
        price > e20,
        e20   > e50,
        e50   > e200,
        price > e200,
    ]
    met = sum(conditions)

    scale = {4: 25, 3: 18, 2: 10, 1: 5, 0: 0}
    score = scale[met]

    if met == 4:
        reasons = ["Full bull EMA stack (price > EMA20 > EMA50 > EMA200)"]
    elif met == 3:
        above_200 = "above" if price > e200 else "below"
        reasons = [f"Strong EMA alignment, price {above_200} EMA200"]
    elif met == 2:
        reasons = ["Partial EMA alignment — mixed trend"]
    elif met == 1:
        reasons = ["Weak EMA alignment — trend unclear"]
    else:
        reasons = ["Bearish EMA stack — all EMAs above price"]

    return score, reasons


# ── Layer 2: Momentum ─────────────────────────────────────────────────────────

def _score_momentum(snap: dict) -> tuple[int, list[str]]:
    """
    RSI zone (8 pts) + MACD position (7 pts) + Stochastic (5 pts). Max 20 pts.
    Divergence bonus/penalty applied separately in score_stock().
    """
    rsi       = snap.get("rsi")
    macd      = snap.get("macd")
    macd_sig  = snap.get("macd_signal")
    macd_hist = snap.get("macd_hist")
    stoch_k   = snap.get("stoch_k")
    stoch_d   = snap.get("stoch_d")

    score   = 0
    reasons = []

    # ── RSI (0-8 pts) ────────────────────────────────────────────────────────
    if rsi is not None:
        if config.RSI_SWEET_SPOT_LOW <= rsi <= config.RSI_SWEET_SPOT_HIGH:
            score += 8
            reasons.append(f"RSI {rsi:.1f} in sweet spot ({config.RSI_SWEET_SPOT_LOW}–{config.RSI_SWEET_SPOT_HIGH})")
        elif config.RSI_SWEET_SPOT_HIGH < rsi <= config.RSI_OVERBOUGHT:
            score += 6
            reasons.append(f"RSI {rsi:.1f} — strong momentum, approaching overbought")
        elif config.RSI_OVERSOLD < rsi < config.RSI_SWEET_SPOT_LOW:
            score += 5
            reasons.append(f"RSI {rsi:.1f} — recovering from oversold")
        elif rsi <= config.RSI_OVERSOLD:
            score += 3
            reasons.append(f"RSI {rsi:.1f} — oversold, reversal watch")
        else:  # > RSI_OVERBOUGHT
            score += 1
            reasons.append(f"RSI {rsi:.1f} — overbought, caution")

    # ── MACD (0-7 pts) ───────────────────────────────────────────────────────
    if macd is not None and macd_sig is not None:
        bull_cross = macd > macd_sig
        above_zero = macd > 0
        hist_growing = macd_hist is not None and macd_hist > 0

        if bull_cross and above_zero and hist_growing:
            score += 7
            reasons.append("MACD bull cross, above zero, histogram expanding")
        elif bull_cross and hist_growing:
            score += 5
            reasons.append("MACD bull cross, histogram expanding (below zero)")
        elif bull_cross:
            score += 3
            reasons.append("MACD above signal line")
        else:
            reasons.append("MACD below signal line")

    # ── Stochastic (0-5 pts) ─────────────────────────────────────────────────
    if stoch_k is not None and stoch_d is not None:
        if 20 <= stoch_k <= 80 and stoch_k > stoch_d:
            score += 5
            reasons.append(f"Stoch K {stoch_k:.1f} > D {stoch_d:.1f}, in neutral zone")
        elif stoch_k < 20:
            score += 3
            reasons.append(f"Stoch K {stoch_k:.1f} oversold — potential bounce")
        elif stoch_k > 80:
            score += 1
            reasons.append(f"Stoch K {stoch_k:.1f} overbought")

    return min(score, 20), reasons


# ── Layer 3: Volume ───────────────────────────────────────────────────────────

def _score_volume(snap: dict, df_tail20=None) -> tuple[int, list[str]]:
    """
    Volume ratio (0-12 pts) + OBV trend (0-8 pts). Max 20 pts.
    df_tail20: last 20 rows of the indicator DataFrame for OBV slope — pass None to skip OBV.
    """
    vol_ratio = snap.get("vol_ratio")
    score     = 0
    reasons   = []

    # ── Volume ratio ─────────────────────────────────────────────────────────
    if vol_ratio is not None:
        if vol_ratio >= 2.0:
            score += 12
            reasons.append(f"Volume {vol_ratio:.1f}× avg — very high conviction")
        elif vol_ratio >= 1.5:
            score += 9
            reasons.append(f"Volume {vol_ratio:.1f}× avg — above average")
        elif vol_ratio >= 1.2:
            score += 6
            reasons.append(f"Volume {vol_ratio:.1f}× avg — slightly elevated")
        elif vol_ratio >= 0.8:
            score += 3
            reasons.append(f"Volume {vol_ratio:.1f}× avg — average")
        else:
            reasons.append(f"Volume {vol_ratio:.1f}× avg — below average, low conviction")

    # ── OBV slope ────────────────────────────────────────────────────────────
    if df_tail20 is not None and "obv" in df_tail20.columns:
        obv = df_tail20["obv"].dropna()
        if len(obv) >= 5:
            slope = np.polyfit(range(len(obv)), obv.values, 1)[0]
            if slope > 0:
                score += 8
                reasons.append("OBV rising — volume-price trend bullish")
            elif slope > -abs(obv.mean()) * 0.001:
                score += 4
                reasons.append("OBV flat — neutral accumulation")
            else:
                reasons.append("OBV falling — distribution pressure")

    return min(score, 20), reasons


# ── Layer 4: Pattern ──────────────────────────────────────────────────────────

def _score_pattern(snap: dict, df) -> tuple[int, list[str]]:
    """
    Breakout (up to 15 pts) or candlestick patterns (up to 12 pts). Max 15 pts.
    Bearish candlestick patterns add a warning to reasons (penalty in caller).
    """
    is_breakout = snap.get("breakout_bull", False)
    vol_ratio   = snap.get("vol_ratio", 1.0) or 1.0
    score       = 0
    reasons     = []

    pat = detect_patterns(df) if df is not None and len(df) >= 3 else {"bullish": [], "bearish": [], "strongest_bull": ""}

    # ── Breakout takes precedence ─────────────────────────────────────────────
    if is_breakout:
        if vol_ratio >= 1.5:
            score = 15
            reasons.append(f"Breakout above {config.BREAKOUT_PERIOD}-day high with volume {vol_ratio:.1f}× avg")
        else:
            score = 8
            reasons.append(f"Breakout above {config.BREAKOUT_PERIOD}-day high — low volume, watch for retest")
    elif pat["strongest_bull"]:
        pattern_scores = {
            "MORNING_STAR":      12,
            "BULLISH_ENGULFING": 11,
            "HAMMER":             8,
            "INVERTED_HAMMER":    7,
            "INSIDE_BAR":         5,
        }
        score = pattern_scores.get(pat["strongest_bull"], 5)
        reasons.append(f"Candlestick: {pat['strongest_bull']}")

    # ── Bearish patterns → warning (penalty applied in score_stock) ──────────
    if pat["bearish"]:
        reasons.append(f"WARNING — bearish pattern: {', '.join(pat['bearish'])}")

    return min(score, 15), reasons, pat["bearish"]


# ── Layer 5: Foreign flow ─────────────────────────────────────────────────────

def _score_foreign(foreign: dict | None) -> tuple[int, list[str]]:
    """
    Net foreign direction + consecutive days. Max 15 pts.
    Defaults to neutral (7 pts) when data is unavailable.
    """
    if not foreign:
        return 7, ["Foreign flow data unavailable — neutral assumed"]

    direction = foreign.get("direction", "NEUTRAL")
    net       = foreign.get("net_val_idr", 0) or 0
    consec    = foreign.get("consecutive_buy_days", 0) or 0
    net_b     = abs(net) / 1_000_000_000   # convert to billions IDR

    score   = 0
    reasons = []

    if direction == "BUY":
        if consec >= 3:
            score = 15
            reasons.append(f"Foreign net buy {consec}+ consecutive days ({net_b:.1f}B IDR)")
        elif consec == 2:
            score = 12
            reasons.append(f"Foreign net buy 2 consecutive days ({net_b:.1f}B IDR)")
        else:
            score = 8
            reasons.append(f"Foreign net buy today ({net_b:.1f}B IDR)")
    elif direction == "NEUTRAL":
        score = 5
        reasons.append("Foreign flow neutral")
    else:  # SELL
        score = 0
        reasons.append(f"Foreign net sell ({net_b:.1f}B IDR) — caution")

    return min(score, 15), reasons


# ── Layer 6: Market breadth ───────────────────────────────────────────────────

def _score_breadth(breadth: dict, sector: str | None = None) -> tuple[int, list[str]]:
    """
    IHSG direction (3 pts) + sector direction (2 pts). Max 5 pts.
    """
    if not breadth:
        return 2, ["Market breadth data unavailable — partial credit"]

    ihsg = breadth.get("IHSG", {})
    score   = 0
    reasons = []

    ihsg_dir = ihsg.get("direction", "FLAT")
    if ihsg_dir == "UP":
        score += 3
        reasons.append(f"IHSG up {ihsg.get('change_pct', 0):.2f}%")
    elif ihsg_dir == "FLAT":
        score += 1
        reasons.append("IHSG flat")
    else:
        reasons.append(f"IHSG down {ihsg.get('change_pct', 0):.2f}% — headwind")

    if sector and sector in breadth:
        sec_dir = breadth[sector].get("direction", "FLAT")
        if sec_dir == "UP":
            score += 2
            reasons.append(f"Sector ({sector}) up")
        elif sec_dir == "FLAT":
            score += 1
        # else: 0 pts

    return min(score, 5), reasons


# ── Trade levels ──────────────────────────────────────────────────────────────

def calc_trade_levels(snap: dict) -> dict | None:
    """
    Compute entry, target, stop loss using ATR-based SL.
    Returns None if R:R < MIN_RR after fees.

    SL = entry − 1.5 × ATR  (gives room for daily noise)
    Target chosen to meet both MIN_RR and MIN_TARGET_PCT after fees.
    """
    price = snap.get("close")
    atr   = snap.get("atr")
    if not price or not atr or atr == 0:
        return None

    entry  = float(price)
    sl     = entry - 1.5 * atr
    sl_pct = (entry - sl) / entry * 100

    # Minimum gross target to clear round-trip fee + required net gain
    min_gross_pct = config.MIN_TARGET_PCT + config.FEE_RT * 100
    target_pct    = max(sl_pct * config.MIN_RR, min_gross_pct)
    target        = entry * (1 + target_pct / 100)
    rr            = target_pct / sl_pct if sl_pct > 0 else 0

    if rr < config.MIN_RR:
        return None

    return {
        "entry":      round(entry),
        "target":     round(target),
        "stop_loss":  round(sl),
        "target_pct": round(target_pct, 2),
        "sl_pct":     round(sl_pct, 2),
        "rr":         round(rr, 2),
    }


# ── Setup classifier ──────────────────────────────────────────────────────────

def classify_setup(snap: dict, layer_scores: dict) -> str:
    """
    Name the primary setup based on dominant scoring signals.
    Returns one of:
      BREAKOUT | OVERSOLD_BOUNCE | MOMENTUM_CONTINUATION |
      PULLBACK | FOREIGN_ACCUMULATION | WATCH
    """
    rsi       = snap.get("rsi", 50) or 50
    div       = snap.get("divergence", "NONE")
    is_bo     = snap.get("breakout_bull", False)
    trend_s   = layer_scores.get("trend", 0)
    foreign_s = layer_scores.get("foreign", 0)

    if is_bo and layer_scores.get("volume", 0) >= 9:
        return "BREAKOUT"

    if rsi <= config.RSI_OVERSOLD or div in ("BULLISH",):
        return "OVERSOLD_BOUNCE"

    if config.RSI_SWEET_SPOT_LOW <= rsi <= config.RSI_SWEET_SPOT_HIGH and trend_s >= 18:
        return "MOMENTUM_CONTINUATION"

    if trend_s >= 15 and rsi < config.RSI_SWEET_SPOT_LOW:
        return "PULLBACK"

    if foreign_s >= 12:
        return "FOREIGN_ACCUMULATION"

    if div == "HIDDEN_BULLISH":
        return "PULLBACK"

    return "WATCH"


# ── Main scorer ───────────────────────────────────────────────────────────────

def score_stock(
    symbol:          str,
    snap:            dict,
    df,                            # full indicator DataFrame for OBV + pattern detection
    foreign:         dict | None,
    breadth:         dict | None,
    sector:          str | None = None,
    pipeline_result: dict | None = None,
) -> dict:
    """
    Full 6-layer score for one stock.

    Returns:
    {
        symbol, total_score, verdict,
        setup, divergence,
        layer_scores: {trend, momentum, volume, pattern, foreign, breadth},
        reasons: {trend, momentum, volume, pattern, foreign, breadth},
        bearish_warnings: list[str],
        trade_levels: {entry, target, stop_loss, target_pct, sl_pct, rr} | None,
        snapshot: snap,
        analysis_score, sim_score, pipeline_ok,
    }
    """
    df_tail20 = df.tail(20) if df is not None and len(df) >= 20 else None

    # ── Layer scores ─────────────────────────────────────────────────────────
    t_score,  t_reasons              = _score_trend(snap)
    m_score,  m_reasons              = _score_momentum(snap)
    v_score,  v_reasons              = _score_volume(snap, df_tail20)
    p_score,  p_reasons, bear_pats   = _score_pattern(snap, df)
    f_score,  f_reasons              = _score_foreign(foreign)
    b_score,  b_reasons              = _score_breadth(breadth, sector)

    # ── Divergence modifier ───────────────────────────────────────────────────
    div = snap.get("divergence", "NONE")
    div_modifier = {
        "BULLISH":        config.DIV_BULLISH_BONUS,
        "HIDDEN_BULLISH": config.DIV_HIDDEN_BULL_BONUS,
        "BEARISH":        config.DIV_BEARISH_PENALTY,
        "HIDDEN_BEARISH": -5,
        "NONE":           0,
    }.get(div, 0)

    # ── Bearish pattern penalty ───────────────────────────────────────────────
    bear_pattern_penalty = -5 if bear_pats else 0

    raw_total = t_score + m_score + v_score + p_score + f_score + b_score

    analysis_bonus = 0
    sim_bonus      = 0
    if pipeline_result and pipeline_result.get("pipeline_ok"):
        analysis_bonus = (pipeline_result.get("analysis") or {}).get("analysis_score", 0)
        sim_bonus      = (pipeline_result.get("simulation") or {}).get("sim_score", 0)

    total = max(0, min(100, raw_total + div_modifier + bear_pattern_penalty))
    # Pipeline bonuses are additive and can push total above 100
    total = total + analysis_bonus + sim_bonus

    layer_scores = {
        "trend":    t_score,
        "momentum": m_score,
        "volume":   v_score,
        "pattern":  p_score,
        "foreign":  f_score,
        "breadth":  b_score,
    }
    reasons = {
        "trend":    t_reasons,
        "momentum": m_reasons,
        "volume":   v_reasons,
        "pattern":  p_reasons,
        "foreign":  f_reasons,
        "breadth":  b_reasons,
    }

    setup = classify_setup(snap, layer_scores)

    # ── Verdict ───────────────────────────────────────────────────────────────
    if total >= 80:
        verdict = "STRONG_BUY"
    elif total >= 65:
        verdict = "BUY"
    elif total >= config.MIN_SCORE:
        verdict = "WATCH"
    else:
        verdict = "SKIP"

    trade_levels = calc_trade_levels(snap) if verdict != "SKIP" else None

    bearish_warnings = []
    if bear_pats:
        bearish_warnings += [f"Bearish candle: {p}" for p in bear_pats]
    if div in ("BEARISH", "HIDDEN_BEARISH"):
        bearish_warnings.append(f"RSI divergence: {div}")

    return {
        "symbol":           symbol,
        "total_score":      total,
        "verdict":          verdict,
        "setup":            setup,
        "divergence":       div,
        "div_modifier":     div_modifier,
        "layer_scores":     layer_scores,
        "reasons":          reasons,
        "bearish_warnings": bearish_warnings,
        "trade_levels":     trade_levels,
        "snapshot":         snap,
        "foreign":          foreign,   # raw foreign flow dict for briefing prompt
        "analysis_score":   analysis_bonus,
        "sim_score":        sim_bonus,
        "pipeline_ok":      bool(pipeline_result and pipeline_result.get("pipeline_ok")),
    }
