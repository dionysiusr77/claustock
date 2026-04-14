"""
Phase 5 — Claude AI verdict.
Reads all 4 scoring layers and returns a structured trade recommendation.
Only called when score >= IDX_MIN_SCORE to keep API costs low.
"""

import json
import logging
import re

import anthropic

import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a professional IDX (Indonesia Stock Exchange) trader and analyst.
Analyze the provided stock data across 4 layers and give a precise trade recommendation.

Fee structure: Buy 0.15% + Sell 0.25% = 0.40% round trip.
Minimum viable target: +1.5%. Recommended R/R: minimum 1:2 after fees.
T+2 settlement: capital is locked for 2 days after selling.
Minimum trade unit: 1 lot = 100 shares.
IDX lot sizing: always round DOWN to nearest whole lot.

Respond with ONLY a valid JSON object, no markdown, no explanation:
{
    "action": "ENTER" | "WAIT" | "SKIP",
    "confidence": int (0-100),
    "entry_price": float,
    "target_price": float,
    "target_pct": float,
    "stop_loss": float,
    "stop_loss_pct": float,
    "risk_reward": float,
    "hold_duration": "intraday" | "1-2 days" | "3-5 days",
    "lots": int,
    "capital_idr": int,
    "risk_level": "LOW" | "MEDIUM" | "HIGH",
    "reasoning": "2-3 sentences referencing specific indicators and score layers"
}"""


def get_ai_verdict(symbol: str, snapshot: dict) -> dict | None:
    """
    Call Claude Haiku with the full snapshot and return a structured verdict.
    Returns None if API is unavailable or call fails.
    """
    if not config.ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — AI verdict skipped")
        return None

    scores   = snapshot.get("scores", {})
    price    = snapshot.get("price", 0)
    capital  = config.IDX_CAPITAL
    max_lots = config.IDX_MAX_LOTS

    user_msg = f"""Stock: {symbol}
Current price: {price:,.0f} IDR
Capital available: {capital:,.0f} IDR
Max lots allowed: {max_lots}

SCORING (total {snapshot.get('total_score', 0)}/100):
  Technical ({scores.get('technical', 0)}/35):
    RSI: {snapshot.get('rsi', 'N/A')}
    MA trend: {snapshot.get('ma_trend', 'N/A')}
    Volume ratio: {snapshot.get('volume_ratio', 'N/A')}x
    Candle: {snapshot.get('candle_pattern', 'N/A')}

  Forecast ({scores.get('prophet', 0)}/25):
    5-day forecast: {snapshot.get('forecast_5d', 'N/A')} IDR
    Trend: {snapshot.get('trend_pct', 'N/A')}% → {snapshot.get('trend', 'N/A')}

  Foreign Flow ({scores.get('foreign', 0)}/20):
    Net foreign: {snapshot.get('net_foreign_buy_idr', 'N/A')} IDR
    Consecutive days: {snapshot.get('days_consecutive', 'N/A')}

  News Sentiment ({scores.get('news', 0)}/20):
    Sentiment: {snapshot.get('news_sentiment', 'N/A')}
    Key headline: {snapshot.get('news_headline', 'None')}

Verdict so far: {snapshot.get('verdict', 'N/A')}
"""

    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(raw)

        # Sanitise
        result["lots"]        = max(1, int(result.get("lots", 1)))
        result["confidence"]  = max(0, min(100, int(result.get("confidence", 50))))
        result["capital_idr"] = result["lots"] * 100 * price

        logger.info(
            f"AI verdict for {symbol}: {result['action']} "
            f"(confidence={result['confidence']}%, lots={result['lots']})"
        )
        return result

    except Exception as e:
        logger.error(f"get_ai_verdict({symbol}): {e}")
        return None
