"""
Combines all 4 scoring layers into a single verdict.
Called by main.py after all layers have been fetched.
"""
from __future__ import annotations

import config


def score_stock(
    symbol: str,
    technical: dict,
    forecast: dict | None,
    foreign: dict | None,
    news: dict | None,
) -> dict:
    """
    Returns full scoring result:
    {
        symbol:      str,
        total_score: int (0–100),
        verdict:     "STRONG_BUY" | "BUY" | "WATCH" | "SKIP",
        scores: {
            technical: int (0–35),
            prophet:   int (0–25),
            foreign:   int (0–20),
            news:      int (0–20),
        },
        reasons: list[str],
    }
    """
    tech_score    = technical.get("score", 0)
    prophet_score = forecast.get("prophet_score", 0) if forecast else 0
    foreign_score = foreign.get("foreign_score", 0)  if foreign  else 0
    news_score    = news.get("score", 0)              if news     else 0

    total = tech_score + prophet_score + foreign_score + news_score

    verdict = _verdict(total)

    reasons = (
        technical.get("reasons", [])
        + (forecast.get("reasons", []) if forecast else ["Forecast unavailable"])
        + (foreign.get("reasons",  []) if foreign  else ["Foreign flow unavailable"])
        + (news.get("reasons",     []) if news     else ["News unavailable"])
    )

    return {
        "symbol":      symbol,
        "total_score": total,
        "verdict":     verdict,
        "scores": {
            "technical": tech_score,
            "prophet":   prophet_score,
            "foreign":   foreign_score,
            "news":      news_score,
        },
        "reasons": reasons,
        # Convenience fields for Telegram formatting
        "price":               technical.get("price"),
        "rsi":                 technical.get("rsi"),
        "ma_trend":            technical.get("ma_trend"),
        "volume_ratio":        technical.get("volume_ratio"),
        "candle_pattern":      technical.get("candle_pattern"),
        "forecast_5d":         forecast.get("forecast_5d")        if forecast else None,
        "trend_pct":           forecast.get("trend_pct")          if forecast else None,
        "trend":               forecast.get("trend")              if forecast else None,
        "net_foreign_buy_idr": foreign.get("net_foreign_buy_idr") if foreign  else None,
        "days_consecutive":    foreign.get("days_consecutive")    if foreign  else None,
        "news_sentiment":      news.get("sentiment")              if news     else None,
        "news_headline":       news.get("key_headline")           if news     else None,
    }


def _verdict(score: int) -> str:
    if score >= 80:
        return "STRONG_BUY"
    elif score >= 60:
        return "BUY"
    elif score >= 40:
        return "WATCH"
    else:
        return "SKIP"


def should_signal(result: dict) -> bool:
    """Returns True if score clears the minimum threshold for a Telegram signal."""
    return result["total_score"] >= config.IDX_MIN_SCORE
