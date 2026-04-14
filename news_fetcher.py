"""
Phase 3 — News Sentiment
Sources:
  1. Google News RSS (Indonesian language)
  2. IDX keterbukaan informasi (official corporate announcements)

Claude Haiku scores headlines 0–20 pts.
Results are cached in memory per 30 min to avoid hammering Claude API.
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import feedparser
import requests

import config
import firestore_client as db

logger = logging.getLogger(__name__)

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id"
IDX_ANNOUNCEMENTS = "https://www.idx.co.id/umbraco/Surface/ListedCompany/GetAnnouncement"

STOCK_NAMES = {
    "BBCA.JK": "Bank Central Asia BCA",
    "BBRI.JK": "Bank Rakyat Indonesia BRI",
    "TLKM.JK": "Telkom Indonesia TLKM",
    "ASII.JK": "Astra International ASII",
    "GOTO.JK": "GoTo Gojek Tokopedia GOTO",
    "BREN.JK": "Barito Renewables BREN",
    "BMRI.JK": "Bank Mandiri BMRI",
    "UNVR.JK": "Unilever Indonesia UNVR",
}

# In-memory cache: { symbol: { data, cached_at } }
_news_cache: dict = {}
NEWS_CACHE_TTL_MIN = 30


# ── Fetchers ──────────────────────────────────────────────────────────────────

def fetch_news(symbol: str, max_age_hours: int = 24) -> list[dict]:
    """
    Fetch recent news from Google News RSS.
    Returns list of: { headline, source, age_hours, url }
    """
    stock_name = STOCK_NAMES.get(symbol, symbol.replace(".JK", ""))
    ticker     = symbol.replace(".JK", "")
    query      = f"{ticker} saham BEI"

    url = GOOGLE_NEWS_RSS.format(query=requests.utils.quote(query))
    items = []

    try:
        feed = feedparser.parse(url)
        now  = datetime.now(timezone.utc)

        for entry in feed.entries[:8]:
            try:
                pub = parsedate_to_datetime(entry.published)
                age_hours = (now - pub).total_seconds() / 3600
            except Exception:
                age_hours = 0

            if age_hours > max_age_hours:
                continue

            items.append({
                "headline":  entry.title,
                "source":    entry.source.title if hasattr(entry, "source") else "",
                "age_hours": round(age_hours, 1),
                "url":       entry.link,
            })
    except Exception as e:
        logger.warning(f"fetch_news({symbol}): Google RSS failed — {e}")

    return items


def fetch_idx_announcements(symbol: str) -> list[dict]:
    """
    Fetch official IDX corporate announcements (keterbukaan informasi).
    Returns list of: { title, date, type, url }
    """
    ticker = symbol.replace(".JK", "")
    items  = []

    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer":    "https://www.idx.co.id/",
        }
        params = {
            "kodeSaham": ticker,
            "indexFrom":  0,
            "pageSize":   5,
        }
        resp = requests.get(IDX_ANNOUNCEMENTS, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        announcements = data if isinstance(data, list) else data.get("announcements", [])
        for ann in announcements[:5]:
            items.append({
                "title": ann.get("judul") or ann.get("title", ""),
                "date":  ann.get("tanggal") or ann.get("date", ""),
                "type":  ann.get("jenis") or ann.get("type", ""),
                "url":   ann.get("url", ""),
            })
    except Exception as e:
        logger.warning(f"fetch_idx_announcements({symbol}): {e}")

    return items


# ── Claude sentiment scoring ──────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an IDX (Indonesia Stock Exchange) news analyst.
Score the sentiment of the provided news headlines for a specific stock.

Scoring rules (return a single integer 0–20):
+20: earnings beat, dividend announcement, government contract, credit rating upgrade
+12: analyst upgrade, sector tailwind, acquisition at premium
+5:  neutral routine disclosure, no material news
0:   no recent news provided
-8:  earnings miss, analyst downgrade, sector headwind
-15: rights issue (penawaran umum terbatas), OJK investigation, insider selling, scandal

If multiple headlines exist, weigh the most impactful one.
The final score must be clamped to [0, 20] — never return negative values.

Respond with ONLY a valid JSON object:
{
    "score": int (0-20),
    "sentiment": "POSITIVE" | "NEUTRAL" | "NEGATIVE",
    "key_headline": "the single most impactful headline",
    "reasoning": "one sentence explanation"
}"""


def _call_claude(symbol: str, news_items: list[dict], announcements: list[dict]) -> dict | None:
    if not config.ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — news scoring skipped")
        return None

    headlines = []
    for item in news_items[:5]:
        headlines.append(f"[NEWS] {item['headline']} ({item['age_hours']:.0f}h ago)")
    for ann in announcements[:3]:
        headlines.append(f"[IDX]  {ann['title']} — {ann['type']}")

    if not headlines:
        return {"score": 0, "sentiment": "NEUTRAL", "key_headline": "", "reasoning": "No recent news"}

    user_msg = f"Stock: {symbol}\n\nHeadlines:\n" + "\n".join(headlines)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception as e:
        logger.error(f"_call_claude({symbol}): {e}")
        return None


# ── Main entry point ──────────────────────────────────────────────────────────

def score_news_sentiment(symbol: str) -> dict:
    """
    Fetch news + announcements, score with Claude, cache for 30 min.

    Returns:
    {
        score:         int (0–20),
        sentiment:     "POSITIVE" | "NEUTRAL" | "NEGATIVE",
        key_headline:  str,
        reasoning:     str,
        news_items:    list,
        reasons:       list[str],   # human-readable for snapshot
    }
    """
    # Check in-memory cache
    cached = _news_cache.get(symbol)
    if cached:
        age_min = (datetime.now(timezone.utc) - cached["cached_at"]).total_seconds() / 60
        if age_min < NEWS_CACHE_TTL_MIN:
            logger.debug(f"News cache hit for {symbol} ({age_min:.0f}min old)")
            return cached["data"]

    # Fetch
    news_items    = fetch_news(symbol)
    announcements = fetch_idx_announcements(symbol)

    # Score with Claude
    result = _call_claude(symbol, news_items, announcements)

    if result is None:
        result = {
            "score":        0,
            "sentiment":    "NEUTRAL",
            "key_headline": "",
            "reasoning":    "Claude API unavailable",
        }

    score     = max(0, min(20, int(result.get("score", 0))))
    sentiment = result.get("sentiment", "NEUTRAL")
    headline  = result.get("key_headline", "")
    reasoning = result.get("reasoning", "")

    reasons = []
    if headline:
        reasons.append(f"News: {headline[:80]}")
    reasons.append(f"Sentiment: {sentiment} ({score}/20 pts) — {reasoning}")

    output = {
        "score":        score,
        "sentiment":    sentiment,
        "key_headline": headline,
        "reasoning":    reasoning,
        "news_items":   news_items[:3],
        "reasons":      reasons,
    }

    # Cache in memory
    _news_cache[symbol] = {
        "data":      output,
        "cached_at": datetime.now(timezone.utc),
    }

    # Persist to Firestore
    db.save_news(symbol, {
        "score":        score,
        "sentiment":    sentiment,
        "key_headline": headline,
        "news_items":   news_items[:3],
        "announcements": announcements[:3],
    })

    return output
