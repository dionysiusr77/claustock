"""
D-1 scan orchestrator.

Pipeline:
  1. Load universe (LQ45 / IDX80 / COMPOSITE)
  2. Stage 1 — liquidity filter (price range + avg daily value)
  3. Batch download 1y daily OHLCV for survivors
  4. Fetch market breadth (IHSG + sectoral indices) — one call
  5. Compute all indicators per stock
  6. Score without foreign flow → first-pass ranking
  7. Fetch foreign flow for top 2× candidates only (saves ~60 API calls)
  8. Re-score top candidates with actual foreign data
  9. Apply hard disqualifiers (bearish divergence, score < MIN_SCORE, no R:R)
  10. Return final ranked list (top N for AI briefing)
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from invezgo_client import (
    fetch_daily_batch,
    fetch_foreign_flow_market,
    fetch_foreign_flow_stock,
    fetch_market_breadth,
    filter_liquid,
    get_universe,
)
from indicators import compute_all, latest_snapshot
from scorer import score_stock

logger = logging.getLogger(__name__)

# ── Sector map: ticker prefix → sector name matching fetcher.SECTOR_TICKERS ──
_SECTOR_MAP = {
    "BB": "Finance", "BN": "Finance", "BT": "Finance", "BJ": "Finance",
    "BM": "Finance", "BP": "Finance", "MF": "Finance",
    "TL": "Infra",   "EX": "Infra",   "IS": "Infra",   "TB": "Infra",
    "TO": "Infra",   "FR": "Infra",
    "AD": "Mining",  "PT": "Mining",  "IT": "Mining",  "HR": "Mining",
    "IN": "Mining",  "ME": "Mining",  "MD": "Mining",  "AN": "Mining",
    "AS": "Manufacture", "UN": "Consumer", "IC": "Consumer", "MY": "Consumer",
    "SI": "Consumer", "CP": "Consumer", "GG": "Consumer", "HM": "Consumer",
    "AM": "Trade",   "AC": "Trade",   "RA": "Trade",   "LP": "Trade",
    "GO": "Trade",   "BU": "Trade",
    "KL": "Consumer", "TS": "Consumer", "MI": "Consumer", "HE": "Consumer",
    "BS": "Property", "CT": "Property", "PW": "Property", "SM": "Property",
    "KI": "Property", "DM": "Property",
    "PG": "Infra",   "JS": "Infra",   "WI": "Infra",
    "BR": "Basic Ind", "TK": "Basic Ind", "TP": "Basic Ind",
}


def _guess_sector(symbol: str) -> str | None:
    code = symbol.replace(".JK", "").upper()
    return _SECTOR_MAP.get(code[:2])


def _fetch_ff_batch(symbols: list[str], max_workers: int = 4) -> dict[str, dict | None]:
    """Fetch multi-day foreign flow (with consecutive streaks) concurrently."""
    results: dict[str, dict | None] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(fetch_foreign_flow_stock, sym): sym for sym in symbols}
        for future in as_completed(future_map):
            sym = future_map[future]
            try:
                results[sym] = future.result()
            except Exception:
                results[sym] = None
    return results


# ── Hard disqualifiers ────────────────────────────────────────────────────────

def _is_disqualified(result: dict) -> tuple[bool, str]:
    """Return (True, reason) if the stock should be excluded from briefing."""
    snap = result.get("snapshot", {})
    div  = snap.get("divergence", "NONE")

    if div == "BEARISH":
        return True, "Bearish RSI divergence"
    if result.get("total_score", 0) < config.MIN_SCORE:
        return True, f"Score {result['total_score']} < MIN_SCORE {config.MIN_SCORE}"
    if result.get("trade_levels") is None:
        return True, "No viable R:R after fees"

    return False, ""


# ── Main scan ─────────────────────────────────────────────────────────────────

def run_scan(universe_tier: str = config.UNIVERSE) -> dict:
    """
    Full D-1 scan. Returns:
    {
        candidates: list[dict],     # scored + ranked, ready for AI briefing
        market:     dict,           # IHSG + sector breadth
        foreign_market: dict|None,  # aggregate market foreign flow
        stats: {
            universe_size, liquid_size, scored_size, candidate_size
        }
    }
    """
    logger.info("=== D-1 scan starting (universe: %s) ===", universe_tier)

    # ── Step 1: Universe ──────────────────────────────────────────────────────
    all_symbols = get_universe(universe_tier)
    logger.info("Universe: %d symbols", len(all_symbols))

    # ── Step 2: Liquidity filter ──────────────────────────────────────────────
    liquid_symbols = filter_liquid(all_symbols)
    if not liquid_symbols:
        logger.error("Liquidity filter returned 0 symbols — aborting scan")
        return {"candidates": [], "market": {}, "foreign_market": None,
                "stats": {"universe_size": len(all_symbols), "liquid_size": 0,
                          "scored_size": 0, "candidate_size": 0}}

    # ── Step 3: Batch OHLCV (1 year daily) ───────────────────────────────────
    logger.info("Downloading 1y OHLCV for %d liquid stocks...", len(liquid_symbols))
    ohlcv_map = fetch_daily_batch(liquid_symbols, days=365)

    # ── Step 4: Market breadth ────────────────────────────────────────────────
    breadth       = fetch_market_breadth()
    foreign_mkt   = fetch_foreign_flow_market()

    # ── Step 5 + 6: Indicators + first-pass scoring (no foreign flow) ─────────
    first_pass: list[dict] = []
    for sym, df in ohlcv_map.items():
        df_ind = compute_all(df)
        if df_ind.empty:
            continue
        snap   = latest_snapshot(df_ind)
        sector = _guess_sector(sym)
        result = score_stock(sym, snap, df_ind, foreign=None, breadth=breadth, sector=sector)
        result["_df"] = df_ind          # stash for re-score; removed before returning
        first_pass.append(result)

    first_pass.sort(key=lambda x: x["total_score"], reverse=True)
    logger.info("First-pass scored: %d stocks", len(first_pass))

    # ── Step 7: Foreign flow for top 2× candidates ───────────────────────────
    top_n  = config.TOP_N_AI * 2
    top_syms = [r["symbol"] for r in first_pass[:top_n]]
    logger.info("Fetching foreign flow for top %d stocks...", len(top_syms))
    ff_map = _fetch_ff_batch(top_syms)

    # ── Step 8: Re-score top candidates with real foreign data ────────────────
    rescored: list[dict] = []
    for result in first_pass[:top_n]:
        sym    = result["symbol"]
        df_ind = result.pop("_df")
        snap   = result["snapshot"]
        sector = _guess_sector(sym)
        fresh  = score_stock(sym, snap, df_ind, foreign=ff_map.get(sym),
                             breadth=breadth, sector=sector)
        rescored.append(fresh)

    # Keep the rest of first_pass as-is (below top_n, foreign = None)
    rest = first_pass[top_n:]
    for r in rest:
        r.pop("_df", None)

    all_scored = rescored + rest
    all_scored.sort(key=lambda x: x["total_score"], reverse=True)

    # ── Step 9: Hard disqualifiers ────────────────────────────────────────────
    candidates: list[dict] = []
    for result in all_scored:
        disq, reason = _is_disqualified(result)
        if disq:
            logger.debug("SKIP %s — %s", result["symbol"], reason)
            continue
        candidates.append(result)

    # ── Step 10: Return top N for AI ──────────────────────────────────────────
    candidates = candidates[:config.TOP_N_AI]

    logger.info(
        "=== Scan complete: %d candidates (from %d liquid / %d universe) ===",
        len(candidates), len(liquid_symbols), len(all_symbols),
    )

    return {
        "candidates":     candidates,
        "market":         breadth,
        "foreign_market": foreign_mkt,
        "stats": {
            "universe_size":  len(all_symbols),
            "liquid_size":    len(liquid_symbols),
            "scored_size":    len(all_scored),
            "candidate_size": len(candidates),
        },
    }


# ── Quick single-stock rescan ─────────────────────────────────────────────────

def scan_single(symbol: str) -> dict | None:
    """
    Score a single stock on demand (used by /pick Telegram command).
    Returns the score_stock result dict or None on failure.
    """
    from invezgo_client import fetch_daily, fetch_foreign_flow_stock, fetch_market_breadth

    df = fetch_daily(symbol, days=365)
    if df is None or df.empty:
        logger.warning("scan_single: no data for %s", symbol)
        return None

    df_ind = compute_all(df)
    if df_ind.empty:
        return None

    snap    = latest_snapshot(df_ind)
    breadth = fetch_market_breadth()
    ff      = fetch_foreign_flow_stock(symbol)
    sector  = _guess_sector(symbol)

    return score_stock(symbol, snap, df_ind, foreign=ff, breadth=breadth, sector=sector)
