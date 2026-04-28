"""
IDX universe loader.
Returns ticker lists in yfinance format (e.g. "BBCA.JK").
Three tiers:
  LQ45      — 45 most liquid IDX stocks
  IDX80     — 80 most liquid (superset of LQ45)
  COMPOSITE — attempts full IDX listing via IDX API; falls back to IDX80
"""

import logging
import requests

logger = logging.getLogger(__name__)

# ── LQ45 ──────────────────────────────────────────────────────────────────────
_LQ45 = [
    "AALI", "ADRO", "AKRA", "AMRT", "ANTM", "ASII", "AVIA",
    "BBCA", "BBNI", "BBRI", "BBTN", "BMRI", "BRPT", "BUKA",
    "CPIN", "EMTK", "ESSA", "EXCL",
    "GGRM",
    "HRUM",
    "ICBP", "INCO", "INDF", "INTP", "ITMG",
    "JSMR",
    "KLBF",
    "MAPI", "MBMA", "MDKA", "MEDC", "MIKA", "MNCN",
    "PGAS", "PTBA", "PTPP",
    "SIDO", "SMGR",
    "TBIG", "TKIM", "TLKM", "TOWR",
    "UNTR", "UNVR",
    "WIKA",
]

# ── IDX80 (LQ45 + additional liquid stocks) ───────────────────────────────────
_IDX80_EXTRA = [
    "ACES", "ADMR", "ADHI", "AUTO",
    "BFIN", "BJBR", "BJTM", "BKSL", "BMRI", "BNGA", "BREN",
    "CASS", "CTRA",
    "DMAS", "DNET", "DOID", "DSNG",
    "ERAA",
    "FREN",
    "GOTO",
    "HEAL", "HMSP", "HRTA",
    "INDY", "INKP", "ISAT",
    "JPFA",
    "KAEF", "KIJA",
    "LPPF",
    "MFIN", "MLPL", "MYOR",
    "NCKL", "NISP",
    "PGEO", "PNBN", "PWON",
    "RAJA", "RALS",
    "SCMA", "SMRA", "SRTG", "SSIA",
    "TBLA", "TELE", "TPIA", "TSPC",
    "WIIM", "WOLF", "WSKT", "WTON",
]

_IDX80 = sorted(set(_LQ45 + _IDX80_EXTRA))


def _to_jk(tickers: list[str]) -> list[str]:
    return [f"{t}.JK" for t in tickers]


def get_lq45() -> list[str]:
    return _to_jk(_LQ45)


def get_idx80() -> list[str]:
    return _to_jk(_IDX80)


def fetch_idx_composite() -> list[str]:
    """
    Fetch all IDX-listed stocks from the IDX public API.
    Falls back to IDX80 if the request fails.
    Returns list of tickers in "XXXX.JK" format.
    """
    url = "https://www.idx.co.id/umbraco/Surface/Helper/GetSecuritiesData"
    params = {
        "start": 0,
        "length": 9999,
        "searchValue": "",
        "indexCode": "COMPOSITE",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.idx.co.id/",
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        tickers = [
            f"{item['StockCode'].strip()}.JK"
            for item in data.get("data", [])
            if item.get("StockCode", "").strip()
        ]
        if not tickers:
            raise ValueError("empty response from IDX API")
        logger.info("IDX composite: %d stocks loaded", len(tickers))
        return tickers
    except Exception as e:
        logger.warning("IDX composite fetch failed (%s), falling back to IDX80", e)
        return get_idx80()


def get_universe(tier: str = "IDX80") -> list[str]:
    """
    tier: "LQ45" | "IDX80" | "COMPOSITE"
    Returns sorted list of tickers in "XXXX.JK" format.
    """
    tier = tier.upper()
    if tier == "LQ45":
        return get_lq45()
    if tier == "COMPOSITE":
        return fetch_idx_composite()
    return get_idx80()  # default
