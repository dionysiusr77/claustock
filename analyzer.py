"""
/analyze BBCA — deep fundamental + technical + sentiment analysis.
Feeds structured data into Claude and returns a full Indonesian-language report.
Covers everything in the user's analysis prompt template.
"""

import logging
import re
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

import config
from news_fetcher import fetch_news, fetch_idx_announcements

logger = logging.getLogger(__name__)

# ── Sector competitors map ────────────────────────────────────────────────────
COMPETITORS = {
    "BBCA.JK": ["BBRI.JK", "BMRI.JK", "BNGA.JK"],
    "BBRI.JK": ["BBCA.JK", "BMRI.JK", "BNGA.JK"],
    "BMRI.JK": ["BBCA.JK", "BBRI.JK", "BNGA.JK"],
    "TLKM.JK": ["EXCL.JK", "ISAT.JK", "TOWR.JK"],
    "ASII.JK": ["IMAS.JK", "INDS.JK", "AUTO.JK"],
    "GOTO.JK": ["BUKA.JK", "EMTK.JK", "MNCN.JK"],
    "BREN.JK": ["ADRO.JK", "MEDC.JK", "PGAS.JK"],
    "UNVR.JK": ["ICBP.JK", "INDF.JK", "SIDO.JK"],
}
DEFAULT_COMPETITORS = ["BBCA.JK", "BBRI.JK", "TLKM.JK"]

STOCK_NAMES = {
    "BBCA.JK": "Bank Central Asia",
    "BBRI.JK": "Bank Rakyat Indonesia",
    "BMRI.JK": "Bank Mandiri",
    "TLKM.JK": "Telkom Indonesia",
    "ASII.JK": "Astra International",
    "GOTO.JK": "GoTo Gojek Tokopedia",
    "BREN.JK": "Barito Renewables Energy",
    "UNVR.JK": "Unilever Indonesia",
}


# ── Data fetchers ─────────────────────────────────────────────────────────────

def _fetch_fundamentals(symbol: str) -> dict:
    """Fetch fundamental data from yfinance ticker.info."""
    try:
        ticker = yf.Ticker(symbol)
        info   = ticker.info or {}

        def _get(key, default=None):
            val = info.get(key, default)
            return val if val not in (None, "N/A", "", float("inf")) else default

        return {
            "name":              _get("longName") or STOCK_NAMES.get(symbol, symbol),
            "sector":            _get("sector", "—"),
            "industry":          _get("industry", "—"),
            "market_cap":        _get("marketCap"),
            "shares_out":        _get("sharesOutstanding"),
            "avg_volume":        _get("averageDailyVolume10Day") or _get("averageVolume"),
            # Price
            "price":             _get("currentPrice") or _get("regularMarketPrice"),
            "52w_high":          _get("fiftyTwoWeekHigh"),
            "52w_low":           _get("fiftyTwoWeekLow"),
            # Valuation
            "pe_trailing":       _get("trailingPE"),
            "pe_forward":        _get("forwardPE"),
            "pb":                _get("priceToBook"),
            "ev_ebitda":         _get("enterpriseToEbitda"),
            "dividend_yield":    _get("dividendYield"),
            "dividend_rate":     _get("dividendRate"),
            # Profitability
            "roe":               _get("returnOnEquity"),
            "roa":               _get("returnOnAssets"),
            "gross_margin":      _get("grossMargins"),
            "operating_margin":  _get("operatingMargins"),
            "net_margin":        _get("profitMargins"),
            # Growth
            "revenue_growth":    _get("revenueGrowth"),
            "earnings_growth":   _get("earningsGrowth"),
            "earnings_qoq":      _get("earningsQuarterlyGrowth"),
            # Balance sheet
            "debt_to_equity":    _get("debtToEquity"),
            "current_ratio":     _get("currentRatio"),
            "total_cash":        _get("totalCash"),
            "total_debt":        _get("totalDebt"),
            "free_cashflow":     _get("freeCashflow"),
            "operating_cashflow": _get("operatingCashflow"),
            # Revenue
            "total_revenue":     _get("totalRevenue"),
            "ebitda":            _get("ebitda"),
            # Analyst
            "recommendation":    _get("recommendationKey", "—"),
            "target_mean":       _get("targetMeanPrice"),
            "target_low":        _get("targetLowPrice"),
            "target_high":       _get("targetHighPrice"),
            "analyst_count":     _get("numberOfAnalystOpinions"),
        }
    except Exception as e:
        logger.error(f"_fetch_fundamentals({symbol}): {e}")
        return {}


def _fetch_competitor_snapshot(symbol: str) -> dict:
    """Fetch key valuation ratios for a competitor."""
    try:
        info = yf.Ticker(symbol).info or {}
        return {
            "symbol":     symbol.replace(".JK", ""),
            "pe":         info.get("trailingPE"),
            "pb":         info.get("priceToBook"),
            "roe":        info.get("returnOnEquity"),
            "net_margin": info.get("profitMargins"),
            "rec":        info.get("recommendationKey", "—"),
        }
    except Exception:
        return {"symbol": symbol.replace(".JK", "")}


def _fetch_multiframe(symbol: str) -> dict:
    """
    Fetch daily (1y), weekly (2y), monthly (3y) OHLCV.
    Returns { daily: df, weekly: df, monthly: df }
    """
    try:
        ticker = yf.Ticker(symbol)
        daily  = ticker.history(period="1y",  interval="1d")
        weekly = ticker.history(period="2y",  interval="1wk")
        monthly = ticker.history(period="3y", interval="1mo")
        for df in [daily, weekly, monthly]:
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
        return {"daily": daily, "weekly": weekly, "monthly": monthly}
    except Exception as e:
        logger.error(f"_fetch_multiframe({symbol}): {e}")
        return {}


def _calc_indicators(df: pd.DataFrame, label: str) -> dict:
    """Calculate MA, RSI, MACD, Bollinger Bands, support/resistance for one timeframe."""
    if df is None or len(df) < 20:
        return {"timeframe": label, "error": "insufficient data"}

    close = df["close"]
    vol   = df["volume"] if "volume" in df.columns else pd.Series(dtype=float)

    result = {"timeframe": label}

    # MAs
    for w in [20, 50, 200]:
        if len(close) >= w:
            result[f"ma{w}"] = round(float(close.rolling(w).mean().iloc[-1]), 2)

    price = float(close.iloc[-1])
    result["price"] = round(price, 2)

    # Trend
    ma20 = result.get("ma20")
    ma50 = result.get("ma50")
    if ma20 and ma50:
        if price > ma20 > ma50:
            result["trend"] = "uptrend"
        elif price < ma20 < ma50:
            result["trend"] = "downtrend"
        else:
            result["trend"] = "sideways"

    # RSI(14)
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    rsi    = 100 - (100 / (1 + rs))
    result["rsi"] = round(float(rsi.iloc[-1]), 1)

    # MACD(12,26,9)
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    result["macd"]        = round(float(macd.iloc[-1]), 4)
    result["macd_signal"] = round(float(signal.iloc[-1]), 4)
    result["macd_hist"]   = round(float((macd - signal).iloc[-1]), 4)
    result["macd_cross"]  = (
        "bullish" if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]
        else "bearish" if macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]
        else "neutral"
    )

    # Bollinger Bands(20, 2σ)
    if len(close) >= 20:
        bb_mid  = close.rolling(20).mean()
        bb_std  = close.rolling(20).std()
        bb_up   = bb_mid + 2 * bb_std
        bb_low  = bb_mid - 2 * bb_std
        result["bb_upper"] = round(float(bb_up.iloc[-1]), 2)
        result["bb_mid"]   = round(float(bb_mid.iloc[-1]), 2)
        result["bb_lower"] = round(float(bb_low.iloc[-1]), 2)
        bb_width = float(bb_up.iloc[-1] - bb_low.iloc[-1])
        if bb_width > 0:
            result["bb_position"] = round((price - float(bb_low.iloc[-1])) / bb_width * 100, 1)

    # Support & Resistance (recent swing highs/lows over last 20 bars)
    highs  = df["high"].rolling(5, center=True).max()
    lows   = df["low"].rolling(5, center=True).min()
    recent = df.tail(20)
    result["resistance"] = round(float(recent["high"].max()), 2)
    result["support"]    = round(float(recent["low"].min()), 2)

    # Volume trend
    if len(vol) >= 20:
        avg_vol = float(vol.rolling(20).mean().iloc[-1])
        last_vol = float(vol.iloc[-1])
        result["vol_ratio"] = round(last_vol / avg_vol, 2) if avg_vol else 1.0

    return result


# ── Claude prompt builder ─────────────────────────────────────────────────────

def _build_prompt(symbol: str, fund: dict, tf: dict, competitors: list[dict],
                  news: list[dict], announcements: list[dict]) -> str:
    ticker = symbol.replace(".JK", "")
    name   = fund.get("name", ticker)
    price  = fund.get("price", "N/A")
    today  = datetime.now().strftime("%d %B %Y")

    def _fmt_pct(v):
        return f"{v*100:.1f}%" if v is not None else "N/A"

    def _fmt_num(v, unit=""):
        if v is None:
            return "N/A"
        if abs(v) >= 1e12:
            return f"{v/1e12:.2f}T{unit}"
        if abs(v) >= 1e9:
            return f"{v/1e9:.2f}B{unit}"
        if abs(v) >= 1e6:
            return f"{v/1e6:.2f}M{unit}"
        return f"{v:,.0f}{unit}"

    # Technical sections
    tf_sections = []
    for key in ["daily", "weekly", "monthly"]:
        d = tf.get(key, {})
        if not d or d.get("error"):
            continue
        label = {"daily": "Harian", "weekly": "Mingguan", "monthly": "Bulanan"}[key]
        ma_str = " | ".join(
            f"MA{w}={d[f'ma{w}']:,.0f}" for w in [20, 50, 200] if f"ma{w}" in d
        )
        tf_sections.append(f"""
  [{label}]
  Trend: {d.get('trend', 'N/A').upper()} | Price: {d.get('price', 'N/A'):,.0f}
  {ma_str}
  RSI(14): {d.get('rsi', 'N/A')} | MACD: {d.get('macd', 'N/A')} (signal: {d.get('macd_signal', 'N/A')}) → {d.get('macd_cross', 'N/A').upper()}
  Bollinger: {d.get('bb_lower', 'N/A'):,.0f} / {d.get('bb_mid', 'N/A'):,.0f} / {d.get('bb_upper', 'N/A'):,.0f} | Position: {d.get('bb_position', 'N/A')}%
  Support: {d.get('support', 'N/A'):,.0f} | Resistance: {d.get('resistance', 'N/A'):,.0f}
  Volume ratio: {d.get('vol_ratio', 'N/A')}x""")

    # Competitor section
    comp_rows = []
    for c in competitors:
        comp_rows.append(
            f"  {c['symbol']:6s} | P/E: {c.get('pe') or 'N/A'} | P/BV: {c.get('pb') or 'N/A'}"
            f" | ROE: {_fmt_pct(c.get('roe'))} | Net margin: {_fmt_pct(c.get('net_margin'))}"
            f" | Analyst: {c.get('rec', '—')}"
        )

    # News section
    news_lines = [f"  - {n['headline']} ({n['source']}, {n['age_hours']:.0f}h ago)" for n in news[:5]]
    ann_lines  = [f"  - [IDX] {a['title']} ({a['date']})" for a in announcements[:3]]

    prompt = f"""Tanggal analisis: {today}
Data untuk: {name} ({ticker}.JK) — Bursa Efek Indonesia

===== DATA FUNDAMENTAL =====
Market Cap: {_fmt_num(fund.get('market_cap'), ' IDR')}
Harga saat ini: {price:,.0f} IDR
52w High/Low: {fund.get('52w_high', 'N/A')} / {fund.get('52w_low', 'N/A')}
Sektor: {fund.get('sector', 'N/A')} | Industri: {fund.get('industry', 'N/A')}

VALUASI:
  P/E (trailing): {fund.get('pe_trailing') or 'N/A'}
  P/E (forward):  {fund.get('pe_forward') or 'N/A'}
  P/BV:           {fund.get('pb') or 'N/A'}
  EV/EBITDA:      {fund.get('ev_ebitda') or 'N/A'}
  Dividend yield: {_fmt_pct(fund.get('dividend_yield'))} ({_fmt_num(fund.get('dividend_rate'), ' IDR/share')})

PROFITABILITAS:
  Gross margin:     {_fmt_pct(fund.get('gross_margin'))}
  Operating margin: {_fmt_pct(fund.get('operating_margin'))}
  Net margin:       {_fmt_pct(fund.get('net_margin'))}
  ROE: {_fmt_pct(fund.get('roe'))} | ROA: {_fmt_pct(fund.get('roa'))}

PERTUMBUHAN (YoY):
  Revenue growth:  {_fmt_pct(fund.get('revenue_growth'))}
  Earnings growth: {_fmt_pct(fund.get('earnings_growth'))}
  Earnings QoQ:    {_fmt_pct(fund.get('earnings_qoq'))}
  Total revenue:   {_fmt_num(fund.get('total_revenue'), ' IDR')}
  EBITDA:          {_fmt_num(fund.get('ebitda'), ' IDR')}

NERACA & ARUS KAS:
  Debt/Equity:       {fund.get('debt_to_equity') or 'N/A'}
  Current ratio:     {fund.get('current_ratio') or 'N/A'}
  Total cash:        {_fmt_num(fund.get('total_cash'), ' IDR')}
  Total debt:        {_fmt_num(fund.get('total_debt'), ' IDR')}
  Free cash flow:    {_fmt_num(fund.get('free_cashflow'), ' IDR')}
  Operating CF:      {_fmt_num(fund.get('operating_cashflow'), ' IDR')}

ANALIS:
  Konsensus: {fund.get('recommendation', 'N/A').upper()}
  Target harga: {fund.get('target_low', 'N/A')} – {fund.get('target_mean', 'N/A')} – {fund.get('target_high', 'N/A')} IDR
  Jumlah analis: {fund.get('analyst_count', 'N/A')}

===== KOMPETITOR =====
(Perbandingan dengan saham sejenis di sektor yang sama)
{chr(10).join(comp_rows) if comp_rows else '  Data tidak tersedia'}

===== DATA TEKNIKAL =====
{''.join(tf_sections) if tf_sections else '  Data tidak tersedia'}

===== BERITA & SENTIMEN =====
Berita terkini:
{chr(10).join(news_lines) if news_lines else '  Tidak ada berita'}
{chr(10).join(ann_lines) if ann_lines else ''}

===== INSTRUKSI =====
Berdasarkan semua data di atas, buat laporan analisis saham lengkap dalam Bahasa Indonesia dengan format berikut:

<b>📊 ANALISIS {ticker}.JK — {today}</b>
<b>{name}</b>

<b>📌 Bagian 1: Profil Emiten</b>
[Sektor, model bisnis, posisi kompetitif, kapitalisasi pasar & likuiditas]

<b>💰 Bagian 2: Analisis Fundamental</b>
<b>2.1 Kinerja Keuangan</b>
[Revenue growth YoY & QoQ, margin, ROE, ROA, DER, arus kas]
<i>Review: ...</i>

<b>2.2 Valuasi</b>
[P/E, P/BV, EV/EBITDA, dividend yield — bandingkan dengan kompetitor]
<i>Review: overvalued / fair / undervalued</i>

<b>2.3 Prospek Bisnis</b>
[Outlook industri, strategi, risiko makro]
<i>Review: ...</i>

<b>📈 Bagian 3: Analisis Teknikal</b>
[Harian, Mingguan, Bulanan — trend, MA, RSI, MACD, BB, support/resistance, pola]
<i>Review: ...</i>

<b>🌊 Bagian 4: Sentimen & Katalis</b>
[Sentimen asing, berita material, aksi korporasi, rekomendasi analis]
<i>Review: ...</i>

<b>🎯 Bagian 5: Rekomendasi</b>
Rekomendasi: BUY / HOLD / SELL
Entry: ... | Target: ... | Stop Loss: ...
Horizon: short / medium / long term
Risk/Reward: ...
<i>Review: ringkasan risiko dan potensi keuntungan</i>

Gunakan HTML bold/italic yang sesuai. Pastikan analisis berbasis data yang diberikan, bukan asumsi.
"""
    return prompt


# ── Main entry point ──────────────────────────────────────────────────────────

def analyze_stock(symbol: str) -> list[str]:
    """
    Run full analysis for a symbol.
    Returns a list of message strings (split for Telegram's 4096-char limit).
    """
    sym = symbol.upper()
    if not sym.endswith(".JK"):
        sym += ".JK"

    logger.info(f"Starting deep analysis for {sym}...")

    # 1. Fetch all data
    fund         = _fetch_fundamentals(sym)
    tf_data      = _fetch_multiframe(sym)
    comp_symbols = COMPETITORS.get(sym, DEFAULT_COMPETITORS)
    competitors  = [_fetch_competitor_snapshot(c) for c in comp_symbols]
    news         = fetch_news(sym, max_age_hours=72)
    announcements = fetch_idx_announcements(sym)

    # 2. Calculate indicators per timeframe
    tf_indicators = {}
    for key in ["daily", "weekly", "monthly"]:
        df = tf_data.get(key)
        if df is not None and not df.empty:
            tf_indicators[key] = _calc_indicators(df, key)

    if not fund and not tf_indicators:
        return [f"❌ Tidak dapat mengambil data untuk {sym}. Cek kode saham."]

    # 3. Build prompt
    prompt = _build_prompt(sym, fund, tf_indicators, competitors, news, announcements)

    # 4. Call Claude
    if not config.ANTHROPIC_API_KEY:
        return ["❌ ANTHROPIC_API_KEY tidak dikonfigurasi."]

    try:
        import anthropic
        client   = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        full_text = response.content[0].text.strip()
        logger.info(f"Analysis complete for {sym} ({len(full_text)} chars)")
        return _split_message(full_text)

    except Exception as e:
        logger.error(f"analyze_stock({sym}) Claude call failed: {e}")
        return [f"❌ Analisis gagal: {e}"]


def _split_message(text: str, limit: int = 4000) -> list[str]:
    """
    Split a long message into Telegram-safe chunks at section boundaries.
    Tries to break at double-newlines (section breaks) rather than mid-sentence.
    """
    if len(text) <= limit:
        return [text]

    parts  = []
    current = ""
    for para in text.split("\n\n"):
        block = para + "\n\n"
        if len(current) + len(block) > limit:
            if current:
                parts.append(current.rstrip())
            current = block
        else:
            current += block

    if current.strip():
        parts.append(current.rstrip())

    return parts if parts else [text[:limit]]
