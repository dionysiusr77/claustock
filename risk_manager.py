"""
IDX-specific risk management.
Gates every signal before it fires:
  Gate 1 — daily loss limit not breached
  Gate 2 — not already holding this stock (T+2 check)
  Gate 3 — enough capital remaining

Also handles lot sizing and P&L tracking.
"""

import logging
from datetime import datetime, timezone, timedelta

import config
import firestore_client as db

logger = logging.getLogger(__name__)

# In-memory P&L tracker for the current trading day
# { symbol: { entry_price, lots, entry_time, status: "open"|"closed", exit_price } }
_positions: dict = {}
_day_realized_pnl: float = 0.0  # IDR, reset each trading day
_last_reset_date: str = ""


# ── Daily reset ───────────────────────────────────────────────────────────────

def _maybe_reset_day():
    global _day_realized_pnl, _last_reset_date, _positions
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    if today != _last_reset_date:
        _day_realized_pnl = 0.0
        _positions        = {}
        _last_reset_date  = today
        logger.info("Risk manager: daily reset")


# ── Lot sizing ────────────────────────────────────────────────────────────────

def calc_lot_size(price: float, confidence: int) -> int:
    """
    Calculate how many lots to buy.
    Base allocation: IDX_CAPITAL * 10% per stock / (price * 100 shares/lot)
    Scale 0.5x–1.0x based on AI confidence (0–100).
    Always rounds DOWN. Returns at least 1 lot.
    """
    if price <= 0:
        return 1

    base_capital = config.IDX_CAPITAL * 0.10          # 10% of capital per stock
    conf_scale   = 0.5 + (confidence / 100) * 0.5     # 0.5 at 0% → 1.0 at 100%
    raw_lots     = (base_capital * conf_scale) / (price * 100)
    lots         = max(1, min(int(raw_lots), config.IDX_MAX_LOTS))
    return lots


# ── Risk gates ────────────────────────────────────────────────────────────────

def check_risk_gates(symbol: str) -> tuple[bool, str]:
    """
    Run all 3 risk gates before allowing a signal to fire.
    Returns (allowed: bool, reason: str).
    """
    _maybe_reset_day()

    # Gate 1 — daily loss limit
    loss_limit_idr = config.IDX_CAPITAL * (config.IDX_DAILY_LOSS_PCT / 100)
    if _day_realized_pnl < -loss_limit_idr:
        msg = f"Daily loss limit breached ({_day_realized_pnl/1e6:.2f}M IDR)"
        logger.warning(f"Risk gate 1 BLOCKED {symbol}: {msg}")
        return False, msg

    # Gate 2 — T+2 check (already in open position for this symbol)
    if symbol in _positions and _positions[symbol]["status"] == "open":
        msg = f"Already holding {symbol} (T+2 — not settled yet)"
        logger.warning(f"Risk gate 2 BLOCKED {symbol}: {msg}")
        return False, msg

    # Gate 3 — capital check
    open_positions = [p for p in _positions.values() if p["status"] == "open"]
    max_concurrent = max(1, config.IDX_CAPITAL // (config.IDX_CAPITAL // 4))  # max 4 concurrent
    if len(open_positions) >= max_concurrent:
        msg = f"Max concurrent positions reached ({len(open_positions)})"
        logger.warning(f"Risk gate 3 BLOCKED {symbol}: {msg}")
        return False, msg

    return True, "ok"


# ── Position tracking ─────────────────────────────────────────────────────────

def record_entry(symbol: str, price: float, lots: int):
    """Record a new position entry."""
    _maybe_reset_day()
    _positions[symbol] = {
        "entry_price": price,
        "lots":        lots,
        "entry_time":  datetime.now(timezone.utc).isoformat(),
        "status":      "open",
        "exit_price":  None,
        "pnl_idr":     None,
    }
    logger.info(f"Position opened: {symbol} @ {price:,.0f} x {lots} lots")


def record_exit(symbol: str, exit_price: float):
    """Record a position exit and update daily P&L."""
    global _day_realized_pnl
    _maybe_reset_day()

    if symbol not in _positions:
        logger.warning(f"record_exit: no open position for {symbol}")
        return

    pos   = _positions[symbol]
    entry = pos["entry_price"]
    lots  = pos["lots"]
    gross = (exit_price - entry) * lots * 100
    fees  = (entry * lots * 100 * config.BUY_FEE_PCT) + (exit_price * lots * 100 * config.SELL_FEE_PCT)
    net   = gross - fees

    _day_realized_pnl += net
    pos["exit_price"] = exit_price
    pos["pnl_idr"]    = net
    pos["status"]     = "closed"

    logger.info(
        f"Position closed: {symbol} @ {exit_price:,.0f} | "
        f"P&L: {net/1000:+.1f}K IDR | Day total: {_day_realized_pnl/1000:+.1f}K IDR"
    )


# ── P&L summary ───────────────────────────────────────────────────────────────

def get_pnl_summary() -> dict:
    """
    Return today's P&L summary based on fired signals and current prices.
    Uses Firestore signals for entry prices, latest snapshots for current price.
    """
    _maybe_reset_day()

    signals = db.get_today_signals()
    rows    = []
    total_unrealized = 0.0

    for sig in signals:
        symbol      = sig.get("symbol", "")
        entry_price = sig.get("price", 0)
        lots        = 1
        ai          = sig.get("ai_verdict") or {}
        if isinstance(ai, dict):
            lots = ai.get("lots", 1)

        # Get latest price for unrealized P&L
        snap = db.get_latest_snapshot(symbol)
        cur_price = snap.get("price", entry_price) if snap else entry_price

        gross       = (cur_price - entry_price) * lots * 100
        fees        = (entry_price * lots * 100 * config.BUY_FEE_PCT)
        unrealized  = gross - fees
        pct         = (cur_price - entry_price) / entry_price * 100 if entry_price else 0

        total_unrealized += unrealized
        rows.append({
            "symbol":      symbol,
            "entry":       entry_price,
            "current":     cur_price,
            "lots":        lots,
            "pnl_idr":     unrealized,
            "pnl_pct":     pct,
        })

    return {
        "signals_count":     len(signals),
        "realized_pnl_idr":  _day_realized_pnl,
        "unrealized_pnl_idr": total_unrealized,
        "total_pnl_idr":     _day_realized_pnl + total_unrealized,
        "rows":              rows,
        "capital":           config.IDX_CAPITAL,
    }


def get_t2_positions() -> list[str]:
    """Returns list of symbols currently in T+2 (open position)."""
    _maybe_reset_day()
    return [sym for sym, pos in _positions.items() if pos["status"] == "open"]
