from datetime import datetime
import pytz
import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

import config

logger = logging.getLogger(__name__)

WIB = pytz.timezone(config.MARKET_TZ)


def is_trading_day(dt: datetime | None = None) -> bool:
    """
    Returns True if the given datetime (or now) is a weekday (Mon–Fri).
    TODO: Add IDX public holiday calendar check.
    """
    now = dt or datetime.now(WIB)
    return now.weekday() < 5  # 0=Mon … 4=Fri


def is_market_open(dt: datetime | None = None) -> bool:
    """
    Returns True if currently inside Session 1 or Session 2 on a trading day.
    Session 1: 09:00–12:00 WIB
    Session 2: 13:00–15:49 WIB  (Friday: 14:00–15:49)
    """
    now = dt or datetime.now(WIB)
    if not is_trading_day(now):
        return False

    h, m = now.hour, now.minute
    t = h * 60 + m  # minutes since midnight

    s1_start = config.SESSION1_START[0] * 60 + config.SESSION1_START[1]
    s1_end   = config.SESSION1_END[0]   * 60 + config.SESSION1_END[1]

    # Friday session 2 starts at 14:00 instead of 13:00
    if now.weekday() == 4:  # Friday
        s2_start = 14 * 60
    else:
        s2_start = config.SESSION2_START[0] * 60 + config.SESSION2_START[1]
    s2_end = config.SESSION2_END[0] * 60 + config.SESSION2_END[1]

    return (s1_start <= t < s1_end) or (s2_start <= t <= s2_end)


def build_scheduler(
    scan_fn,
    presession1_fn,
    presession2_fn,
    eod_fn,
) -> BlockingScheduler:
    """
    Build and return a configured APScheduler BlockingScheduler.

    scan_fn:         called every 5 min — guarded by is_market_open()
    presession1_fn:  called at 08:45 WIB Mon–Fri
    presession2_fn:  called at 12:30 WIB Mon–Fri
    eod_fn:          called at 16:00 WIB Mon–Fri
    """
    scheduler = BlockingScheduler(timezone=WIB)

    # ── 5-min scan (only executes during market hours) ────────────────────
    def _guarded_scan():
        if is_market_open():
            scan_fn()
        else:
            logger.debug("Scan skipped — market closed")

    scheduler.add_job(
        _guarded_scan,
        trigger=IntervalTrigger(seconds=config.SCAN_INTERVAL_SEC, timezone=WIB),
        id="scan",
        name="5-min market scan",
        max_instances=1,
        coalesce=True,
    )

    # ── Pre-session 1 briefing  08:45 WIB ────────────────────────────────
    scheduler.add_job(
        presession1_fn,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=config.BRIEFING1_TIME[0],
            minute=config.BRIEFING1_TIME[1],
            timezone=WIB,
        ),
        id="briefing1",
        name="Pre-session 1 briefing",
    )

    # ── Pre-session 2 briefing  12:30 WIB ────────────────────────────────
    scheduler.add_job(
        presession2_fn,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=config.BRIEFING2_TIME[0],
            minute=config.BRIEFING2_TIME[1],
            timezone=WIB,
        ),
        id="briefing2",
        name="Pre-session 2 briefing",
    )

    # ── End of day summary  16:00 WIB ─────────────────────────────────────
    scheduler.add_job(
        eod_fn,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=config.EOD_TIME[0],
            minute=config.EOD_TIME[1],
            timezone=WIB,
        ),
        id="eod",
        name="End of day summary",
    )

    return scheduler
