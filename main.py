"""
Entry point.
Starts the Telegram bot with three scheduled jobs:
  16:30 WIB Mon–Fri — EOD D-1 scan
  08:30 WIB Mon–Fri — Morning briefing delivery
  13:15 WIB Mon–Fri — Pre-Sesi 2 midday briefing
"""

import datetime
import logging

import pytz

import config
from telegram_bot import build_app, job_eod_scan, job_eod_report, job_morning_briefing, job_midday_briefing

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
# Quiet down noisy libraries — keep Railway under 500 logs/sec
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)        # JSONDecodeError on ^JKSE handled gracefully
logging.getLogger("urllib3").setLevel(logging.ERROR)            # covers connectionpool + retry
logging.getLogger("peewee").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)         # suppress polling heartbeats
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("invezgo").setLevel(logging.WARNING)          # suppress SDK request traces

logger = logging.getLogger(__name__)
_WIB   = pytz.timezone(config.MARKET_TZ)


def main() -> None:
    logger.info("Starting Claustock IDX v2 (universe=%s)", config.UNIVERSE)

    app = build_app()
    jq  = app.job_queue

    # ── Scheduled jobs (Mon–Fri only) ─────────────────────────────────────────
    jq.run_daily(
        job_eod_report,
        time=datetime.time(
            config.EOD_REPORT_TIME[0],
            config.EOD_REPORT_TIME[1],
            tzinfo=_WIB,
        ),
        days=(0, 1, 2, 3, 4),
        name="eod_report",
    )

    jq.run_daily(
        job_eod_scan,
        time=datetime.time(
            config.EOD_SCAN_TIME[0],
            config.EOD_SCAN_TIME[1],
            tzinfo=_WIB,
        ),
        days=(0, 1, 2, 3, 4),
        name="eod_scan",
    )

    jq.run_daily(
        job_morning_briefing,
        time=datetime.time(
            config.BRIEFING_TIME[0],
            config.BRIEFING_TIME[1],
            tzinfo=_WIB,
        ),
        days=(0, 1, 2, 3, 4),
        name="morning_briefing",
    )

    jq.run_daily(
        job_midday_briefing,
        time=datetime.time(
            config.MIDDAY_TIME[0],
            config.MIDDAY_TIME[1],
            tzinfo=_WIB,
        ),
        days=(0, 1, 2, 3, 4),
        name="midday_briefing",
    )

    logger.info(
        "Jobs scheduled: EOD report %02d:%02d | EOD scan %02d:%02d | "
        "Briefing %02d:%02d | Midday %02d:%02d WIB",
        *config.EOD_REPORT_TIME,
        *config.EOD_SCAN_TIME,
        *config.BRIEFING_TIME,
        *config.MIDDAY_TIME,
    )

    app.run_polling(drop_pending_updates=True, allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()
