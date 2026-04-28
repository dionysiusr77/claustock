"""
Entry point.
Starts the Telegram bot with two scheduled jobs:
  16:30 WIB Mon–Fri — EOD D-1 scan
  08:30 WIB Mon–Fri — Morning briefing delivery
"""

import datetime
import logging

import pytz

import config
from telegram_bot import build_app, job_eod_scan, job_morning_briefing

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
# Quiet down noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
_WIB   = pytz.timezone(config.MARKET_TZ)


def main() -> None:
    logger.info("Starting Claustock IDX v2 (universe=%s)", config.UNIVERSE)

    app = build_app()
    jq  = app.job_queue

    # ── Scheduled jobs (Mon–Fri only) ─────────────────────────────────────────
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

    logger.info(
        "Jobs scheduled: EOD scan %02d:%02d WIB | Briefing %02d:%02d WIB",
        *config.EOD_SCAN_TIME,
        *config.BRIEFING_TIME,
    )

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
