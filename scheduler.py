"""
Runs run_bot.run_once() automatically once per weekday shortly after market
open, so this can be deployed on a small always-on cloud host (e.g. a
$5-7/mo VM, Railway, or Render) for a 24/7-scheduled bot.

Usage: python scheduler.py   (leave running in the background / in a screen
session / as a systemd or supervisor service)
"""
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from run_bot import run_once

# 9:35am US/Eastern, Mon-Fri: a few minutes after the market opens at 9:30.
TRIGGER = CronTrigger(day_of_week="mon-fri", hour=9, minute=35, timezone="US/Eastern")


def main():
    scheduler = BlockingScheduler()
    scheduler.add_job(run_once, TRIGGER, id="daily_trading_run")
    print("Scheduler started. Bot will run weekdays at 9:35am US/Eastern.")
    print("Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
