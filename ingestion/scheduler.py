"""
APScheduler-based ingestion scheduler.

Runs as:  python -m ingestion.scheduler
"""

from __future__ import annotations

import logging
import signal
import sys
from datetime import date
from pathlib import Path

logger = logging.getLogger("ingestion.scheduler")


def _setup_logging() -> None:
    import yaml, logging.config
    cfg_path = Path("configs/logging.yaml")
    if cfg_path.exists():
        with open(cfg_path) as f:
            logging.config.dictConfig(yaml.safe_load(f))
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


def daily_ingest_job() -> None:
    """Scheduled job: ingest yesterday's GDELT file."""
    logger.info("=== Daily GDELT ingest job triggered ===")
    from ingestion.ingestion_pipeline import IngestionPipeline, PipelineConfig
    cfg = PipelineConfig.from_yaml("configs/config.yaml")
    with IngestionPipeline(cfg) as pipeline:
        result = pipeline.ingest_latest()
    if result.success:
        logger.info("Daily ingest OK: %s events inserted", result.events_inserted)
    else:
        logger.error("Daily ingest FAILED: %s", result.error)


def startup_backfill(days: int = 30) -> None:
    """Run a backfill on first startup if DB is empty."""
    logger.info("Checking if backfill needed (last %d days)...", days)
    from ingestion.ingestion_pipeline import IngestionPipeline, PipelineConfig
    from ingestion.db_writer import DBWriter

    cfg = PipelineConfig.from_yaml("configs/config.yaml")
    db = DBWriter(dsn=cfg.dsn)
    db.connect()
    try:
        yesterday = date.today()
        from datetime import timedelta
        yesterday = date.today() - timedelta(days=1)
        if db.check_date_ingested(yesterday):
            logger.info("DB already populated — skipping startup backfill")
            return
    finally:
        db.disconnect()

    logger.info("DB empty — starting %d-day backfill", days)
    with IngestionPipeline(cfg) as pipeline:
        results = pipeline.backfill(days=days)
    success_count = sum(1 for r in results if r.success)
    logger.info("Startup backfill done: %d/%d dates", success_count, len(results))


def main() -> None:
    _setup_logging()
    logger.info("Starting GDELT ingestion scheduler")

    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("APScheduler not installed — pip install apscheduler")
        sys.exit(1)

    # Run startup backfill (will skip if DB already has data)
    try:
        startup_backfill(days=30)
    except Exception as exc:
        logger.error("Startup backfill error: %s", exc, exc_info=True)

    # Schedule daily at 02:00 UTC
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        daily_ingest_job,
        trigger=CronTrigger(hour=2, minute=0),
        id="gdelt_daily",
        name="GDELT Daily Ingestion",
        misfire_grace_time=3600,   # allow up to 1h late
        coalesce=True,
    )

    def _shutdown(sig, frame):
        logger.info("Shutdown signal received")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info("Scheduler running — daily job at 02:00 UTC")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
