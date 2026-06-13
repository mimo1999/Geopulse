"""
POLECAT ingestion pipeline.

Reads POLECAT event data from data/POLECAT/dataverse_files.zip,
aggregates daily per-country features, and upserts them into the
country_daily_features table (same schema as feature_extractor.py).

Usage:
    # Full historical load (2018-2024)
    python -m ingestion.polecat_pipeline

    # Single date
    python -m ingestion.polecat_pipeline --date 2023-06-15

    # Specific years
    python -m ingestion.polecat_pipeline --years 2023 2024
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date
from pathlib import Path

import psycopg2
import psycopg2.extras

from ingestion.polecat_parser import iter_zip

logger = logging.getLogger("ingestion.polecat_pipeline")

_UPSERT_SQL = """
    INSERT INTO country_daily_features (
        country, feature_date,
        total_events, conflict_events, cooperation_events,
        protest_score, violence_score, diplomatic_stress,
        economic_stress, terrorism_score,
        avg_sentiment, avg_goldstein,
        computed_at
    ) VALUES (
        %(country)s, %(feature_date)s,
        %(total_events)s, %(conflict_events)s, %(cooperation_events)s,
        %(protest_score)s, %(violence_score)s, %(diplomatic_stress)s,
        %(economic_stress)s, %(terrorism_score)s,
        %(avg_sentiment)s, %(avg_goldstein)s,
        NOW()
    )
    ON CONFLICT (country, feature_date)
    DO UPDATE SET
        total_events        = EXCLUDED.total_events,
        conflict_events     = EXCLUDED.conflict_events,
        cooperation_events  = EXCLUDED.cooperation_events,
        protest_score       = EXCLUDED.protest_score,
        violence_score      = EXCLUDED.violence_score,
        diplomatic_stress   = EXCLUDED.diplomatic_stress,
        economic_stress     = EXCLUDED.economic_stress,
        terrorism_score     = EXCLUDED.terrorism_score,
        avg_sentiment       = EXCLUDED.avg_sentiment,
        avg_goldstein       = EXCLUDED.avg_goldstein,
        computed_at         = NOW()
"""

_AUDIT_SQL = """
    INSERT INTO ingestion_runs (
        source, run_date, rows_inserted, rows_updated, duration_seconds, status, notes
    ) VALUES (
        %(source)s, NOW(), %(rows_inserted)s, 0, %(duration_seconds)s, %(status)s, %(notes)s
    )
    ON CONFLICT DO NOTHING
"""


def _get_dsn() -> str:
    return (
        os.environ.get("DATABASE_URL")
        or f"postgresql://{os.environ.get('POSTGRES_USER','gldt')}:"
           f"{os.environ.get('POSTGRES_PASSWORD','gldt_secret')}@"
           f"{os.environ.get('POSTGRES_HOST','localhost')}:"
           f"{os.environ.get('POSTGRES_PORT','5432')}/"
           f"{os.environ.get('POSTGRES_DB','gdelt_risk')}"
    )


def run(
    zip_path: Path,
    dsn: str,
    target_date: date | None = None,
    years: list[int] | None = None,
    batch_size: int = 500,
    dry_run: bool = False,
) -> int:
    """
    Load POLECAT data into country_daily_features.

    Returns:
        Number of rows upserted.
    """
    t0 = time.time()
    rows_done = 0
    batch: list[dict] = []

    def flush(conn):
        nonlocal rows_done
        if not batch:
            return
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, _UPSERT_SQL, batch, page_size=batch_size)
        conn.commit()
        rows_done += len(batch)
        batch.clear()
        logger.info("Upserted %d rows total", rows_done)

    conn = None if dry_run else psycopg2.connect(dsn)
    try:
        for row in iter_zip(zip_path, target_date=target_date, years=years):
            batch.append(row)
            if len(batch) >= batch_size:
                if dry_run:
                    rows_done += len(batch)
                    batch.clear()
                    logger.info("[dry-run] Would upsert %d rows total", rows_done)
                else:
                    flush(conn)

        # Final partial batch
        if dry_run:
            rows_done += len(batch)
            logger.info("[dry-run] Would upsert %d rows total", rows_done)
        else:
            flush(conn)

        duration = round(time.time() - t0, 1)
        logger.info("POLECAT load complete: %d rows in %.1fs", rows_done, duration)

        if not dry_run:
            try:
                with conn.cursor() as cur:
                    cur.execute(_AUDIT_SQL, {
                        "source": "polecat",
                        "rows_inserted": rows_done,
                        "duration_seconds": duration,
                        "status": "success",
                        "notes": f"zip={zip_path.name} date={target_date} years={years}",
                    })
                conn.commit()
            except Exception:
                conn.rollback()  # audit failure is non-fatal

    except Exception as exc:
        logger.exception("POLECAT load failed: %s", exc)
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

    return rows_done


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Ingest POLECAT data into country_daily_features")
    parser.add_argument(
        "--zip",
        default="data/POLECAT/dataverse_files.zip",
        help="Path to POLECAT zip file",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Ingest only this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Restrict to these years, e.g. --years 2022 2023",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and count rows without writing to DB",
    )
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise FileNotFoundError(f"POLECAT zip not found: {zip_path}")

    target_date = date.fromisoformat(args.date) if args.date else None

    dsn = _get_dsn()
    run(
        zip_path=zip_path,
        dsn=dsn,
        target_date=target_date,
        years=args.years,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
