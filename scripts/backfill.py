"""
CLI script: backfill GDELT data for a date range.

Usage:
    python scripts/backfill.py --days 30
    python scripts/backfill.py --start 2024-01-01 --end 2024-03-31
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.ingestion_pipeline import IngestionPipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser(description="GDELT backfill CLI")
    parser.add_argument("--days", type=int, default=30,
                        help="Number of days to backfill (default: 30)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--force", action="store_true",
                        help="Re-ingest even if date already exists")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Config file path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = PipelineConfig.from_yaml(args.config)

    end_date = None
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    with IngestionPipeline(cfg) as pipeline:
        if args.start:
            start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
            end_date   = end_date or (date.today() - __import__("datetime").timedelta(days=1))
            days = (end_date - start_date).days + 1
            results = pipeline.backfill(
                days=days,
                end_date=end_date,
                skip_if_exists=not args.force,
            )
        else:
            results = pipeline.backfill(
                days=args.days,
                end_date=end_date,
                skip_if_exists=not args.force,
            )

    success = sum(1 for r in results if r.success)
    total_events = sum(r.events_inserted for r in results)

    print(f"\n{'='*50}")
    print(f"Backfill complete: {success}/{len(results)} dates")
    print(f"Total events inserted: {total_events:,}")
    print(f"{'='*50}")

    failed = [r for r in results if not r.success]
    if failed:
        print(f"\nFailed dates ({len(failed)}):")
        for r in failed:
            print(f"  {r.target_date}: {r.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
