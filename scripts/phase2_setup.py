"""
Phase 2 setup script.

Applies the Phase 2 DB migration and runs initial analysis:
  1. Apply init_v2.sql migration
  2. Generate proxy labels for ingested data
  3. Compute event clusters
  4. Compute spillover network

Usage:
    python scripts/phase2_setup.py --days 30
    python scripts/phase2_setup.py --days 30 --skip-migration
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase2_setup")


def load_dsn(config_path: str = "configs/config.yaml") -> str:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    db = cfg["database"]
    return (
        f"postgresql://{db['user']}:{db['password']}@"
        f"{db['host']}:{db['port']}/{db['name']}"
    )


def apply_migration(dsn: str, migration_path: str = "docker/init_v2.sql") -> None:
    logger.info("Applying Phase 2 migration: %s", migration_path)
    sql = Path(migration_path).read_text(encoding="utf-8")
    conn = psycopg2.connect(dsn)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        logger.info("Migration applied successfully")
    except Exception as exc:
        logger.error("Migration failed: %s", exc)
        raise
    finally:
        conn.close()


def generate_labels(dsn: str, start: date, end: date) -> None:
    logger.info("Generating proxy labels: %s → %s", start, end)
    from preprocessing.label_generator import LabelGenerator
    gen = LabelGenerator(dsn=dsn)
    results = gen.compute_labels_range(start, end)
    total = sum(results.values())
    logger.info("Labels generated: %d country-date rows", total)


def compute_clusters(dsn: str, start: date, end: date) -> None:
    logger.info("Computing event clusters: %s → %s", start, end)
    from preprocessing.event_clusterer import EventClusterer
    clusterer = EventClusterer(dsn=dsn)
    total = clusterer.compute_range(start, end)
    logger.info("Event clusters computed: %d rows", total)


def compute_spillover(dsn: str) -> None:
    logger.info("Computing spillover network...")
    from inference.spillover import SpilloverAnalyzer
    analyzer = SpilloverAnalyzer(dsn=dsn, window_days=90)
    n = analyzer.compute_and_save()
    logger.info("Spillover network: %d pairs", n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 setup")
    parser.add_argument("--days", type=int, default=30,
                        help="Days of history to process (default 30)")
    parser.add_argument("--skip-migration", action="store_true",
                        help="Skip DB migration (if already applied)")
    parser.add_argument("--skip-labels", action="store_true")
    parser.add_argument("--skip-clusters", action="store_true")
    parser.add_argument("--skip-spillover", action="store_true")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    dsn = load_dsn(args.config)
    end   = date.today() - timedelta(days=1)
    start = end - timedelta(days=args.days - 1)

    if not args.skip_migration:
        apply_migration(dsn)

    if not args.skip_labels:
        generate_labels(dsn, start, end)

    if not args.skip_clusters:
        compute_clusters(dsn, start, end)

    if not args.skip_spillover:
        compute_spillover(dsn)

    logger.info("Phase 2 setup complete ✓")


if __name__ == "__main__":
    main()
