"""
CLI script: compute country daily features for a date range.

Usage:
    python scripts/compute_features.py --days 30
    python scripts/compute_features.py --date 2024-03-15
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.feature_extractor import FeatureExtractor
import yaml


def main():
    parser = argparse.ArgumentParser(description="Feature extraction CLI")
    parser.add_argument("--date", type=str, default=None,
                        help="Single date YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=7,
                        help="Days to process ending today-1 (default: 7)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    db = cfg.get("database", {})
    dsn = (
        f"postgresql://{db['user']}:{db['password']}@"
        f"{db['host']}:{db['port']}/{db['name']}"
    )

    extractor = FeatureExtractor(dsn=dsn)

    if args.date:
        d = datetime.strptime(args.date, "%Y-%m-%d").date()
        count = extractor.compute_daily_features(d)
        print(f"Computed features for {d}: {count} country-rows")
    else:
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=args.days - 1)
        results = extractor.compute_date_range(start, end)
        total = sum(results.values())
        print(f"Computed features for {args.days} days: {total} total rows")


if __name__ == "__main__":
    main()
