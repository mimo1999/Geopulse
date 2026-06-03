"""
Rolling normalization for country feature scores.

Computes z-score normalization per country over a rolling window,
saved as statistics to PostgreSQL for inference-time use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import psycopg2
import psycopg2.extras

logger = logging.getLogger("preprocessing.normalizer")


FEATURE_COLUMNS = [
    "protest_score",
    "violence_score",
    "diplomatic_stress",
    "economic_stress",
    "terrorism_score",
    "avg_sentiment",
    "avg_goldstein",
]


@dataclass
class CountryNormStats:
    country: str
    feature: str
    mean: float
    std: float
    computed_date: date
    window_days: int

    def normalize(self, value: float) -> float:
        if self.std < 1e-9:
            return 0.0
        return (value - self.mean) / self.std

    def clip_normalize(self, value: float, clip: float = 3.0) -> float:
        z = self.normalize(value)
        return max(-clip, min(clip, z))


class RollingNormalizer:
    """
    Maintains per-country rolling normalization statistics.

    These stats are used at inference time to normalize the
    feature sequence fed into the PyTorch model.
    """

    _CREATE_STATS_TABLE = """
        CREATE TABLE IF NOT EXISTS feature_norm_stats (
            country         TEXT        NOT NULL,
            feature_name    TEXT        NOT NULL,
            mean            FLOAT       NOT NULL,
            std             FLOAT       NOT NULL,
            computed_date   DATE        NOT NULL,
            window_days     INT         NOT NULL,
            PRIMARY KEY (country, feature_name, computed_date)
        )
    """

    _UPSERT_STATS = """
        INSERT INTO feature_norm_stats
            (country, feature_name, mean, std, computed_date, window_days)
        VALUES (%(country)s, %(feature_name)s, %(mean)s, %(std)s,
                %(computed_date)s, %(window_days)s)
        ON CONFLICT (country, feature_name, computed_date)
        DO UPDATE SET mean = EXCLUDED.mean, std = EXCLUDED.std
    """

    _FETCH_WINDOW = """
        SELECT {features}
        FROM country_daily_features
        WHERE country = %s
          AND feature_date >= %s
          AND feature_date <= %s
        ORDER BY feature_date
    """

    _FETCH_STATS = """
        SELECT feature_name, mean, std
        FROM feature_norm_stats
        WHERE country = %s
          AND computed_date = (
              SELECT MAX(computed_date)
              FROM feature_norm_stats
              WHERE country = %s
          )
    """

    def __init__(self, dsn: str, window_days: int = 90):
        self._dsn = dsn
        self._window = window_days

    def compute_and_save(
        self,
        countries: list[str] | None = None,
        as_of: Optional[date] = None,
    ) -> int:
        """
        Compute rolling statistics for all countries and persist.

        Returns number of (country, feature) rows saved.
        """
        if as_of is None:
            as_of = date.today()

        window_start = as_of - timedelta(days=self._window)

        conn = psycopg2.connect(self._dsn)
        try:
            # Ensure stats table exists
            with conn, conn.cursor() as cur:
                cur.execute(self._CREATE_STATS_TABLE)

            if countries is None:
                with conn, conn.cursor() as cur:
                    cur.execute(
                        "SELECT DISTINCT country FROM country_daily_features"
                    )
                    countries = [r[0] for r in cur.fetchall()]

            total = 0
            feat_select = ", ".join(FEATURE_COLUMNS)

            for country in countries:
                with conn, conn.cursor() as cur:
                    cur.execute(
                        self._FETCH_WINDOW.format(features=feat_select),
                        (country, window_start, as_of),
                    )
                    data = cur.fetchall()

                if len(data) < 3:
                    continue   # not enough history

                arr = np.array(data, dtype=np.float64)   # (T, F)
                means = np.nanmean(arr, axis=0)
                stds  = np.nanstd(arr, axis=0)

                rows = []
                for i, feat in enumerate(FEATURE_COLUMNS):
                    rows.append({
                        "country":       country,
                        "feature_name":  feat,
                        "mean":          float(means[i]),
                        "std":           float(stds[i]) if stds[i] > 1e-9 else 1.0,
                        "computed_date": as_of,
                        "window_days":   self._window,
                    })

                with conn, conn.cursor() as cur:
                    psycopg2.extras.execute_batch(cur, self._UPSERT_STATS, rows)
                total += len(rows)

            logger.info("Saved %d normalization stat rows", total)
            return total
        finally:
            conn.close()

    def load_stats(self, country: str) -> dict[str, CountryNormStats]:
        """Load latest normalization stats for a country."""
        conn = psycopg2.connect(self._dsn)
        try:
            with conn, conn.cursor() as cur:
                cur.execute(self._FETCH_STATS, (country, country))
                rows = cur.fetchall()
            return {
                row[0]: CountryNormStats(
                    country=country,
                    feature=row[0],
                    mean=row[1],
                    std=row[2],
                    computed_date=date.today(),
                    window_days=self._window,
                )
                for row in rows
            }
        finally:
            conn.close()
