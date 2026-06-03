"""
Country-level daily feature extractor.

Reads raw GDELT events from PostgreSQL and materializes
structured features into country_daily_features.

Features computed per (country, date):
  - protest_score       fraction of protest events (CAMEO 14x)
  - violence_score      fraction of quad_class=4 (material conflict)
  - diplomatic_stress   inverted avg goldstein for diplomatic events
  - economic_stress     fraction of sanction/economic events (CAMEO 15x, 163)
  - terrorism_score     fraction of terror/assault events (CAMEO 18x-20x)
  - avg_sentiment       average AvgTone (normalized -10..+10 → 0..1)
  - avg_goldstein       average GoldsteinScale (normalized → 0..1)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import psycopg2
import psycopg2.extras

logger = logging.getLogger("preprocessing.feature_extractor")


# CAMEO event root codes by category
PROTEST_CODES = {14}
VIOLENCE_CODES = {18, 19, 20}
TERROR_CODES   = {18, 19, 20}          # overlap, weighted separately
ECONOMIC_CODES = {15, 16, 163}
SANCTION_CODES = {163}
MILITARY_CODES = {15, 16, 17}
VERBAL_CONF    = {11, 12, 13}


class FeatureExtractor:
    """
    Extracts country-level daily features from raw GDELT events.

    Usage::

        extractor = FeatureExtractor(dsn="postgresql://...")
        extractor.compute_daily_features(date(2024, 3, 15))
    """

    # SQL to pull all events for a given country+date from partitioned table
    _FETCH_SQL = """
        SELECT
            event_root_code,
            quad_class,
            goldstein,
            avg_tone,
            num_mentions,
            num_sources
        FROM gdelt_events
        WHERE action_geo_country = %s
          AND event_date = %s
    """

    # Upsert into feature store
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

    # Get all distinct countries with events on a date
    _COUNTRIES_SQL = """
        SELECT DISTINCT action_geo_country
        FROM gdelt_events
        WHERE event_date = %s
          AND action_geo_country IS NOT NULL
          AND action_geo_country != ''
    """

    def __init__(self, dsn: str):
        self._dsn = dsn

    def compute_daily_features(
        self,
        target_date: date,
        countries: list[str] | None = None,
    ) -> int:
        """
        Compute and upsert daily features for all (or specified) countries.

        Returns:
            Number of country-date rows written.
        """
        conn = psycopg2.connect(self._dsn)
        try:
            with conn:
                with conn.cursor() as cur:
                    # Discover countries if not given
                    if not countries:
                        cur.execute(self._COUNTRIES_SQL, (target_date,))
                        countries = [row[0] for row in cur.fetchall()]

                    if not countries:
                        logger.info("No events found for %s", target_date)
                        return 0

                    logger.info(
                        "Computing features for %d countries on %s",
                        len(countries), target_date,
                    )

                    rows_written = 0
                    for country in countries:
                        cur.execute(self._FETCH_SQL, (country, target_date))
                        events = cur.fetchall()
                        if not events:
                            continue

                        feature_row = self._compute_features(country, target_date, events)
                        cur.execute(self._UPSERT_SQL, feature_row)
                        rows_written += 1

            logger.info(
                "Feature extraction done: %d rows for %s",
                rows_written, target_date,
            )
            return rows_written

        finally:
            conn.close()

    def compute_date_range(
        self,
        start: date,
        end: date,
    ) -> dict[date, int]:
        """Compute features for a range of dates. Returns {date: row_count}."""
        results = {}
        current = start
        while current <= end:
            count = self.compute_daily_features(current)
            results[current] = count
            current += timedelta(days=1)
        return results

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute_features(
        self,
        country: str,
        feature_date: date,
        events: list[tuple],
    ) -> dict[str, Any]:
        """
        Compute normalized feature scores from raw event tuples.

        Input rows: (event_root_code, quad_class, goldstein, avg_tone,
                     num_mentions, num_sources)
        """
        n = len(events)

        # Accumulators
        goldsteins: list[float] = []
        tones: list[float] = []
        conflict_count = 0
        coop_count = 0
        protest_count = 0
        violence_count = 0
        terror_count = 0
        economic_count = 0
        total_mentions = 0

        for row in events:
            root_code, quad_class, goldstein, avg_tone, mentions, _ = row

            if goldstein is not None:
                goldsteins.append(float(goldstein))
            if avg_tone is not None:
                tones.append(float(avg_tone))
            if mentions:
                total_mentions += int(mentions)

            # Category flags
            if quad_class in (3, 4):
                conflict_count += 1
            if quad_class in (1, 2):
                coop_count += 1
            if root_code in PROTEST_CODES:
                protest_count += 1
            if quad_class == 4 or root_code in VIOLENCE_CODES:
                violence_count += 1
            if root_code in TERROR_CODES and quad_class == 4:
                terror_count += 1
            if root_code in ECONOMIC_CODES:
                economic_count += 1

        # --- Normalized scores (0–1) ---
        def ratio(count: int) -> float:
            return min(count / n, 1.0)

        avg_goldstein_raw = sum(goldsteins) / len(goldsteins) if goldsteins else 0.0
        avg_tone_raw      = sum(tones) / len(tones) if tones else 0.0

        # Goldstein: -10 to +10 → 0 to 1 (inverted: more negative = higher stress)
        avg_goldstein_norm = (avg_goldstein_raw + 10.0) / 20.0
        diplomatic_stress  = 1.0 - avg_goldstein_norm   # higher = more conflict

        # Tone: roughly -100 to +100 → 0 to 1
        avg_sentiment_norm = (avg_tone_raw + 100.0) / 200.0
        avg_sentiment_norm = max(0.0, min(1.0, avg_sentiment_norm))

        return {
            "country":             country,
            "feature_date":        feature_date,
            "total_events":        n,
            "conflict_events":     conflict_count,
            "cooperation_events":  coop_count,
            "protest_score":       round(ratio(protest_count), 6),
            "violence_score":      round(ratio(violence_count), 6),
            "diplomatic_stress":   round(diplomatic_stress, 6),
            "economic_stress":     round(ratio(economic_count), 6),
            "terrorism_score":     round(ratio(terror_count), 6),
            "avg_sentiment":       round(avg_sentiment_norm, 6),
            "avg_goldstein":       round(avg_goldstein_norm, 6),
        }
