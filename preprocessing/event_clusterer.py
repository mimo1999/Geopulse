"""
Phase 2: Event clusterer.

Groups raw GDELT events per (country, date) into semantic clusters
for the country drilldown UI:
    - protest       CAMEO 14x
    - military      CAMEO 15x–17x, 19, 20
    - terrorism     CAMEO 18x–20x (asymmetric)
    - sanctions     CAMEO 163, 168
    - diplomatic    CAMEO 1x–3x, verbal conflict 11–13

Each cluster stores: event_count, total_mentions, avg_goldstein,
avg_tone, max_intensity, top_actor_pairs (JSON).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

import psycopg2
import psycopg2.extras

logger = logging.getLogger("preprocessing.event_clusterer")


CATEGORY_RULES: dict[str, dict] = {
    "protest": {
        "root_codes":    {14},
        "base_codes":    {141, 142, 143, 144, 145},
        "quad_classes":  {3, 4},
    },
    "military": {
        "root_codes":    {15, 16, 17, 19, 20},
        "base_codes":    set(),
        "quad_classes":  {3, 4},
    },
    "terrorism": {
        "root_codes":    {18, 20},           # 19=conventional fight → military
        "base_codes":    {180, 181, 182, 183, 184, 185, 186},  # 190=conventional military force → military
        "quad_classes":  {4},
    },
    "sanctions": {
        "root_codes":    {16, 17},
        "base_codes":    {163, 168, 172, 173},
        "quad_classes":  {3, 4},
    },
    "diplomatic": {
        "root_codes":    {1, 2, 3, 11, 12, 13},
        "base_codes":    set(),
        "quad_classes":  {1, 2, 3},
    },
}


class EventClusterer:
    """
    Computes event clusters per (country, date) and persists them
    to the `event_clusters` table for fast UI retrieval.
    """

    _FETCH_EVENTS_SQL = """
        SELECT
            event_root_code,
            event_base_code,
            quad_class,
            goldstein,
            avg_tone,
            num_mentions,
            actor1_name,
            actor2_name,
            actor1_country,
            actor2_country
        FROM gdelt_events
        WHERE action_geo_country = %s
          AND event_date = %s
    """

    _UPSERT_CLUSTER_SQL = """
        INSERT INTO event_clusters (
            country, cluster_date, category,
            event_count, total_mentions, avg_goldstein, avg_tone,
            max_intensity, top_actor_pairs
        ) VALUES (
            %(country)s, %(cluster_date)s, %(category)s,
            %(event_count)s, %(total_mentions)s,
            %(avg_goldstein)s, %(avg_tone)s,
            %(max_intensity)s, %(top_actor_pairs)s
        )
        ON CONFLICT (country, cluster_date, category)
        DO UPDATE SET
            event_count    = EXCLUDED.event_count,
            total_mentions = EXCLUDED.total_mentions,
            avg_goldstein  = EXCLUDED.avg_goldstein,
            avg_tone       = EXCLUDED.avg_tone,
            max_intensity  = EXCLUDED.max_intensity,
            top_actor_pairs = EXCLUDED.top_actor_pairs,
            computed_at    = NOW()
    """

    _COUNTRIES_SQL = """
        SELECT DISTINCT action_geo_country
        FROM gdelt_events
        WHERE event_date = %s AND action_geo_country IS NOT NULL
    """

    def __init__(self, dsn: str):
        self._dsn = dsn

    def compute_for_date(
        self,
        target_date: date,
        countries: list[str] | None = None,
    ) -> int:
        """Compute and persist event clusters for all countries on a date."""
        conn = psycopg2.connect(self._dsn)
        try:
            if not countries:
                with conn, conn.cursor() as cur:
                    cur.execute(self._COUNTRIES_SQL, (target_date,))
                    countries = [r[0] for r in cur.fetchall()]

            rows_written = 0
            for country in countries:
                with conn, conn.cursor() as cur:
                    cur.execute(self._FETCH_EVENTS_SQL, (country, target_date))
                    events = cur.fetchall()

                if not events:
                    continue

                clusters = self._build_clusters(country, target_date, events)
                with conn, conn.cursor() as cur:
                    psycopg2.extras.execute_batch(cur, self._UPSERT_CLUSTER_SQL, clusters)
                rows_written += len(clusters)

            logger.info(
                "Event clusters: %d rows for %s", rows_written, target_date
            )
            return rows_written
        finally:
            conn.close()

    def compute_range(self, start: date, end: date) -> int:
        """Compute clusters for a date range."""
        total = 0
        current = start
        while current <= end:
            total += self.compute_for_date(current)
            current += timedelta(days=1)
        return total

    def fetch_clusters(
        self,
        country: str,
        target_date: date,
        conn=None,
    ) -> list[dict[str, Any]]:
        """Fetch precomputed clusters from DB (for API use)."""
        close = False
        if conn is None:
            conn = psycopg2.connect(self._dsn)
            close = True
        try:
            with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM event_clusters
                    WHERE country = %s AND cluster_date = %s
                    ORDER BY total_mentions DESC
                    """,
                    (country, target_date),
                )
                return [dict(r) for r in cur.fetchall()]
        finally:
            if close:
                conn.close()

    # ------------------------------------------------------------------
    # Cluster building
    # ------------------------------------------------------------------

    def _build_clusters(
        self,
        country: str,
        cluster_date: date,
        events: list[tuple],
    ) -> list[dict[str, Any]]:
        """Assign events to categories and aggregate per category."""
        # Buckets: category → list of event tuples
        buckets: dict[str, list[tuple]] = defaultdict(list)

        for event in events:
            root, base, quad = event[0], event[1], event[2]
            category = self._classify_event(root, base, quad)
            buckets[category].append(event)

        result = []
        for category, cat_events in buckets.items():
            if not cat_events:
                continue

            goldsteins = [e[3] for e in cat_events if e[3] is not None]
            tones      = [e[4] for e in cat_events if e[4] is not None]
            mentions   = [e[5] or 0 for e in cat_events]

            avg_g = float(sum(goldsteins) / len(goldsteins)) if goldsteins else None
            avg_t = float(sum(tones) / len(tones)) if tones else None
            max_i = float(max(abs(g) for g in goldsteins)) if goldsteins else None

            top_pairs = self._top_actor_pairs(cat_events, top_n=5)

            result.append({
                "country":         country,
                "cluster_date":    cluster_date,
                "category":        category,
                "event_count":     len(cat_events),
                "total_mentions":  sum(mentions),
                "avg_goldstein":   round(avg_g, 4) if avg_g is not None else None,
                "avg_tone":        round(avg_t, 4) if avg_t is not None else None,
                "max_intensity":   round(max_i, 4) if max_i is not None else None,
                "top_actor_pairs": json.dumps(top_pairs),
            })

        return result

    def _classify_event(
        self,
        root_code: int | None,
        base_code: int | None,
        quad_class: int | None,
    ) -> str:
        """Return the category for a single event.

        Priority (highest → lowest):
          1. terrorism  — base_code in 18x/190 set  (asymmetric violence)
          2. sanctions  — base_code in {163,168,172,173}  (economic coercion)
          3. military   — root_code in {15,16,17,19,20} AND quad_class in {3,4}
          4. protest    — root_code == 14
          5. diplomatic — fallback
        """
        if root_code is None:
            return "diplomatic"

        r_terror = CATEGORY_RULES["terrorism"]

        # 1. Terrorism — requires Material Conflict (quad 4) AND either:
        #    a) specific terrorism base-code (18x, 190), or
        #    b) root-code in {18, 19, 20}
        if quad_class in r_terror["quad_classes"]:
            if base_code in r_terror["base_codes"] or root_code in r_terror["root_codes"]:
                return "terrorism"

        # 2. Sanctions — specific base-codes take priority over root overlap
        if base_code in CATEGORY_RULES["sanctions"]["base_codes"]:
            return "sanctions"

        # 3. Military — root-code in {15–17, 19, 20} with any conflict quad
        r = CATEGORY_RULES["military"]
        if root_code in r["root_codes"] and quad_class in r["quad_classes"]:
            return "military"

        # 4. Protest
        if root_code in CATEGORY_RULES["protest"]["root_codes"]:
            return "protest"

        return "diplomatic"

    def _top_actor_pairs(
        self,
        events: list[tuple],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top-N actor pairs by co-occurrence count."""
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        for event in events:
            a1 = event[6] or event[8] or ""  # actor1_name or actor1_country
            a2 = event[7] or event[9] or ""  # actor2_name or actor2_country
            if a1 and a2:
                pair = (a1[:40], a2[:40])
                pair_counts[pair] += 1

        sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])
        return [
            {"actor1": p[0], "actor2": p[1], "count": c}
            for p, c in sorted_pairs[:top_n]
        ]
