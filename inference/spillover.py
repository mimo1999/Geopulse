"""
Phase 2: Country spillover / contagion network.

Computes pairwise country relationships from two signals:
  1. Risk correlation: Pearson correlation of risk_score time series
     over a rolling window. High correlation → similar risk trajectories
     (shared drivers or contagion).

  2. Co-occurrence: events where both countries appear as actor1 + actor2.
     High co-occurrence → direct bilateral relationship in GDELT.

Combined spillover weight:
    spillover = 0.5 × |risk_correlation| + 0.5 × log1p(cooccurrence) / log1p(max_cooc)

Results persisted to country_spillover table.
Queried by the API and Streamlit drilldown for "Related Countries" display.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from itertools import combinations
from typing import Any

import numpy as np
import psycopg2
import psycopg2.extras

logger = logging.getLogger("inference.spillover")


# Hardcoded adjacency pairs (ISO alpha-2)
# Kept small — only major risk-relevant border pairs
ADJACENT_PAIRS: set[frozenset] = {
    frozenset({"IN", "PK"}), frozenset({"IN", "CN"}), frozenset({"IN", "AF"}),
    frozenset({"PK", "AF"}), frozenset({"RU", "UA"}), frozenset({"RU", "BY"}),
    frozenset({"IL", "LB"}), frozenset({"IL", "SY"}), frozenset({"IR", "IQ"}),
    frozenset({"KP", "KR"}), frozenset({"CN", "TW"}), frozenset({"US", "MX"}),
    frozenset({"SA", "YE"}), frozenset({"ET", "SO"}), frozenset({"SD", "SS"}),
}


class SpilloverAnalyzer:
    """
    Computes and persists the country spillover network.

    Usage::

        analyzer = SpilloverAnalyzer(dsn)
        analyzer.compute_and_save(as_of=date.today(), window_days=90)
    """

    _FETCH_RISK_SERIES_SQL = """
        SELECT country, feature_date, risk_score
        FROM country_daily_features
        WHERE feature_date >= %s AND feature_date <= %s
          AND risk_score IS NOT NULL
        ORDER BY country, feature_date
    """

    _FETCH_COOCCURRENCE_SQL = """
        SELECT
            actor1_country,
            actor2_country,
            COUNT(*) AS pair_count
        FROM gdelt_events
        WHERE event_date >= %s AND event_date <= %s
          AND actor1_country IS NOT NULL
          AND actor2_country IS NOT NULL
          AND actor1_country != actor2_country
          AND quad_class IN (3, 4)
        GROUP BY actor1_country, actor2_country
        HAVING COUNT(*) >= 5
    """

    _UPSERT_SPILLOVER_SQL = """
        INSERT INTO country_spillover (
            country_a, country_b, computed_date,
            risk_correlation, cooccurrence_count, cooccurrence_score,
            is_adjacent, spillover_weight
        ) VALUES (
            %(country_a)s, %(country_b)s, %(computed_date)s,
            %(risk_correlation)s, %(cooccurrence_count)s, %(cooccurrence_score)s,
            %(is_adjacent)s, %(spillover_weight)s
        )
        ON CONFLICT (country_a, country_b, computed_date)
        DO UPDATE SET
            risk_correlation    = EXCLUDED.risk_correlation,
            cooccurrence_count  = EXCLUDED.cooccurrence_count,
            cooccurrence_score  = EXCLUDED.cooccurrence_score,
            spillover_weight    = EXCLUDED.spillover_weight
    """

    _FETCH_NEIGHBORS_SQL = """
        SELECT * FROM country_top_neighbors
        WHERE country = %s
        ORDER BY spillover_weight DESC
    """

    def __init__(
        self,
        dsn: str,
        window_days: int = 90,
        min_overlap_days: int = 30,
        top_n_pairs: int = 500,
    ):
        self._dsn = dsn
        self._window = window_days
        self._min_overlap = min_overlap_days
        self._top_n = top_n_pairs

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute_and_save(self, as_of: Optional[date] = None) -> int:
        """
        Compute full spillover network and persist.
        Returns number of pairs saved.
        """
        from typing import Optional
        if as_of is None:
            as_of = date.today()

        window_start = as_of - timedelta(days=self._window)

        conn = psycopg2.connect(self._dsn)
        try:
            # 1. Risk correlation matrix
            corr_matrix = self._compute_risk_correlations(conn, window_start, as_of)

            # 2. Co-occurrence counts
            cooc_matrix = self._compute_cooccurrence(conn, window_start, as_of)

            # 3. Combine + select top pairs
            rows = self._build_spillover_rows(corr_matrix, cooc_matrix, as_of)

            if not rows:
                logger.warning("No spillover pairs computed")
                return 0

            # 4. Persist
            with conn, conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, self._UPSERT_SPILLOVER_SQL, rows)

            logger.info("Spillover network: %d pairs saved for %s", len(rows), as_of)
            return len(rows)
        finally:
            conn.close()

    def fetch_neighbors(
        self,
        country: str,
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Fetch top spillover neighbors for a country (from view)."""
        conn = psycopg2.connect(self._dsn)
        try:
            with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(self._FETCH_NEIGHBORS_SQL, (country,))
                rows = cur.fetchmany(top_n)
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("Failed to fetch neighbors for %s: %s", country, exc)
            return []
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Internal computations
    # ------------------------------------------------------------------

    def _compute_risk_correlations(
        self,
        conn,
        start: date,
        end: date,
    ) -> dict[tuple[str, str], float]:
        """
        Compute pairwise Pearson correlations of risk_score time series.
        Returns {(country_a, country_b): correlation} for significant pairs.
        """
        with conn, conn.cursor() as cur:
            cur.execute(self._FETCH_RISK_SERIES_SQL, (start, end))
            rows = cur.fetchall()

        if not rows:
            return {}

        # Build per-country arrays keyed by date
        from collections import defaultdict
        country_series: dict[str, dict[date, float]] = defaultdict(dict)
        for country, d, score in rows:
            country_series[country][d] = float(score)

        countries = list(country_series.keys())
        n = len(countries)
        if n < 2:
            return {}

        # Build dense date index
        all_dates = sorted(set(
            d for series in country_series.values() for d in series
        ))
        date_idx = {d: i for i, d in enumerate(all_dates)}
        T = len(all_dates)

        # Build matrix (C, T)
        mat = np.full((n, T), np.nan)
        for i, country in enumerate(countries):
            for d, v in country_series[country].items():
                mat[i, date_idx[d]] = v

        # Compute pairwise Pearson (vectorized)
        correlations: dict[tuple[str, str], float] = {}
        for i, j in combinations(range(n), 2):
            # Find overlapping non-NaN positions
            valid = ~(np.isnan(mat[i]) | np.isnan(mat[j]))
            if valid.sum() < self._min_overlap:
                continue

            xi, xj = mat[i, valid], mat[j, valid]
            corr = float(np.corrcoef(xi, xj)[0, 1])

            if np.isnan(corr):
                continue

            a, b = countries[i], countries[j]
            if a > b:
                a, b = b, a
            correlations[(a, b)] = corr

        logger.debug(
            "Risk correlations computed: %d country pairs", len(correlations)
        )
        return correlations

    def _compute_cooccurrence(
        self,
        conn,
        start: date,
        end: date,
    ) -> dict[tuple[str, str], int]:
        """
        Compute bilateral event co-occurrence counts from GDELT.
        Only conflict events (quad_class 3,4) for signal quality.
        """
        with conn, conn.cursor() as cur:
            cur.execute(self._FETCH_COOCCURRENCE_SQL, (start, end))
            rows = cur.fetchall()

        cooc: dict[tuple[str, str], int] = {}
        for a1, a2, count in rows:
            a, b = (a1, a2) if a1 < a2 else (a2, a1)
            cooc[(a, b)] = cooc.get((a, b), 0) + count

        return cooc

    def _build_spillover_rows(
        self,
        corr_matrix: dict[tuple[str, str], float],
        cooc_matrix: dict[tuple[str, str], int],
        computed_date: date,
    ) -> list[dict[str, Any]]:
        """Merge correlation and co-occurrence into spillover rows."""
        all_pairs = set(corr_matrix.keys()) | set(cooc_matrix.keys())

        if not all_pairs:
            return []

        max_cooc = max(cooc_matrix.values()) if cooc_matrix else 1

        rows = []
        for (a, b) in all_pairs:
            corr  = corr_matrix.get((a, b), 0.0)
            cooc  = cooc_matrix.get((a, b), 0)

            # Normalized co-occurrence: 0–1 via log scale
            cooc_norm = np.log1p(cooc) / np.log1p(max_cooc)

            # Combined spillover weight
            weight = 0.5 * abs(corr) + 0.5 * cooc_norm

            # Adjacency bonus
            is_adj = frozenset({a, b}) in ADJACENT_PAIRS
            if is_adj:
                weight = min(weight + 0.15, 1.0)

            # Canonical ordering: always store lex-smaller code as country_a
            # so (A,B) and (B,A) map to the same pair in the DB.
            ca, cb = (a, b) if a <= b else (b, a)
            rows.append({
                "country_a":          ca,
                "country_b":          cb,
                "computed_date":      computed_date,
                "risk_correlation":   round(corr, 6),
                "cooccurrence_count": cooc,
                "cooccurrence_score": round(float(cooc_norm), 6),
                "is_adjacent":        is_adj,
                "spillover_weight":   round(float(weight), 6),
            })

        # Sort by spillover weight, keep top N
        rows.sort(key=lambda x: -x["spillover_weight"])
        return rows[: self._top_n]

    # ------------------------------------------------------------------
    # Type annotation fix (used in Optional)
    # ------------------------------------------------------------------


from typing import Optional   # noqa: E402  (re-export for class body usage)
