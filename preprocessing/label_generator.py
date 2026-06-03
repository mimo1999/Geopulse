"""
Phase 2: Proxy ground-truth label generator.

Derives four continuous (0–1) proxy labels from raw GDELT events
using CAMEO event codes, intensity thresholds, and rolling statistics.

Labels are *proxies*, not ground truth — they enable supervised
multi-task training without external annotation.

Label definitions
─────────────────
instability_label:
    Rolling conflict density + protest events.
    High when: quad_class∈{3,4} is dominant, goldstein < -3 frequently.

war_label:
    Military/combat event density with cross-border actor pairs.
    High when: root codes 15–20 (especially 19=Fight, 20=Mass violence)
    dominate, large mention counts, bilateral inter-state pairs.

terrorism_label:
    Asymmetric violence events.
    High when: root codes 180–190 (bombings, kidnappings, attacks)
    with high num_mentions.

financial_label:
    Sanctions + economic coercion events.
    High when: root codes 163 (Impose sanctions), 168 (Impose embargo),
    17x (Coerce), negative economic tone.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import numpy as np
import psycopg2
import psycopg2.extras

logger = logging.getLogger("preprocessing.label_generator")


# ---------------------------------------------------------------------------
# CAMEO code sets
# ---------------------------------------------------------------------------

WAR_ROOT_CODES       = {15, 16, 17, 18, 19, 20}
WAR_HIGH_CODES       = {19, 20}            # Fight, Mass violence
TERRORISM_CODES      = {18, 19, 20}        # Assault / fight / mass violence
TERRORISM_BASE_CODES = {180, 181, 182, 183, 184, 185, 186}  # Specific acts
PROTEST_CODES        = {14}
RIOT_CODES           = {145}               # Riot / violent protest base
SANCTION_CODES       = {163, 168}
ECONOMIC_CODES       = {15, 16, 163, 168, 172, 173}
MILITARY_CODES       = {15, 195, 196}


class LabelGenerator:
    """
    Generates multi-task proxy labels for (country, date) pairs.

    Algorithm per label
    ───────────────────
    For each (country, date):
      1. Fetch events ± smoothing_days for context
      2. Compute raw signal (weighted event ratio)
      3. Smooth with exponential decay
      4. Normalize to 0–1 via sigmoid transform

    Call flow:
        gen = LabelGenerator(dsn)
        gen.compute_labels_for_date(date(2024, 3, 15))
        gen.compute_labels_range(start, end)
    """

    _FETCH_EVENTS_SQL = """
        SELECT
            event_root_code,
            event_base_code,
            quad_class,
            goldstein,
            avg_tone,
            num_mentions,
            actor1_country,
            actor2_country,
            event_date
        FROM gdelt_events
        WHERE action_geo_country = %s
          AND event_date >= %s
          AND event_date <= %s
    """

    _UPSERT_LABELS_SQL = """
        INSERT INTO country_multitask_labels (
            country, label_date,
            instability_label, war_label, terrorism_label, financial_label,
            label_version, event_count
        ) VALUES (
            %(country)s, %(label_date)s,
            %(instability_label)s, %(war_label)s,
            %(terrorism_label)s, %(financial_label)s,
            %(label_version)s, %(event_count)s
        )
        ON CONFLICT (country, label_date)
        DO UPDATE SET
            instability_label = EXCLUDED.instability_label,
            war_label         = EXCLUDED.war_label,
            terrorism_label   = EXCLUDED.terrorism_label,
            financial_label   = EXCLUDED.financial_label,
            event_count       = EXCLUDED.event_count,
            computed_at       = NOW()
    """

    _COUNTRIES_SQL = """
        SELECT DISTINCT action_geo_country
        FROM gdelt_events
        WHERE event_date = %s
          AND action_geo_country IS NOT NULL
    """

    def __init__(
        self,
        dsn: str,
        smoothing_days: int = 7,
        label_version: str = "v1",
    ):
        self._dsn = dsn
        self._smoothing = smoothing_days
        self._version = label_version

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_labels_for_date(
        self,
        target_date: date,
        countries: list[str] | None = None,
    ) -> int:
        """
        Compute and upsert labels for all countries on a given date.
        Returns rows written.
        """
        conn = psycopg2.connect(self._dsn)
        try:
            if not countries:
                with conn, conn.cursor() as cur:
                    cur.execute(self._COUNTRIES_SQL, (target_date,))
                    countries = [r[0] for r in cur.fetchall()]

            if not countries:
                return 0

            rows_written = 0
            for country in countries:
                label_row = self._compute_country_labels(conn, country, target_date)
                if label_row:
                    with conn, conn.cursor() as cur:
                        cur.execute(self._UPSERT_LABELS_SQL, label_row)
                    rows_written += 1

            logger.info(
                "Labels computed: %d countries for %s",
                rows_written, target_date,
            )
            return rows_written
        finally:
            conn.close()

    def compute_labels_range(
        self,
        start: date,
        end: date,
    ) -> dict[date, int]:
        """Compute labels for a date range. Returns {date: rows_written}."""
        results = {}
        current = start
        while current <= end:
            count = self.compute_labels_for_date(current)
            results[current] = count
            current += timedelta(days=1)
        return results

    # ------------------------------------------------------------------
    # Label computation
    # ------------------------------------------------------------------

    def _compute_country_labels(
        self,
        conn,
        country: str,
        target_date: date,
    ) -> dict[str, Any] | None:
        """Fetch events in smoothing window and compute all 4 labels."""
        window_start = target_date - timedelta(days=self._smoothing)

        with conn.cursor() as cur:
            cur.execute(
                self._FETCH_EVENTS_SQL,
                (country, window_start, target_date),
            )
            rows = cur.fetchall()

        if not rows:
            return None

        # Split into target-date events and context
        target_events = [r for r in rows if r[8] == target_date]
        context_events = [r for r in rows if r[8] < target_date]

        n_target = len(target_events)
        n_context = len(context_events) if context_events else 1  # avoid /0

        if n_target == 0:
            return None

        # Extract signals
        instability = self._instability_signal(target_events, n_target)
        war         = self._war_signal(target_events, context_events, n_target)
        terrorism   = self._terrorism_signal(target_events, n_target)
        financial   = self._financial_signal(target_events, n_target)

        # Smooth: blend with context baseline
        context_instab = self._instability_signal(context_events, n_context) if context_events else 0.0
        context_war    = self._war_signal(context_events, [], n_context) if context_events else 0.0

        alpha = 0.7  # weight on target day
        instability = alpha * instability + (1 - alpha) * context_instab
        war         = alpha * war         + (1 - alpha) * context_war

        return {
            "country":           country,
            "label_date":        target_date,
            "instability_label": round(min(instability, 1.0), 6),
            "war_label":         round(min(war,         1.0), 6),
            "terrorism_label":   round(min(terrorism,   1.0), 6),
            "financial_label":   round(min(financial,   1.0), 6),
            "label_version":     self._version,
            "event_count":       n_target,
        }

    def _instability_signal(self, events: list, n: int) -> float:
        """
        Instability = weighted fraction of conflictual events.
        Weights: quad_class=4 (material conflict) → 1.0,
                 quad_class=3 (verbal conflict)   → 0.5,
                 protest (root 14)                → 0.3
        """
        if n == 0:
            return 0.0

        score = 0.0
        for row in events:
            root, base, quad, goldstein, tone, mentions, *_ = row[:7] + (None,) * 2
            w = 1.0
            if quad == 4:
                w = 1.0
            elif quad == 3:
                w = 0.5
            elif root in PROTEST_CODES:
                w = 0.3
            else:
                continue

            # Weight by mention count (log-scaled)
            mention_w = 1.0 + np.log1p(mentions or 1) / 10.0
            score += w * mention_w

        # Normalize: sigmoid((score/n - threshold) * scale)
        raw = score / n
        return float(self._sigmoid((raw - 0.3) * 8))

    def _war_signal(
        self,
        events: list,
        context_events: list,
        n: int,
    ) -> float:
        """
        War signal: military + fight events + cross-border bilateral pairs.
        Escalation bonus when fight events occurred in prior context.
        """
        if n == 0:
            return 0.0

        fight_count = 0
        mass_violence = 0
        military_count = 0
        crossborder_count = 0
        total_mentions = 0

        for row in events:
            root, base, quad, goldstein, tone, mentions, a1c, a2c, _ = row
            mentions = mentions or 1

            if root in WAR_HIGH_CODES:
                fight_count += 1
                total_mentions += mentions
            if root == 20:
                mass_violence += 1
            if root in WAR_ROOT_CODES:
                military_count += 1
            if a1c and a2c and a1c != a2c:
                crossborder_count += 1

        # Context escalation bonus
        context_fight_ratio = sum(
            1 for r in context_events if r[0] in WAR_HIGH_CODES
        ) / max(len(context_events), 1)

        raw = (
            fight_count * 1.5
            + mass_violence * 2.0
            + military_count * 0.5
            + crossborder_count * 0.3
        ) / n

        # Escalation: if fights were already occurring, amplify
        raw *= (1.0 + context_fight_ratio * 0.5)

        # Mention intensity bonus (heavy coverage = more significant)
        if total_mentions > 0:
            intensity = np.log1p(total_mentions / max(fight_count, 1)) / 10
            raw *= (1.0 + intensity)

        return float(self._sigmoid((raw - 0.4) * 6))

    def _terrorism_signal(self, events: list, n: int) -> float:
        """
        Terrorism: asymmetric violence, bombings, kidnappings.
        Weighted heavily by mention count (terrorist acts are widely reported).
        """
        if n == 0:
            return 0.0

        terror_score = 0.0
        for row in events:
            root, base, quad, goldstein, tone, mentions, *_ = row[:7] + (None,) * 2
            mentions = mentions or 1

            if base in TERRORISM_BASE_CODES or (root in TERRORISM_CODES and quad == 4):
                # Bombings/attacks get higher weight
                weight = 2.0 if base in {183, 185} else 1.0
                terror_score += weight * (1 + np.log1p(mentions) / 8)

        raw = terror_score / n
        return float(self._sigmoid((raw - 0.3) * 7))

    def _financial_signal(self, events: list, n: int) -> float:
        """
        Financial stress: sanctions, economic coercion events.
        Proxy for economic disruption — not direct market data.
        """
        if n == 0:
            return 0.0

        sanction_count = 0
        econ_conflict_count = 0
        negative_tone_count = 0

        for row in events:
            root, base, quad, goldstein, tone, mentions, *_ = row[:7] + (None,) * 2

            if base in SANCTION_CODES or root in SANCTION_CODES:
                sanction_count += 1
            if root in ECONOMIC_CODES and quad in (3, 4):
                econ_conflict_count += 1
            if tone is not None and tone < -10:
                negative_tone_count += 1

        raw = (
            sanction_count * 1.5
            + econ_conflict_count * 0.8
            + negative_tone_count * 0.2
        ) / n

        return float(self._sigmoid((raw - 0.2) * 8))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid."""
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        ex = np.exp(x)
        return ex / (1.0 + ex)
