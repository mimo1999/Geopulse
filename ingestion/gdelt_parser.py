"""
GDELT event parser.

Converts raw dict rows from GDELTDownloader into clean,
typed records ready for database insertion and feature extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

logger = logging.getLogger("ingestion.parser")


# ---------------------------------------------------------------------------
# Typed event record
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class GDELTEvent:
    global_event_id: int
    event_date: date
    actor1_code: Optional[str]
    actor1_name: Optional[str]
    actor1_country: Optional[str]
    actor1_type1: Optional[str]
    actor2_code: Optional[str]
    actor2_name: Optional[str]
    actor2_country: Optional[str]
    actor2_type1: Optional[str]
    event_code: Optional[int]
    event_base_code: Optional[int]
    event_root_code: Optional[int]
    quad_class: Optional[int]
    goldstein: Optional[float]
    num_mentions: int
    num_sources: int
    num_articles: int
    avg_tone: Optional[float]
    action_geo_country: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    source_url: Optional[str]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class GDELTParser:
    """
    Parses raw GDELT 1.0 tab-separated rows into typed GDELTEvent objects.

    Handles:
    - Type coercion with safe fallbacks
    - Date parsing (YYYYMMDD → date)
    - Empty-string → None normalization
    - Basic validity filtering
    """

    MIN_EVENT_ID = 1
    VALID_QUAD_CLASSES = {1, 2, 3, 4}
    GOLDSTEIN_RANGE = (-10.0, 10.0)

    def parse_chunk(
        self,
        raw_rows: list[dict[str, str]],
    ) -> tuple[list[GDELTEvent], int]:
        """
        Parse a chunk of raw row dicts.

        Returns:
            (valid_events, skipped_count)
        """
        events: list[GDELTEvent] = []
        skipped = 0

        for row in raw_rows:
            event = self._parse_row(row)
            if event is not None:
                events.append(event)
            else:
                skipped += 1

        return events, skipped

    def _parse_row(self, row: dict[str, str]) -> Optional[GDELTEvent]:
        try:
            global_event_id = self._int(row.get("global_event_id"))
            if global_event_id is None or global_event_id < self.MIN_EVENT_ID:
                return None

            raw_date = row.get("sqldate", "").strip()
            if len(raw_date) != 8:
                return None
            event_date = date(int(raw_date[:4]), int(raw_date[4:6]), int(raw_date[6:8]))

            quad_class = self._int(row.get("quad_class"))
            goldstein = self._float(row.get("goldstein_scale"))
            avg_tone = self._float(row.get("avg_tone"))

            # Clamp goldstein to valid range
            if goldstein is not None:
                goldstein = max(self.GOLDSTEIN_RANGE[0],
                                min(self.GOLDSTEIN_RANGE[1], goldstein))

            return GDELTEvent(
                global_event_id=global_event_id,
                event_date=event_date,
                actor1_code=self._str(row.get("actor1_code")),
                actor1_name=self._str(row.get("actor1_name")),
                actor1_country=self._str(row.get("actor1_country_code")),
                actor1_type1=self._str(row.get("actor1_type1_code")),
                actor2_code=self._str(row.get("actor2_code")),
                actor2_name=self._str(row.get("actor2_name")),
                actor2_country=self._str(row.get("actor2_country_code")),
                actor2_type1=self._str(row.get("actor2_type1_code")),
                event_code=self._int(row.get("event_code")),
                event_base_code=self._int(row.get("event_base_code")),
                event_root_code=self._int(row.get("event_root_code")),
                quad_class=quad_class,
                goldstein=goldstein,
                num_mentions=self._int(row.get("num_mentions")) or 0,
                num_sources=self._int(row.get("num_sources")) or 0,
                num_articles=self._int(row.get("num_articles")) or 0,
                avg_tone=avg_tone,
                action_geo_country=self._str(row.get("action_geo_country_code")),
                latitude=self._float(row.get("action_geo_lat")),
                longitude=self._float(row.get("action_geo_long")),
                source_url=self._str(row.get("source_url")),
            )

        except (ValueError, KeyError, TypeError) as exc:
            logger.debug("Row parse error: %s | row keys: %s", exc, list(row.keys())[:5])
            return None

    # ------------------------------------------------------------------
    # Safe type helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _str(val: Any) -> Optional[str]:
        if val is None:
            return None
        s = str(val).strip()
        return s if s else None

    @staticmethod
    def _int(val: Any) -> Optional[int]:
        if val is None:
            return None
        s = str(val).strip()
        if not s:
            return None
        try:
            return int(float(s))  # handles "14.0"
        except ValueError:
            return None

    @staticmethod
    def _float(val: Any) -> Optional[float]:
        if val is None:
            return None
        s = str(val).strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def to_db_dict(self, event: GDELTEvent) -> dict[str, Any]:
        """Convert a GDELTEvent to a dict suitable for bulk DB insert."""
        return {
            "global_event_id": event.global_event_id,
            "event_date": event.event_date,
            "actor1_code": event.actor1_code,
            "actor1_name": event.actor1_name,
            "actor1_country": event.actor1_country,
            "actor1_type1": event.actor1_type1,
            "actor2_code": event.actor2_code,
            "actor2_name": event.actor2_name,
            "actor2_country": event.actor2_country,
            "actor2_type1": event.actor2_type1,
            "event_code": event.event_code,
            "event_base_code": event.event_base_code,
            "event_root_code": event.event_root_code,
            "quad_class": event.quad_class,
            "goldstein": event.goldstein,
            "num_mentions": event.num_mentions,
            "num_sources": event.num_sources,
            "num_articles": event.num_articles,
            "avg_tone": event.avg_tone,
            "action_geo_country": event.action_geo_country,
            "latitude": event.latitude,
            "longitude": event.longitude,
            "source_url": event.source_url,
        }
