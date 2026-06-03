"""
PostgreSQL bulk writer for GDELT events.

Uses psycopg2 execute_values for high-throughput batch inserts.
ON CONFLICT DO NOTHING handles re-ingestion gracefully.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

import psycopg2
import psycopg2.extras
from psycopg2 import sql

logger = logging.getLogger("ingestion.db_writer")


class DBWriter:
    """
    Handles all database write operations for the ingestion pipeline.
    """

    INSERT_EVENTS_SQL = """
        INSERT INTO gdelt_events (
            global_event_id, event_date, actor1_code, actor1_name,
            actor1_country, actor1_type1, actor2_code, actor2_name,
            actor2_country, actor2_type1, event_code, event_base_code,
            event_root_code, quad_class, goldstein, num_mentions,
            num_sources, num_articles, avg_tone, action_geo_country,
            latitude, longitude, source_url
        ) VALUES %s
        ON CONFLICT (global_event_id, event_date) DO NOTHING
    """

    INSERT_RUN_SQL = """
        INSERT INTO ingestion_runs
            (source_file, events_parsed, events_inserted, events_skipped,
             duration_sec, status, error_message)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """

    def __init__(self, dsn: str, batch_size: int = 5_000):
        """
        Args:
            dsn: PostgreSQL connection string, e.g.
                 "postgresql://user:pass@host:5432/dbname"
            batch_size: rows per executemany call
        """
        self._dsn = dsn
        self._batch_size = batch_size
        self._conn: psycopg2.extensions.connection | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._conn = psycopg2.connect(self._dsn)
        self._conn.autocommit = False
        logger.info("Connected to PostgreSQL")

    def disconnect(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.info("Disconnected from PostgreSQL")

    @contextmanager
    def transaction(self) -> Generator[psycopg2.extensions.cursor, None, None]:
        if self._conn is None or self._conn.closed:
            self.connect()
        cursor = self._conn.cursor()  # type: ignore[union-attr]
        try:
            yield cursor
            self._conn.commit()  # type: ignore[union-attr]
        except Exception:
            self._conn.rollback()  # type: ignore[union-attr]
            raise
        finally:
            cursor.close()

    # ------------------------------------------------------------------
    # Public write methods
    # ------------------------------------------------------------------

    def bulk_insert_events(
        self,
        rows: list[dict[str, Any]],
    ) -> int:
        """
        Bulk-insert a list of event dicts.
        Returns number of rows actually inserted (after ON CONFLICT skip).
        """
        if not rows:
            return 0

        # Split into batches
        inserted = 0
        for i in range(0, len(rows), self._batch_size):
            batch = rows[i : i + self._batch_size]
            inserted += self._insert_batch(batch)

        return inserted

    def _insert_batch(self, batch: list[dict[str, Any]]) -> int:
        """Insert one batch, return row count inserted."""
        tuples = [
            (
                r["global_event_id"],
                r["event_date"],
                r.get("actor1_code"),
                r.get("actor1_name"),
                r.get("actor1_country"),
                r.get("actor1_type1"),
                r.get("actor2_code"),
                r.get("actor2_name"),
                r.get("actor2_country"),
                r.get("actor2_type1"),
                r.get("event_code"),
                r.get("event_base_code"),
                r.get("event_root_code"),
                r.get("quad_class"),
                r.get("goldstein"),
                r.get("num_mentions", 0),
                r.get("num_sources", 0),
                r.get("num_articles", 0),
                r.get("avg_tone"),
                r.get("action_geo_country"),
                r.get("latitude"),
                r.get("longitude"),
                r.get("source_url"),
            )
            for r in batch
        ]

        t0 = time.monotonic()
        with self.transaction() as cur:
            psycopg2.extras.execute_values(
                cur,
                self.INSERT_EVENTS_SQL,
                tuples,
                template=None,
                page_size=self._batch_size,
            )
            inserted = cur.rowcount

        elapsed = time.monotonic() - t0
        rate = len(batch) / max(elapsed, 0.001)
        logger.debug(
            "Inserted %d/%d rows in %.2fs (%.0f rows/sec)",
            inserted, len(batch), elapsed, rate,
        )
        return max(inserted, 0)

    def log_ingestion_run(
        self,
        source_file: str,
        events_parsed: int,
        events_inserted: int,
        events_skipped: int,
        duration_sec: float,
        status: str,
        error_message: str | None = None,
    ) -> int:
        """Insert an audit record and return its ID."""
        with self.transaction() as cur:
            cur.execute(
                self.INSERT_RUN_SQL,
                (
                    source_file,
                    events_parsed,
                    events_inserted,
                    events_skipped,
                    duration_sec,
                    status,
                    error_message,
                ),
            )
            row = cur.fetchone()
            return row[0] if row else -1

    def check_date_ingested(self, d) -> bool:
        """True if events for a given date already exist in the DB."""
        with self.transaction() as cur:
            cur.execute(
                "SELECT 1 FROM gdelt_events WHERE event_date = %s LIMIT 1",
                (d,),
            )
            return cur.fetchone() is not None

    def get_event_count_for_date(self, d) -> int:
        """Return count of events for a given date."""
        with self.transaction() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM gdelt_events WHERE event_date = %s",
                (d,),
            )
            row = cur.fetchone()
            return row[0] if row else 0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()
