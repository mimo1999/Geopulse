"""
GDELT 2.0 ingestion — 15-minute update files.

GDELT 2.0 differences from v1:
  - Files published every 15 minutes
  - 61 columns (3 extra: EventTimeDate, MentionType, MentionSourceName removed
    from separate mentions file; export still tab-separated)
  - Master file list at:
    http://data.gdeltproject.org/gdeltv2/masterfilelist.txt
  - Last update list (most recent files only):
    http://data.gdeltproject.org/gdeltv2/lastupdate.txt

Usage:
    processor = GDELTV2Processor(dsn, raw_data_dir="data/raw/v2")
    processor.run_catchup()                  # process all pending files
    processor.process_latest()               # latest 15-min file only
    processor.process_url(url)               # specific file

Cursor state persisted to gdelt_v2_cursor table to enable
exactly-once processing across restarts.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

import psycopg2
import psycopg2.extras
import requests

from ingestion.gdelt_parser import GDELTParser
from ingestion.event_cleaner import EventCleaner
from ingestion.db_writer import DBWriter

logger = logging.getLogger("ingestion.gdelt_v2")

MASTER_LIST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
LAST_UPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

# GDELT v2 export has the same 58 columns as v1 in the .export.CSV files
# (The separate .mentions.CSV and .gkg.CSV are additional files we skip for now)


@dataclass
class V2ProcessResult:
    url: str
    status: str               # 'success' | 'skipped' | 'error'
    events_inserted: int = 0
    duration_sec: float = 0.0
    error: Optional[str] = None


class GDELTV2Processor:
    """
    Processes GDELT 2.0 15-minute export files.

    State machine:
      1. Fetch master list → filter to .export.CSV.zip files
      2. Mark new files as 'pending' in gdelt_v2_cursor
      3. For each pending file: download → parse → clean → insert → mark 'done'
      4. On error: mark 'error', log, continue to next file
    """

    _INSERT_CURSOR_SQL = """
        INSERT INTO gdelt_v2_cursor (file_url, file_timestamp)
        VALUES (%s, %s)
        ON CONFLICT (file_url) DO NOTHING
    """

    _PENDING_SQL = """
        SELECT file_url, file_timestamp
        FROM gdelt_v2_cursor
        WHERE status = 'pending'
        ORDER BY file_timestamp ASC
        LIMIT %s
    """

    _MARK_PROCESSING_SQL = """
        UPDATE gdelt_v2_cursor SET status = 'processing' WHERE file_url = %s
    """

    _MARK_DONE_SQL = """
        UPDATE gdelt_v2_cursor
        SET status = 'done', events_inserted = %s, processed_at = NOW()
        WHERE file_url = %s
    """

    _MARK_ERROR_SQL = """
        UPDATE gdelt_v2_cursor
        SET status = 'error', error_message = %s, processed_at = NOW()
        WHERE file_url = %s
    """

    def __init__(
        self,
        dsn: str,
        raw_data_dir: str = "data/raw/v2",
        chunk_size_rows: int = 50_000,
        batch_insert_size: int = 5_000,
        timeout: int = 90,
        max_retries: int = 3,
    ):
        self._dsn = dsn
        self._raw_dir = Path(raw_data_dir)
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._chunk_size = chunk_size_rows
        self._timeout = timeout
        self._max_retries = max_retries

        self._parser  = GDELTParser()
        self._cleaner = EventCleaner()
        self._db      = DBWriter(dsn=dsn, batch_size=batch_insert_size)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "GLDT-Research/1.0"})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_latest(self) -> list[V2ProcessResult]:
        """
        Process only the most recent files listed in lastupdate.txt.
        Suitable for a cron job running every 15 minutes.
        """
        urls = self._fetch_last_update_urls()
        return [self.process_url(url) for url in urls]

    def run_catchup(self, batch_size: int = 50) -> list[V2ProcessResult]:
        """
        Register all master list files as pending, then process
        the oldest `batch_size` pending files.
        """
        logger.info("Fetching GDELT v2 master list for catch-up...")
        all_urls = self._fetch_master_list()
        self._register_urls(all_urls)

        conn = psycopg2.connect(self._dsn)
        try:
            with conn, conn.cursor() as cur:
                cur.execute(self._PENDING_SQL, (batch_size,))
                pending = [(r[0], r[1]) for r in cur.fetchall()]
        finally:
            conn.close()

        logger.info("Processing %d pending v2 files", len(pending))
        results = []
        for url, _ in pending:
            result = self.process_url(url)
            results.append(result)
            time.sleep(0.2)   # small delay between files

        return results

    def process_url(self, url: str) -> V2ProcessResult:
        """Download, parse, and ingest a single GDELT v2 export file."""
        t0 = time.monotonic()
        self._mark_processing(url)

        try:
            raw_bytes = self._download(url)
            if raw_bytes is None:
                raise RuntimeError(f"Download failed after retries: {url}")

            total_inserted = 0
            for raw_chunk in self._stream_zip(raw_bytes):
                events, _ = self._parser.parse_chunk(raw_chunk)
                row_dicts  = [self._parser.to_db_dict(e) for e in events]
                clean_rows, _ = self._cleaner.clean_batch(row_dicts)
                with self._db:
                    inserted = self._db.bulk_insert_events(clean_rows)
                total_inserted += inserted

            duration = time.monotonic() - t0
            self._mark_done(url, total_inserted)
            logger.info(
                "v2 file done: %d events in %.1fs — %s",
                total_inserted, duration, url.split("/")[-1],
            )
            return V2ProcessResult(
                url=url, status="success",
                events_inserted=total_inserted, duration_sec=duration,
            )

        except Exception as exc:
            duration = time.monotonic() - t0
            self._mark_error(url, str(exc))
            logger.error("v2 processing failed: %s — %s", url.split("/")[-1], exc)
            return V2ProcessResult(
                url=url, status="error", duration_sec=duration, error=str(exc),
            )

    # ------------------------------------------------------------------
    # Cursor management
    # ------------------------------------------------------------------

    def _register_urls(self, urls: list[str]) -> int:
        """Insert new URLs into the cursor table. Returns count registered."""
        if not urls:
            return 0

        rows = []
        for url in urls:
            ts = self._parse_url_timestamp(url)
            rows.append((url, ts))

        conn = psycopg2.connect(self._dsn)
        try:
            with conn, conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, self._INSERT_CURSOR_SQL, rows)
            logger.info("Registered %d v2 URLs in cursor table", len(rows))
            return len(rows)
        finally:
            conn.close()

    def _mark_processing(self, url: str) -> None:
        # Ensure the URL is registered before marking it as processing.
        # This handles the case where process_url / process_latest is called
        # directly without a prior run_catchup / _register_urls call.
        ts = self._parse_url_timestamp(url)
        conn = psycopg2.connect(self._dsn)
        with conn, conn.cursor() as cur:
            cur.execute(self._INSERT_CURSOR_SQL, (url, ts))
            cur.execute(self._MARK_PROCESSING_SQL, (url,))
        conn.close()

    def _mark_done(self, url: str, count: int) -> None:
        conn = psycopg2.connect(self._dsn)
        with conn, conn.cursor() as cur:
            cur.execute(self._MARK_DONE_SQL, (count, url))
        conn.close()

    def _mark_error(self, url: str, message: str) -> None:
        conn = psycopg2.connect(self._dsn)
        with conn, conn.cursor() as cur:
            cur.execute(self._MARK_ERROR_SQL, (message[:500], url))
        conn.close()

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def _fetch_last_update_urls(self) -> list[str]:
        """Fetch 3 URLs from lastupdate.txt (export only, not mentions/gkg)."""
        try:
            resp = self._session.get(LAST_UPDATE_URL, timeout=30)
            resp.raise_for_status()
            return [
                line.split()[-1]
                for line in resp.text.splitlines()
                if line.strip() and ".export.CSV.zip" in line
            ]
        except Exception as exc:
            logger.error("Failed to fetch lastupdate.txt: %s", exc)
            return []

    def _fetch_master_list(self) -> list[str]:
        """Fetch all export URLs from master list."""
        try:
            resp = self._session.get(MASTER_LIST_URL, timeout=120, stream=True)
            resp.raise_for_status()
            return [
                line.split()[-1]
                for line in resp.text.splitlines()
                if line.strip() and ".export.CSV.zip" in line
            ]
        except Exception as exc:
            logger.error("Failed to fetch master list: %s", exc)
            return []

    def _download(self, url: str) -> Optional[bytes]:
        """Download with retry."""
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._session.get(url, timeout=self._timeout, stream=True)
                resp.raise_for_status()
                chunks = []
                for chunk in resp.iter_content(65536):
                    if chunk:
                        chunks.append(chunk)
                return b"".join(chunks)
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    return None
                logger.warning("HTTP error attempt %d: %s", attempt, exc)
            except Exception as exc:
                logger.warning("Download error attempt %d: %s", attempt, exc)
            if attempt < self._max_retries:
                time.sleep(2 ** attempt)
        return None

    def _stream_zip(self, raw: bytes) -> Generator[list[dict], None, None]:
        """Parse zip → CSV → yield row chunks."""
        import csv
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                csv_name = next(
                    (n for n in zf.namelist() if ".export.CSV" in n.upper()),
                    None,
                )
                if not csv_name:
                    return
                # GDELT 2.0 export has 61 tab-separated columns (no header).
                # Using the correct v2 column list ensures action_geo_country_code
                # and other geo fields are mapped to the right positions.
                from ingestion.gdelt_downloader import GDELT_V2_COLUMNS
                with zf.open(csv_name) as f:
                    reader = csv.reader(
                        io.TextIOWrapper(f, encoding="latin-1"),
                        delimiter="\t",
                    )
                    batch: list[dict] = []
                    for row in reader:
                        # Accept rows with 58 cols (v1 legacy) or 61 cols (v2)
                        if len(row) < 58:
                            continue
                        record = dict(zip(GDELT_V2_COLUMNS, row))
                        batch.append(record)
                        if len(batch) >= self._chunk_size:
                            yield batch
                            batch = []
                    if batch:
                        yield batch
        except zipfile.BadZipFile as exc:
            logger.error("Bad ZIP: %s", exc)

    @staticmethod
    def _parse_url_timestamp(url: str) -> datetime:
        """
        Extract timestamp from v2 filename.
        Pattern: 20241015123000.export.CSV.zip
        """
        filename = url.split("/")[-1]
        ts_str = filename[:14]   # YYYYMMDDHHmmss
        try:
            return datetime(
                int(ts_str[:4]), int(ts_str[4:6]), int(ts_str[6:8]),
                int(ts_str[8:10]), int(ts_str[10:12]), int(ts_str[12:14]),
                tzinfo=timezone.utc,
            )
        except Exception:
            return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Scheduler integration (15-min cron)
# ---------------------------------------------------------------------------

def schedule_v2_ingestion(dsn: str) -> None:
    """
    Start APScheduler with a 15-minute GDELT v2 job.
    Called from ingestion/scheduler.py.
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.interval import IntervalTrigger
    except ImportError:
        logger.error("APScheduler not installed")
        return

    processor = GDELTV2Processor(dsn=dsn)

    def job():
        logger.info("GDELT v2 15-min job triggered")
        results = processor.process_latest()
        total = sum(r.events_inserted for r in results if r.status == "success")
        logger.info("v2 job done: %d events inserted", total)

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        job,
        trigger=IntervalTrigger(minutes=15),
        id="gdelt_v2_15min",
        misfire_grace_time=300,
        coalesce=True,
    )
    logger.info("GDELT v2 scheduler starting — interval: 15 minutes")
    scheduler.start()
