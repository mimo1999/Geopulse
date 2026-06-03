"""
GDELT file downloader with streaming support.

Handles both GDELT 1.0 (daily .export.CSV.zip) and
GDELT 2.0 (15-minute update files from the master list).
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Generator, Iterator, Optional
from urllib.parse import urljoin

import requests

logger = logging.getLogger("ingestion.downloader")


# ---------------------------------------------------------------------------
# GDELT 1.0 column names (all 58 columns)
# ---------------------------------------------------------------------------
GDELT_V1_COLUMNS = [
    "global_event_id", "sqldate", "month_year", "year", "fraction_date",
    "actor1_code", "actor1_name", "actor1_country_code", "actor1_known_group_code",
    "actor1_ethnic_code", "actor1_religion1_code", "actor1_religion2_code",
    "actor1_type1_code", "actor1_type2_code", "actor1_type3_code",
    "actor2_code", "actor2_name", "actor2_country_code", "actor2_known_group_code",
    "actor2_ethnic_code", "actor2_religion1_code", "actor2_religion2_code",
    "actor2_type1_code", "actor2_type2_code", "actor2_type3_code",
    "is_root_event", "event_code", "event_base_code", "event_root_code",
    "quad_class", "goldstein_scale", "num_mentions", "num_sources",
    "num_articles", "avg_tone",
    "actor1_geo_type", "actor1_geo_fullname", "actor1_geo_country_code",
    "actor1_geo_adm1_code", "actor1_geo_lat", "actor1_geo_long", "actor1_geo_feature_id",
    "actor2_geo_type", "actor2_geo_fullname", "actor2_geo_country_code",
    "actor2_geo_adm1_code", "actor2_geo_lat", "actor2_geo_long", "actor2_geo_feature_id",
    "action_geo_type", "action_geo_fullname", "action_geo_country_code",
    "action_geo_adm1_code", "action_geo_lat", "action_geo_long", "action_geo_feature_id",
    "date_added", "source_url",
]

# ---------------------------------------------------------------------------
# GDELT 2.0 column names (61 columns)
#
# GDELT 2.0 export adds an ADM2 (second-level administrative) code field
# inside each of the three geo blocks (Actor1, Actor2, Action), placing it
# right after the ADM1 field.  All other columns keep the same positions as
# v1 up to the first geo block.  The parser must use these names when reading
# 15-minute .export.CSV.zip files from gdeltv2/.
# ---------------------------------------------------------------------------
GDELT_V2_COLUMNS = [
    # --- Global identifiers (same as v1, cols 0-4) ---
    "global_event_id", "sqldate", "month_year", "year", "fraction_date",
    # --- Actor 1 (same as v1, cols 5-14) ---
    "actor1_code", "actor1_name", "actor1_country_code", "actor1_known_group_code",
    "actor1_ethnic_code", "actor1_religion1_code", "actor1_religion2_code",
    "actor1_type1_code", "actor1_type2_code", "actor1_type3_code",
    # --- Actor 2 (same as v1, cols 15-24) ---
    "actor2_code", "actor2_name", "actor2_country_code", "actor2_known_group_code",
    "actor2_ethnic_code", "actor2_religion1_code", "actor2_religion2_code",
    "actor2_type1_code", "actor2_type2_code", "actor2_type3_code",
    # --- Event metadata (same as v1, cols 25-34) ---
    "is_root_event", "event_code", "event_base_code", "event_root_code",
    "quad_class", "goldstein_scale", "num_mentions", "num_sources",
    "num_articles", "avg_tone",
    # --- Actor1 geo (v2 adds adm2_code, cols 35-43) ---
    "actor1_geo_type", "actor1_geo_fullname", "actor1_geo_country_code",
    "actor1_geo_adm1_code", "actor1_geo_adm2_code",   # adm2 is NEW in v2
    "actor1_geo_lat", "actor1_geo_long", "actor1_geo_feature_id",
    # --- Actor2 geo (v2 adds adm2_code, cols 43-51) ---
    "actor2_geo_type", "actor2_geo_fullname", "actor2_geo_country_code",
    "actor2_geo_adm1_code", "actor2_geo_adm2_code",   # adm2 is NEW in v2
    "actor2_geo_lat", "actor2_geo_long", "actor2_geo_feature_id",
    # --- Action geo (v2 adds adm2_code, cols 51-59) ---
    "action_geo_type", "action_geo_fullname", "action_geo_country_code",
    "action_geo_adm1_code", "action_geo_adm2_code",   # adm2 is NEW in v2
    "action_geo_lat", "action_geo_long", "action_geo_feature_id",
    # --- Provenance (cols 59-60) ---
    "date_added", "source_url",
]

assert len(GDELT_V2_COLUMNS) == 61, f"Expected 61 v2 columns, got {len(GDELT_V2_COLUMNS)}"

GDELT_V1_BASE_URL = "http://data.gdeltproject.org/events"
GDELT_V2_MASTER_LIST = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"


@dataclass
class DownloadResult:
    url: str
    success: bool
    local_path: Optional[Path] = None
    bytes_downloaded: int = 0
    duration_sec: float = 0.0
    error: Optional[str] = None


@dataclass
class DownloadConfig:
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 5.0
    chunk_size: int = 65_536      # 64 KB
    raw_data_dir: Path = field(default_factory=lambda: Path("data/raw"))
    keep_files: bool = False


class GDELTDownloader:
    """
    Downloads GDELT files with retry logic and streaming.

    Supports:
    - GDELT 1.0: one file per calendar day
    - GDELT 2.0: 15-minute update files via master list
    """

    def __init__(self, config: DownloadConfig | None = None):
        self.cfg = config or DownloadConfig()
        self.cfg.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "GLDT-Research/1.0"})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream_csv_rows(
        self,
        target_date: date,
        chunk_size_rows: int = 50_000,
    ) -> Generator[list[dict], None, None]:
        """
        Download a GDELT 1.0 daily file and yield it in row-chunks
        without writing to disk (in-memory streaming).

        Args:
            target_date: The calendar date to fetch.
            chunk_size_rows: Number of rows per yielded chunk.

        Yields:
            List of raw row dicts (column names from GDELT_V1_COLUMNS).
        """
        url = self._v1_url(target_date)
        logger.info("Streaming GDELT v1 for %s → %s", target_date, url)

        raw_bytes = self._download_with_retry(url)
        if raw_bytes is None:
            logger.error("Failed to download %s after retries", url)
            return

        yield from self._parse_zip_stream(raw_bytes, chunk_size_rows)

    def download_date_range(
        self,
        start: date,
        end: date,
    ) -> Iterator[tuple[date, DownloadResult]]:
        """
        Download GDELT 1.0 files for a range of dates.
        Files are saved to raw_data_dir.

        Yields: (date, DownloadResult)
        """
        current = start
        while current <= end:
            result = self._download_v1_file(current)
            yield current, result
            current += timedelta(days=1)

    def fetch_v2_master_list(self) -> list[str]:
        """
        Fetch the GDELT 2.0 master file list and return all export URLs.
        """
        logger.info("Fetching GDELT 2.0 master list")
        try:
            resp = self._session.get(
                GDELT_V2_MASTER_LIST,
                timeout=self.cfg.timeout,
                stream=True,
            )
            resp.raise_for_status()
            lines = resp.text.splitlines()
            # Format: "bytes hash url"
            urls = [
                line.split()[-1]
                for line in lines
                if line.strip() and line.split()[-1].endswith(".export.CSV.zip")
            ]
            logger.info("Master list: %d export files found", len(urls))
            return urls
        except Exception as exc:
            logger.error("Failed to fetch master list: %s", exc)
            return []

    def stream_v2_url(
        self,
        url: str,
        chunk_size_rows: int = 50_000,
    ) -> Generator[list[dict], None, None]:
        """Stream a GDELT 2.0 export file by URL."""
        raw_bytes = self._download_with_retry(url)
        if raw_bytes is None:
            return
        yield from self._parse_zip_stream(raw_bytes, chunk_size_rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _v1_url(self, d: date) -> str:
        filename = d.strftime("%Y%m%d") + ".export.CSV.zip"
        return f"{GDELT_V1_BASE_URL}/{filename}"

    def _download_v1_file(self, d: date) -> DownloadResult:
        url = self._v1_url(d)
        dest = self.cfg.raw_data_dir / f"{d.strftime('%Y%m%d')}.export.CSV.zip"

        if dest.exists():
            logger.debug("Already downloaded: %s", dest)
            return DownloadResult(url=url, success=True, local_path=dest)

        start = time.monotonic()
        raw_bytes = self._download_with_retry(url)
        duration = time.monotonic() - start

        if raw_bytes is None:
            return DownloadResult(url=url, success=False, duration_sec=duration,
                                  error="Max retries exceeded")

        dest.write_bytes(raw_bytes)
        logger.info("Saved %s (%.1f KB)", dest.name, len(raw_bytes) / 1024)
        return DownloadResult(
            url=url,
            success=True,
            local_path=dest,
            bytes_downloaded=len(raw_bytes),
            duration_sec=duration,
        )

    def _download_with_retry(self, url: str) -> bytes | None:
        """Download URL with exponential backoff. Returns raw bytes or None."""
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                logger.debug("GET %s (attempt %d)", url, attempt)
                resp = self._session.get(
                    url,
                    timeout=self.cfg.timeout,
                    stream=True,
                )
                resp.raise_for_status()

                chunks = []
                for chunk in resp.iter_content(chunk_size=self.cfg.chunk_size):
                    if chunk:
                        chunks.append(chunk)
                return b"".join(chunks)

            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    logger.warning("File not found (404): %s", url)
                    return None
                logger.warning("HTTP error on attempt %d: %s", attempt, exc)
            except requests.exceptions.RequestException as exc:
                logger.warning("Request error on attempt %d: %s", attempt, exc)

            if attempt < self.cfg.max_retries:
                sleep_time = self.cfg.retry_delay * (2 ** (attempt - 1))
                logger.debug("Retrying in %.1fs", sleep_time)
                time.sleep(sleep_time)

        return None

    def _parse_zip_stream(
        self,
        raw_bytes: bytes,
        chunk_size_rows: int,
    ) -> Generator[list[dict], None, None]:
        """
        Extract the CSV from a ZIP in memory and yield row chunks
        without requiring pandas — pure stdlib for minimal memory footprint.
        Caller can pass rows to pandas for bulk processing.
        """
        import csv

        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                csv_name = next(
                    (n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")),
                    None,
                )
                if csv_name is None:
                    logger.error("No CSV found in ZIP")
                    return

                with zf.open(csv_name) as csv_file:
                    reader = csv.reader(
                        io.TextIOWrapper(csv_file, encoding="latin-1"),
                        delimiter="\t",
                    )
                    batch: list[dict] = []
                    for row in reader:
                        if len(row) < len(GDELT_V1_COLUMNS):
                            continue  # malformed row
                        record = dict(zip(GDELT_V1_COLUMNS, row))
                        batch.append(record)
                        if len(batch) >= chunk_size_rows:
                            yield batch
                            batch = []
                    if batch:
                        yield batch

        except zipfile.BadZipFile as exc:
            logger.error("Bad ZIP file: %s", exc)
        except Exception as exc:
            logger.error("Unexpected parse error: %s", exc, exc_info=True)

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
