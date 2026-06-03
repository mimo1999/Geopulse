"""
Main GDELT ingestion pipeline.

Orchestrates: download → parse → clean → feature extract → DB insert.
Can run as a one-shot backfill or as a scheduled daily job.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import yaml

from ingestion.gdelt_downloader import GDELTDownloader, DownloadConfig
from ingestion.gdelt_parser import GDELTParser
from ingestion.event_cleaner import EventCleaner
from ingestion.db_writer import DBWriter

logger = logging.getLogger("ingestion.pipeline")


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # DB
    dsn: str = "postgresql://gldt:gldt_secret@localhost:5432/gdelt_risk"
    batch_insert_size: int = 5_000

    # Download
    raw_data_dir: Path = field(default_factory=lambda: Path("data/raw"))
    keep_raw_files: bool = False
    download_timeout: int = 120
    max_retries: int = 3
    chunk_size_bytes: int = 65_536

    # Processing
    chunk_size_rows: int = 50_000
    min_date: str = "2020-01-01"
    dedup_window_days: int = 3

    # Feature extraction toggle (Phase 1)
    run_feature_extraction: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        db = cfg.get("database", {})
        gdelt = cfg.get("gdelt", {})
        return cls(
            dsn=(
                f"postgresql://{db.get('user','gldt')}:"
                f"{db.get('password','gldt_secret')}@"
                f"{db.get('host','localhost')}:"
                f"{db.get('port',5432)}/"
                f"{db.get('name','gdelt_risk')}"
            ),
            batch_insert_size=gdelt.get("processing", {}).get("batch_insert_size", 5_000),
            raw_data_dir=Path(gdelt.get("raw_data_dir", "data/raw")),
            keep_raw_files=gdelt.get("keep_raw_files", False),
            download_timeout=gdelt.get("download", {}).get("timeout_seconds", 120),
            max_retries=gdelt.get("download", {}).get("max_retries", 3),
            chunk_size_bytes=gdelt.get("download", {}).get("chunk_size_bytes", 65_536),
            chunk_size_rows=gdelt.get("processing", {}).get("chunk_size_rows", 50_000),
            min_date=gdelt.get("processing", {}).get("min_event_date", "2020-01-01"),
        )


# ---------------------------------------------------------------------------
# Per-run result
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    target_date: date
    success: bool
    events_downloaded: int = 0
    events_parsed: int = 0
    events_inserted: int = 0
    events_skipped: int = 0
    duration_sec: float = 0.0
    error: Optional[str] = None

    def summary(self) -> str:
        return (
            f"[{self.target_date}] "
            f"{'✓' if self.success else '✗'} "
            f"parsed={self.events_parsed} "
            f"inserted={self.events_inserted} "
            f"skipped={self.events_skipped} "
            f"({self.duration_sec:.1f}s)"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class IngestionPipeline:
    """
    End-to-end GDELT ingestion pipeline.

    Example — single date::

        pipeline = IngestionPipeline(PipelineConfig())
        result = pipeline.ingest_date(date(2024, 1, 15))
        print(result.summary())

    Example — backfill::

        results = pipeline.backfill(days=30)
        for r in results:
            print(r.summary())
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.cfg = config or PipelineConfig()
        self._downloader = GDELTDownloader(
            DownloadConfig(
                timeout=self.cfg.download_timeout,
                max_retries=self.cfg.max_retries,
                chunk_size=self.cfg.chunk_size_bytes,
                raw_data_dir=self.cfg.raw_data_dir,
                keep_files=self.cfg.keep_raw_files,
            )
        )
        self._parser = GDELTParser()
        self._cleaner = EventCleaner(min_date_str=self.cfg.min_date)
        self._db = DBWriter(
            dsn=self.cfg.dsn,
            batch_size=self.cfg.batch_insert_size,
        )

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def ingest_date(self, target_date: date, skip_if_exists: bool = True) -> RunResult:
        """
        Ingest all GDELT events for a single calendar date.

        Args:
            target_date: Date to ingest.
            skip_if_exists: If True, skip dates already in the DB.
        """
        result = RunResult(target_date=target_date, success=False)
        t0 = time.monotonic()

        try:
            with self._db:
                if skip_if_exists and self._db.check_date_ingested(target_date):
                    logger.info("Date %s already ingested — skipping", target_date)
                    result.success = True
                    result.duration_sec = time.monotonic() - t0
                    return result

                total_parsed = 0
                total_inserted = 0
                total_skipped = 0

                for raw_chunk in self._downloader.stream_csv_rows(
                    target_date,
                    chunk_size_rows=self.cfg.chunk_size_rows,
                ):
                    # 1. Parse typed records
                    events, parse_skipped = self._parser.parse_chunk(raw_chunk)
                    total_skipped += parse_skipped

                    # 2. Convert to dicts for cleaning
                    row_dicts = [self._parser.to_db_dict(e) for e in events]

                    # 3. Dedup + normalize
                    clean_rows, clean_stats = self._cleaner.clean_batch(row_dicts)
                    total_skipped += clean_stats.duplicates_removed + clean_stats.invalid_removed
                    total_parsed += clean_stats.total_in - parse_skipped

                    # 4. Bulk DB insert
                    inserted = self._db.bulk_insert_events(clean_rows)
                    total_inserted += inserted

                    logger.info(
                        "Chunk: parsed=%d clean=%d inserted=%d",
                        len(raw_chunk), len(clean_rows), inserted,
                    )

                duration = time.monotonic() - t0

                # 5. Log audit record
                self._db.log_ingestion_run(
                    source_file=str(target_date),
                    events_parsed=total_parsed,
                    events_inserted=total_inserted,
                    events_skipped=total_skipped,
                    duration_sec=duration,
                    status="success",
                )

                result.success = True
                result.events_parsed = total_parsed
                result.events_inserted = total_inserted
                result.events_skipped = total_skipped
                result.duration_sec = duration

                # 6. Trigger feature extraction for this date
                if self.cfg.run_feature_extraction and total_inserted > 0:
                    self._run_feature_extraction(target_date)

        except Exception as exc:
            duration = time.monotonic() - t0
            result.duration_sec = duration
            result.error = str(exc)
            logger.error("Ingestion failed for %s: %s", target_date, exc, exc_info=True)
            try:
                with self._db:
                    self._db.log_ingestion_run(
                        source_file=str(target_date),
                        events_parsed=0, events_inserted=0, events_skipped=0,
                        duration_sec=duration, status="failed", error_message=str(exc),
                    )
            except Exception:
                pass

        logger.info(result.summary())
        return result

    def backfill(
        self,
        days: int = 30,
        end_date: Optional[date] = None,
        skip_if_exists: bool = True,
    ) -> list[RunResult]:
        """
        Ingest the last N calendar days in reverse-chronological order.

        Args:
            days: Number of days to backfill.
            end_date: Last date to include (default: yesterday).
            skip_if_exists: Skip dates already in DB.
        """
        if end_date is None:
            end_date = date.today() - timedelta(days=1)

        start_date = end_date - timedelta(days=days - 1)
        dates = [
            start_date + timedelta(days=i)
            for i in range(days)
        ]
        # Process newest first (most valuable data first)
        dates.sort(reverse=True)

        logger.info(
            "Starting backfill: %s → %s (%d days)",
            start_date, end_date, len(dates),
        )

        results = []
        for i, d in enumerate(dates, 1):
            logger.info("Backfill progress: %d/%d — %s", i, len(dates), d)
            result = self.ingest_date(d, skip_if_exists=skip_if_exists)
            results.append(result)
            # Small pause between downloads to be respectful
            if i < len(dates):
                time.sleep(0.5)

        successful = sum(1 for r in results if r.success)
        total_inserted = sum(r.events_inserted for r in results)
        logger.info(
            "Backfill complete: %d/%d dates OK, %d events inserted",
            successful, len(dates), total_inserted,
        )
        return results

    def ingest_latest(self) -> RunResult:
        """Ingest yesterday's data (suitable for daily cron job)."""
        yesterday = date.today() - timedelta(days=1)
        return self.ingest_date(yesterday)

    def _run_feature_extraction(self, target_date: date) -> None:
        """Trigger country feature aggregation for a date."""
        try:
            from preprocessing.feature_extractor import FeatureExtractor
            extractor = FeatureExtractor(dsn=self.cfg.dsn)
            extractor.compute_daily_features(target_date)
            logger.info("Feature extraction complete for %s", target_date)
        except ImportError:
            logger.debug("preprocessing module not yet available — skipping features")
        except Exception as exc:
            logger.warning("Feature extraction failed for %s: %s", target_date, exc)

    def close(self) -> None:
        self._downloader.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
