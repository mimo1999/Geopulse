"""
Event deduplication and cleaning layer.

Strategies:
1. In-memory Bloom-filter for same-batch dedup (fast).
2. DB-side ON CONFLICT DO NOTHING for cross-batch dedup.
3. Optional look-back window query for re-ingestion protection.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger("ingestion.cleaner")


# ---------------------------------------------------------------------------
# Minimal Bloom Filter (no extra deps)
# ---------------------------------------------------------------------------

class BloomFilter:
    """
    Simple Bloom filter for fast in-memory seen-ID tracking.
    False-positive rate ≈ 0.1% for expected_items.
    """

    def __init__(self, expected_items: int = 1_000_000, fp_rate: float = 0.001):
        # Calculate bit array size
        m = -(expected_items * math.log(fp_rate)) / (math.log(2) ** 2)
        self._size = int(m)
        self._bits = bytearray(math.ceil(self._size / 8))
        # Number of hash functions
        self._k = max(1, int((self._size / expected_items) * math.log(2)))

    def _hashes(self, item: int) -> list[int]:
        raw = item.to_bytes(8, "little", signed=False)
        h1 = int(hashlib.md5(raw).hexdigest(), 16)
        h2 = int(hashlib.sha1(raw).hexdigest(), 16)
        return [(h1 + i * h2) % self._size for i in range(self._k)]

    def add(self, item: int) -> None:
        for pos in self._hashes(item):
            self._bits[pos >> 3] |= 1 << (pos & 7)

    def __contains__(self, item: int) -> bool:
        return all(
            (self._bits[pos >> 3] >> (pos & 7)) & 1
            for pos in self._hashes(item)
        )

    def __len__(self) -> int:
        return sum(bin(b).count("1") for b in self._bits)


# ---------------------------------------------------------------------------
# Deduplicator
# ---------------------------------------------------------------------------

@dataclass
class DeduplicationStats:
    total_in: int = 0
    duplicates_removed: int = 0
    invalid_removed: int = 0

    @property
    def total_out(self) -> int:
        return self.total_in - self.duplicates_removed - self.invalid_removed

    @property
    def dedup_rate(self) -> float:
        return self.duplicates_removed / max(1, self.total_in)


class EventCleaner:
    """
    Cleans and deduplicates GDELT event records before DB insertion.

    Usage pattern:
        cleaner = EventCleaner()
        for chunk in ...:
            clean_rows, stats = cleaner.clean_batch(chunk)
            db.bulk_insert(clean_rows)
    """

    # Countries to map from GDELT 2-letter codes → ISO 3166-1 alpha-2
    # (GDELT uses FIPS-10 for some countries — normalize common ones)
    FIPS_TO_ISO: dict[str, str] = {
        "RS": "RU",   # Russia (GDELT uses RS)
        "CH": "CN",   # China
        "GM": "DE",   # Germany
        "JA": "JP",   # Japan
        "SP": "ES",   # Spain
        "UK": "GB",   # United Kingdom
        "UP": "UA",   # Ukraine
        "PO": "PL",   # Poland
        "FR": "FR",   # France (same)
        "IT": "IT",   # Italy (same)
    }

    def __init__(
        self,
        expected_daily_events: int = 500_000,
        min_date_str: str = "2020-01-01",
    ):
        self._bloom = BloomFilter(expected_items=expected_daily_events * 30)
        from datetime import date
        y, m, d = min_date_str.split("-")
        self._min_date = date(int(y), int(m), int(d))
        self._stats = DeduplicationStats()

    def clean_batch(
        self,
        rows: Iterable[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], DeduplicationStats]:
        """
        Remove duplicates and invalid rows from a batch.

        Returns:
            (clean_rows, batch_stats)
        """
        batch_stats = DeduplicationStats()
        clean: list[dict[str, Any]] = []

        for row in rows:
            batch_stats.total_in += 1
            self._stats.total_in += 1

            # --- Validity checks ---
            eid = row.get("global_event_id")
            edate = row.get("event_date")

            if eid is None or edate is None:
                batch_stats.invalid_removed += 1
                self._stats.invalid_removed += 1
                continue

            if edate < self._min_date:
                batch_stats.invalid_removed += 1
                self._stats.invalid_removed += 1
                continue

            # Skip rows with no country (can't assign to feature store)
            if not row.get("action_geo_country"):
                batch_stats.invalid_removed += 1
                self._stats.invalid_removed += 1
                continue

            # --- Bloom filter dedup ---
            if eid in self._bloom:
                batch_stats.duplicates_removed += 1
                self._stats.duplicates_removed += 1
                continue

            self._bloom.add(eid)

            # --- Normalization ---
            row = self._normalize(row)
            clean.append(row)

        logger.debug(
            "Batch cleaned: %d → %d (dups=%d invalid=%d)",
            batch_stats.total_in,
            batch_stats.total_out,
            batch_stats.duplicates_removed,
            batch_stats.invalid_removed,
        )
        return clean, batch_stats

    def _normalize(self, row: dict[str, Any]) -> dict[str, Any]:
        """Apply field-level normalizations in place (returns copy)."""
        row = dict(row)

        # Country code normalization
        for field in ("action_geo_country", "actor1_country", "actor2_country"):
            code = row.get(field)
            if code:
                row[field] = self.FIPS_TO_ISO.get(code.upper(), code.upper())

        # Truncate URLs that exceed column width
        url = row.get("source_url")
        if url and len(url) > 2048:
            row["source_url"] = url[:2048]

        # Clamp coordinates
        lat = row.get("latitude")
        lon = row.get("longitude")
        if lat is not None and (lat < -90 or lat > 90):
            row["latitude"] = None
        if lon is not None and (lon < -180 or lon > 180):
            row["longitude"] = None

        return row

    @property
    def cumulative_stats(self) -> DeduplicationStats:
        return self._stats

    def reset_bloom(self) -> None:
        """Reset the bloom filter — call between days to avoid false positives."""
        old_size = len(self._bloom)
        self._bloom = BloomFilter(expected_items=500_000 * 30)
        logger.info("Bloom filter reset (was tracking ~%d IDs)", old_size)
