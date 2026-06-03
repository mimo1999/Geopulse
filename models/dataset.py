"""
PyTorch Dataset and DataLoader for country risk time-series.

CountryRiskDataset:
  - Loads 90-day windows of country_daily_features from PostgreSQL.
  - Each sample is (features_tensor, label_tensor, country, date).
  - Handles missing dates with forward-fill then zero-fill.
  - Supports train / val / test splits by date.

RiskDataLoader:
  - Thin wrapper around DataLoader with sensible defaults.
  - Provides collate_fn that handles variable-length sequences via padding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import psycopg2
import psycopg2.extras
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("models.dataset")


# ---------------------------------------------------------------------------
# Feature / label column definitions
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "protest_score",
    "violence_score",
    "diplomatic_stress",
    "economic_stress",
    "terrorism_score",
    "avg_sentiment",
    "avg_goldstein",
]
NUM_FEATURES = len(FEATURE_COLUMNS)   # 7

LABEL_COLUMNS = [
    "risk_score",          # composite risk (0–1)
]
NUM_LABELS = len(LABEL_COLUMNS)       # 1 (Phase 1); expand in Phase 2


# ---------------------------------------------------------------------------
# Sample type
# ---------------------------------------------------------------------------

@dataclass
class RiskSample:
    country: str
    window_end: date
    features: Tensor          # (seq_len, num_features)  float32
    labels: Tensor            # (num_labels,)             float32
    mask: Tensor              # (seq_len,)  1=valid 0=padded


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CountryRiskDataset(Dataset):
    """
    Temporal sliding-window dataset over country_daily_features.

    Each sample covers `seq_len` consecutive days ending at `window_end`.
    Label is the risk_score on `window_end`.

    Args:
        dsn: PostgreSQL connection string.
        start_date: First possible window-end date.
        end_date: Last possible window-end date.
        seq_len: Days of history per sample (default 90).
        stride: Step size between windows in days (default 1).
        countries: Optional list of countries to include (None = all).
        min_coverage: Minimum fraction of non-null days required (0–1).
    """

    _FETCH_WINDOW_SQL = """
        SELECT
            feature_date,
            {feature_cols},
            risk_score
        FROM country_daily_features
        WHERE country = %s
          AND feature_date >= %s
          AND feature_date <= %s
        ORDER BY feature_date ASC
    """

    _FETCH_COUNTRIES_SQL = """
        SELECT DISTINCT country
        FROM country_daily_features
        WHERE feature_date >= %s
          AND feature_date <= %s
          AND risk_score IS NOT NULL
        ORDER BY country
    """

    def __init__(
        self,
        dsn: str,
        start_date: date,
        end_date: date,
        seq_len: int = 90,
        stride: int = 7,
        countries: Optional[list[str]] = None,
        min_coverage: float = 0.5,
    ):
        self._dsn = dsn
        self._seq_len = seq_len
        self._stride = stride
        self._min_coverage = min_coverage

        # Build index of (country, window_end) pairs
        self._samples: list[tuple[str, date]] = []
        self._cache: dict[tuple[str, date, date], np.ndarray] = {}

        if countries is None:
            countries = self._fetch_countries(start_date, end_date)
        logger.info(
            "Building dataset index: %d countries, %s → %s",
            len(countries), start_date, end_date,
        )

        for country in countries:
            current = start_date + timedelta(days=seq_len - 1)
            while current <= end_date:
                self._samples.append((country, current))
                current += timedelta(days=stride)

        logger.info("Dataset: %d samples indexed", len(self._samples))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> RiskSample:
        country, window_end = self._samples[idx]
        window_start = window_end - timedelta(days=self._seq_len - 1)

        matrix = self._load_window(country, window_start, window_end)

        # Features: first NUM_FEATURES columns
        feat_arr = matrix[:, :NUM_FEATURES].copy()  # (seq_len, 7)
        # Labels: last column
        label_arr = matrix[-1, NUM_FEATURES:]       # (1,)

        # Build valid-day mask (row is valid if at least one feature is non-zero)
        mask = (feat_arr.sum(axis=1) != 0).astype(np.float32)

        return RiskSample(
            country=country,
            window_end=window_end,
            features=torch.from_numpy(feat_arr.astype(np.float32)),
            labels=torch.from_numpy(label_arr.astype(np.float32)),
            mask=torch.from_numpy(mask),
        )

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_window(
        self,
        country: str,
        start: date,
        end: date,
    ) -> np.ndarray:
        """
        Load a dense (seq_len, num_features + num_labels) matrix.
        Missing dates are forward-filled then zero-filled.
        """
        cache_key = (country, start, end)
        if cache_key in self._cache:
            return self._cache[cache_key]

        conn = psycopg2.connect(self._dsn)
        try:
            feat_sel = ", ".join(FEATURE_COLUMNS)
            sql = self._FETCH_WINDOW_SQL.format(feature_cols=feat_sel)
            with conn, conn.cursor() as cur:
                cur.execute(sql, (country, start, end))
                db_rows = cur.fetchall()
        finally:
            conn.close()

        # Build date → row mapping
        row_by_date: dict[date, list] = {}
        for row in db_rows:
            d = row[0]
            values = list(row[1:])   # features + risk_score
            row_by_date[d] = values

        num_cols = NUM_FEATURES + NUM_LABELS
        matrix = np.zeros((self._seq_len, num_cols), dtype=np.float64)

        # Fill in known days; forward-fill gaps
        last_valid: Optional[list] = None
        for i in range(self._seq_len):
            d = start + timedelta(days=i)
            if d in row_by_date:
                vals = row_by_date[d]
                # Replace None with NaN for numpy
                clean = [v if v is not None else np.nan for v in vals]
                matrix[i] = clean
                last_valid = clean
            elif last_valid is not None:
                matrix[i] = last_valid  # forward fill

        # Zero-fill any remaining NaN
        np.nan_to_num(matrix, copy=False, nan=0.0)

        self._cache[cache_key] = matrix
        return matrix

    def _fetch_countries(self, start: date, end: date) -> list[str]:
        conn = psycopg2.connect(self._dsn)
        try:
            with conn, conn.cursor() as cur:
                cur.execute(self._FETCH_COUNTRIES_SQL, (start, end))
                return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Dataset splitting utilities
    # ------------------------------------------------------------------

    @classmethod
    def train_val_test_split(
        cls,
        dsn: str,
        train_end: date,
        val_end: date,
        test_end: date,
        seq_len: int = 90,
        **kwargs,
    ) -> tuple["CountryRiskDataset", "CountryRiskDataset", "CountryRiskDataset"]:
        """
        Create chronological train / val / test splits.
        No overlap — splits by window_end date.
        """
        # Training windows end before val_end
        train_start = date(2020, 1, 1)
        train_ds = cls(dsn, train_start, train_end, seq_len=seq_len, **kwargs)

        # Val windows end before test_end
        val_ds = cls(dsn, train_end + timedelta(days=1), val_end, seq_len=seq_len, **kwargs)

        # Test windows are newest
        test_ds = cls(dsn, val_end + timedelta(days=1), test_end, seq_len=seq_len, **kwargs)

        logger.info(
            "Split sizes — train: %d  val: %d  test: %d",
            len(train_ds), len(val_ds), len(test_ds),
        )
        return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def risk_collate_fn(batch: list[RiskSample]) -> dict[str, Tensor]:
    """
    Stack a list of RiskSample into batched tensors.

    Returns dict with keys:
        features  (B, T, F)
        labels    (B, L)
        mask      (B, T)
    """
    features = torch.stack([s.features for s in batch])
    labels   = torch.stack([s.labels   for s in batch])
    mask     = torch.stack([s.mask     for s in batch])
    return {"features": features, "labels": labels, "mask": mask}


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

class RiskDataLoader:
    """
    Factory for train / val / test DataLoaders with sensible defaults.
    """

    def __init__(
        self,
        dsn: str,
        train_end: date,
        val_end: date,
        test_end: date,
        seq_len: int = 90,
        batch_size: int = 64,
        num_workers: int = 2,
        stride_train: int = 7,
        stride_eval: int = 14,
        countries: Optional[list[str]] = None,
    ):
        common = dict(dsn=dsn, seq_len=seq_len, countries=countries)
        train_ds, val_ds, test_ds = CountryRiskDataset.train_val_test_split(
            stride=stride_train, **common,
            train_end=train_end, val_end=val_end, test_end=test_end,
        )
        # Override stride for eval sets
        val_ds._stride  = stride_eval
        test_ds._stride = stride_eval

        self.train = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=risk_collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
        self.val = DataLoader(
            val_ds,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=risk_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
        self.test = DataLoader(
            test_ds,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=risk_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

        logger.info(
            "DataLoaders ready — train batches: %d  val: %d  test: %d",
            len(self.train), len(self.val), len(self.test),
        )
