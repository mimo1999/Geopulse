"""
Phase 2: Multi-task dataset with 4-dimensional ground-truth labels.

Extends CountryRiskDataset to load proxy labels from
country_multitask_labels instead of using a single risk_score label.

Labels loaded: instability_label, war_label, terrorism_label, financial_label
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import psycopg2
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from models.dataset import (
    FEATURE_COLUMNS,
    NUM_FEATURES,
    risk_collate_fn,
    RiskSample,
)

logger = logging.getLogger("models.multitask_dataset")

LABEL_COLUMNS_V2 = [
    "instability_label",
    "war_label",
    "terrorism_label",
    "financial_label",
]
NUM_LABELS_V2 = 4


def multitask_collate_fn(batch: list[RiskSample]) -> dict[str, Tensor]:
    """Stack batch into (B,T,F) features and (B,4) labels."""
    features = torch.stack([s.features for s in batch])
    labels   = torch.stack([s.labels   for s in batch])
    mask     = torch.stack([s.mask     for s in batch])
    return {"features": features, "labels": labels, "mask": mask}


class MultiTaskRiskDataset(Dataset):
    """
    Temporal sliding-window dataset with 4 ground-truth proxy labels.

    Each sample: 90-day feature window → (instability, war, terrorism, financial).

    Requires country_multitask_labels to be populated by LabelGenerator.
    Falls back gracefully to zeros if a label row is missing.

    Args:
        dsn:          PostgreSQL DSN.
        start_date:   First window-end date to include.
        end_date:     Last window-end date to include.
        seq_len:      Days of history per sample (default 90).
        stride:       Step between windows in days (default 7).
        countries:    Restrict to these countries (None = all labeled).
        min_coverage: Min fraction of non-zero days to include a sample.
        label_version: Only use labels with this version tag.
    """

    _FETCH_FEATURES_SQL = """
        SELECT feature_date, {cols}
        FROM country_daily_features
        WHERE country = %s
          AND feature_date >= %s AND feature_date <= %s
        ORDER BY feature_date ASC
    """

    _FETCH_LABEL_SQL = """
        SELECT instability_label, war_label, terrorism_label, financial_label
        FROM country_multitask_labels
        WHERE country = %s AND label_date = %s
    """

    _FETCH_LABELED_COUNTRIES_SQL = """
        SELECT DISTINCT country
        FROM country_multitask_labels
        WHERE label_date >= %s AND label_date <= %s
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
        min_coverage: float = 0.4,
        label_version: Optional[str] = None,
    ):
        self._dsn = dsn
        self._seq_len = seq_len
        self._min_coverage = min_coverage
        self._label_version = label_version
        self._feature_cache: dict = {}
        self._label_cache: dict = {}

        if countries is None:
            countries = self._fetch_labeled_countries(start_date, end_date)

        logger.info(
            "Building MultiTask dataset: %d countries %s→%s",
            len(countries), start_date, end_date,
        )

        self._samples: list[tuple[str, date]] = []
        for country in countries:
            current = start_date + timedelta(days=seq_len - 1)
            while current <= end_date:
                self._samples.append((country, current))
                current += timedelta(days=stride)

        logger.info("MultiTask dataset: %d samples", len(self._samples))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> RiskSample:
        country, window_end = self._samples[idx]
        window_start = window_end - timedelta(days=self._seq_len - 1)

        # Feature matrix (seq_len, num_features)
        feat_matrix = self._load_features(country, window_start, window_end)

        # Multi-task labels for the window-end date
        labels = self._load_labels(country, window_end)

        # Valid-day mask
        mask = (feat_matrix.sum(axis=1) != 0).astype(np.float32)

        return RiskSample(
            country=country,
            window_end=window_end,
            features=torch.from_numpy(feat_matrix.astype(np.float32)),
            labels=torch.from_numpy(labels.astype(np.float32)),
            mask=torch.from_numpy(mask),
        )

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_features(self, country: str, start: date, end: date) -> np.ndarray:
        key = (country, start, end)
        if key in self._feature_cache:
            return self._feature_cache[key]

        cols = ", ".join(FEATURE_COLUMNS)
        conn = psycopg2.connect(self._dsn)
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    self._FETCH_FEATURES_SQL.format(cols=cols),
                    (country, start, end),
                )
                db_rows = cur.fetchall()
        finally:
            conn.close()

        by_date = {r[0]: list(r[1:]) for r in db_rows}
        matrix = np.zeros((self._seq_len, NUM_FEATURES), dtype=np.float64)
        last_valid = None
        for i in range(self._seq_len):
            d = start + timedelta(days=i)
            if d in by_date:
                vals = [v if v is not None else 0.0 for v in by_date[d]]
                matrix[i] = vals
                last_valid = vals
            elif last_valid is not None:
                matrix[i] = last_valid

        np.nan_to_num(matrix, copy=False, nan=0.0)
        self._feature_cache[key] = matrix
        return matrix

    def _load_labels(self, country: str, target_date: date) -> np.ndarray:
        key = (country, target_date)
        if key in self._label_cache:
            return self._label_cache[key]

        conn = psycopg2.connect(self._dsn)
        try:
            with conn, conn.cursor() as cur:
                cur.execute(self._FETCH_LABEL_SQL, (country, target_date))
                row = cur.fetchone()
        finally:
            conn.close()

        if row:
            labels = np.array([r if r is not None else 0.0 for r in row], dtype=np.float64)
        else:
            labels = np.zeros(NUM_LABELS_V2, dtype=np.float64)

        self._label_cache[key] = labels
        return labels

    def _fetch_labeled_countries(self, start: date, end: date) -> list[str]:
        conn = psycopg2.connect(self._dsn)
        try:
            with conn, conn.cursor() as cur:
                cur.execute(self._FETCH_LABELED_COUNTRIES_SQL, (start, end))
                return [r[0] for r in cur.fetchall()]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Split utility
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
    ) -> tuple["MultiTaskRiskDataset", "MultiTaskRiskDataset", "MultiTaskRiskDataset"]:
        from datetime import date as _date

        train = cls(dsn, _date(2020, 1, 1), train_end, seq_len=seq_len, **kwargs)
        val   = cls(dsn, train_end + timedelta(days=1), val_end, seq_len=seq_len, **kwargs)
        test  = cls(dsn, val_end + timedelta(days=1), test_end, seq_len=seq_len, **kwargs)

        logger.info(
            "Split: train=%d val=%d test=%d",
            len(train), len(val), len(test),
        )
        return train, val, test


class MultiTaskDataLoader:
    """
    Factory for multi-task train/val/test DataLoaders.
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
        train_ds, val_ds, test_ds = MultiTaskRiskDataset.train_val_test_split(
            dsn=dsn,
            train_end=train_end,
            val_end=val_end,
            test_end=test_end,
            seq_len=seq_len,
            stride=stride_train,
            countries=countries,
        )
        val_ds._samples  = _resample(val_ds._samples, stride_eval, seq_len, val_end)
        test_ds._samples = _resample(test_ds._samples, stride_eval, seq_len, test_end)

        self.train = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=multitask_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
        self.val = DataLoader(
            val_ds, batch_size=batch_size * 2, shuffle=False,
            num_workers=num_workers, collate_fn=multitask_collate_fn,
        )
        self.test = DataLoader(
            test_ds, batch_size=batch_size * 2, shuffle=False,
            num_workers=num_workers, collate_fn=multitask_collate_fn,
        )
        logger.info(
            "MultiTask loaders ready — train=%d val=%d test=%d batches",
            len(self.train), len(self.val), len(self.test),
        )


def _resample(
    samples: list[tuple[str, date]],
    stride: int,
    seq_len: int,
    end: date,
) -> list[tuple[str, date]]:
    """Re-stride an eval sample list."""
    by_country: dict[str, list[date]] = {}
    for country, d in samples:
        by_country.setdefault(country, []).append(d)

    result = []
    for country, dates in by_country.items():
        start = min(dates)
        current = start
        while current <= end:
            result.append((country, current))
            current += timedelta(days=stride)
    return result
