"""
Phase 3: ParquetForecastDataset — training data for EscalationForecaster.

Loads the bi-weekly parquet cache files produced by train_real_data.py
and builds (context_window, target_labels) pairs for seq2seq training.

File format: {YYYYMMDD}_features.parquet
    Columns: country, f0, f1, f2, f3, f4, f5, f6, date
    f0=protest_score, f1=violence_score, f2=diplomatic_stress,
    f3=economic_stress, f4=terrorism_score, f5=avg_sentiment(tone_neg),
    f6=goldstein_norm

Dataset construction:
    For each country C and each file index i where i >= context_steps:
        context  = last `context_steps` bi-weekly snapshots ending at i-1
                   shape: (context_steps, 7)
        targets  = next `horizon` snapshots starting at i
                   shape: (horizon, 7)  → converted to labels (horizon, 5)
        labels   = features_to_labels(targets[h]) for each h

    Labels vector (5-dim):
        [instability, war, terrorism, financial, risk_score]
        derived from features using domain mapping (matches heuristic scorer).

Chronological train/val/test split by snapshot file index:
    train: files  0 .. n_train-1
    val:   files  n_train .. n_train+n_val-1
    test:  files  n_train+n_val .. end
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("models.forecaster_dataset")

FEATURE_NAMES = [
    "protest_score", "violence_score", "diplomatic_stress",
    "economic_stress", "terrorism_score", "avg_sentiment", "goldstein_norm",
]
PARQUET_FEAT_COLS = ["f0", "f1", "f2", "f3", "f4", "f5", "f6"]   # same order as FEATURE_NAMES


# ---------------------------------------------------------------------------
# Label derivation (must match inference/risk_scorer.py heuristic logic)
# ---------------------------------------------------------------------------

def features_to_labels(feat: np.ndarray) -> np.ndarray:
    """
    Map a (7,) feature vector to a (5,) label vector.
    [instability, war, terrorism, financial, risk_score]

    Formula mirrors _heuristic_score() in inference/risk_scorer.py
    so training targets are consistent with runtime scoring.
    """
    protest   = float(feat[0])
    violence  = float(feat[1])
    diplo     = float(feat[2])
    economic  = float(feat[3])
    terror    = float(feat[4])
    tone_neg  = float(feat[5])
    goldstein = float(feat[6])

    instability = min(0.5 * violence + 0.5 * protest, 1.0)
    war         = min(0.4 * violence + 0.4 * diplo + 0.2 * goldstein, 1.0)
    terrorism   = min(terror * 1.2, 1.0)
    financial   = min(0.7 * economic + 0.3 * tone_neg, 1.0)
    risk_score  = min(
        0.40 * instability
        + 0.30 * war
        + 0.20 * terrorism
        + 0.10 * financial,
        1.0,
    )
    return np.array([instability, war, terrorism, financial, risk_score], dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    cache_dir: str = "data/real_cache"
    context_steps: int = 6     # bi-weekly steps of context (= ~12 weeks)
    horizon: int = 4            # bi-weekly steps to predict (= ~8 weeks)
    split: str = "train"        # "train" | "val" | "test"
    train_frac: float = 0.70
    val_frac: float = 0.15
    # test_frac = 1 - train_frac - val_frac


class ParquetForecastDataset(Dataset):
    """
    PyTorch Dataset backed by bi-weekly parquet feature snapshots.

    Each sample:
        context:  (context_steps, 7) float32 — historical feature windows
        targets:  (horizon, 7)       float32 — future feature windows
        labels:   (horizon, 5)       float32 — derived risk labels per step
        country:  str
        snap_date: str  (ISO date string of the context-end snapshot)
    """

    def __init__(self, cfg: DatasetConfig):
        self._cfg = cfg
        self._samples: list[dict] = []
        self._load_and_build(cfg)

    def _load_and_build(self, cfg: DatasetConfig) -> None:
        cache_path = Path(cfg.cache_dir)
        parquet_files = sorted(cache_path.glob("*_features.parquet"))

        if not parquet_files:
            logger.warning("No parquet files found in %s", cfg.cache_dir)
            return

        logger.info("Loading %d parquet files from %s", len(parquet_files), cache_path)

        # Load all files into memory: {file_idx: {country: feature_vec(7,)}}
        snapshots: list[dict[str, np.ndarray]] = []
        snap_dates: list[str] = []

        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                # Normalise column names
                if "f0" not in df.columns and "protest_score" in df.columns:
                    col_map = {n: f"f{i}" for i, n in enumerate(FEATURE_NAMES)}
                    df = df.rename(columns=col_map)

                snap: dict[str, np.ndarray] = {}
                for _, row in df.iterrows():
                    country = str(row.get("country", ""))
                    if not country:
                        continue
                    feat = np.array(
                        [float(row.get(c, 0.0)) for c in PARQUET_FEAT_COLS],
                        dtype=np.float32,
                    )
                    snap[country] = feat

                snapshots.append(snap)
                # Extract date from filename: YYYYMMDD_features.parquet
                date_str = pf.stem.split("_")[0]
                snap_dates.append(date_str)

            except Exception as exc:
                logger.warning("Failed to load %s: %s", pf.name, exc)

        if len(snapshots) < cfg.context_steps + cfg.horizon:
            logger.warning(
                "Not enough snapshots (%d) for context_steps=%d + horizon=%d",
                len(snapshots), cfg.context_steps, cfg.horizon,
            )
            return

        n = len(snapshots)
        total_needed = cfg.context_steps + cfg.horizon

        # Determine split ranges by file index
        n_train = int(n * cfg.train_frac)
        n_val   = int(n * cfg.val_frac)

        if cfg.split == "train":
            valid_range = range(cfg.context_steps, n_train)
        elif cfg.split == "val":
            valid_range = range(max(cfg.context_steps, n_train), n_train + n_val)
        else:   # test
            valid_range = range(max(cfg.context_steps, n_train + n_val), n)

        # Get all countries that appear in at least (context+horizon) snapshots
        all_countries: set[str] = set()
        for snap in snapshots:
            all_countries |= snap.keys()

        samples = []
        for i in valid_range:
            # context: snapshots [i-context_steps .. i-1]
            # targets: snapshots [i .. i+horizon-1]
            if i + cfg.horizon > n:
                continue

            ctx_idxs = list(range(i - cfg.context_steps, i))
            tgt_idxs = list(range(i, i + cfg.horizon))

            for country in all_countries:
                # Build context window — use zeros for missing dates
                ctx = np.zeros((cfg.context_steps, 7), dtype=np.float32)
                for k, si in enumerate(ctx_idxs):
                    if country in snapshots[si]:
                        ctx[k] = snapshots[si][country]

                # Skip countries with almost no data in the context
                if (ctx.sum(axis=1) > 0).sum() < 2:
                    continue

                # Build target window
                tgt = np.zeros((cfg.horizon, 7), dtype=np.float32)
                has_tgt = 0
                for k, si in enumerate(tgt_idxs):
                    if country in snapshots[si]:
                        tgt[k] = snapshots[si][country]
                        has_tgt += 1

                if has_tgt == 0:
                    continue

                # Derive labels
                labels = np.stack(
                    [features_to_labels(tgt[k]) for k in range(cfg.horizon)],
                    axis=0,
                )   # (horizon, 5)

                samples.append({
                    "context":   ctx,
                    "targets":   tgt,
                    "labels":    labels,
                    "country":   country,
                    "snap_date": snap_dates[i - 1] if (i - 1) < len(snap_dates) else "",
                })

        self._samples = samples
        logger.info(
            "Dataset [%s]: %d samples, %d snapshots, %d countries",
            cfg.split, len(samples), n, len(all_countries),
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        s = self._samples[idx]
        return {
            "context":   torch.from_numpy(s["context"]).float(),    # (context_steps, 7)
            "targets":   torch.from_numpy(s["targets"]).float(),    # (horizon, 7)
            "labels":    torch.from_numpy(s["labels"]).float(),     # (horizon, 5)
            "country":   s["country"],
            "snap_date": s["snap_date"],
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Custom collate that keeps string fields out of the tensor stack."""
        countries  = [b["country"]   for b in batch]
        snap_dates = [b["snap_date"] for b in batch]
        return {
            "context":   torch.stack([b["context"] for b in batch]),
            "targets":   torch.stack([b["targets"] for b in batch]),
            "labels":    torch.stack([b["labels"]  for b in batch]),
            "country":   countries,
            "snap_date": snap_dates,
        }
