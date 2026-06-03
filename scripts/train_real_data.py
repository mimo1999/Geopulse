"""
Real-data training for HybridRiskTransformer.

Downloads GDELT v1 daily files for the last 2 years at configurable
sampling intervals, builds country-level feature sequences, trains the
model for N epochs, and reports per-task accuracy metrics.

No psycopg2 / database required  -- runs standalone.

Usage:
    python scripts/train_real_data.py                  # defaults
    python scripts/train_real_data.py --interval 7     # weekly files
    python scripts/train_real_data.py --epochs 5 --batch 32
    python scripts/train_real_data.py --cache-only     # skip download, use cache
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, random_split

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so we can import models.*
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.risk_model import HybridRiskTransformer   # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GDELT_V1_BASE = "http://data.gdeltproject.org/events"

# Full 58-column schema for GDELT v1 (matches ingestion/gdelt_downloader.py)
GDELT_V1_ALL_COLS = [
    "global_event_id", "sqldate", "month_year", "year", "fraction_date",
    "actor1_code", "actor1_name", "actor1_country_code",
    "actor1_known_group_code", "actor1_ethnic_code",
    "actor1_religion1_code", "actor1_religion2_code",
    "actor1_type1_code", "actor1_type2_code", "actor1_type3_code",
    "actor2_code", "actor2_name", "actor2_country_code",
    "actor2_known_group_code", "actor2_ethnic_code",
    "actor2_religion1_code", "actor2_religion2_code",
    "actor2_type1_code", "actor2_type2_code", "actor2_type3_code",
    "is_root_event", "event_code", "event_base_code", "event_root_code",
    "quad_class", "goldstein_scale", "num_mentions", "num_sources",
    "num_articles", "avg_tone",
    "actor1_geo_type", "actor1_geo_fullname", "actor1_geo_country_code",
    "actor1_geo_adm1_code", "actor1_geo_lat", "actor1_geo_long",
    "actor1_geo_feature_id",
    "actor2_geo_type", "actor2_geo_fullname", "actor2_geo_country_code",
    "actor2_geo_adm1_code", "actor2_geo_lat", "actor2_geo_long",
    "actor2_geo_feature_id",
    "action_geo_type", "action_geo_fullname", "action_geo_country_code",
    "action_geo_adm1_code", "action_geo_lat", "action_geo_long",
    "action_geo_feature_id",
    "date_added", "source_url",
]  # 58 columns total

# The subset we actually keep after loading
KEEP_COLS = [
    "sqldate",
    "actor1_country_code",
    "event_base_code",
    "event_root_code",
    "quad_class",
    "goldstein_scale",
    "num_mentions",
    "avg_tone",
    "action_geo_country_code",
]

# 7-feature vector matching HybridRiskTransformer's num_features=7
FEATURE_NAMES = [
    "protest_score",    # fraction of conflictual events (quad 3-4)
    "violence_score",   # fraction of material conflict (quad 4)
    "diplo_stress",     # normalized negative goldstein (conflict signal)
    "econ_stress",      # fraction of sanctions / economic coercion events
    "terror_score",     # fraction of CAMEO 18x (terrorism) events
    "tone_neg",         # normalized negative avg_tone
    "goldstein_norm",   # normalized mean goldstein [0,1]
]

NUM_FEATURES = 7
TASK_NAMES = ["instability", "war", "terrorism", "financial"]

# ---------------------------------------------------------------------------
# Helpers: GDELT download + parse
# ---------------------------------------------------------------------------

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "GLDT-Research-Academic/2.0"})


def _v1_url(d: date) -> str:
    return f"{GDELT_V1_BASE}/{d.strftime('%Y%m%d')}.export.CSV.zip"


def download_and_parse(d: date, max_retries: int = 3, timeout: int = 90) -> Optional[pd.DataFrame]:
    """
    Download one GDELT v1 daily ZIP and return a tidy DataFrame with only
    the columns we need.  Returns None on failure.
    """
    url = _v1_url(d)
    for attempt in range(max_retries):
        try:
            resp = _SESSION.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()
            raw = resp.content
            break
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                return None

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as fh:
                # Provide all 58 column names so usecols can reference by name
                df = pd.read_csv(
                    fh,
                    sep="\t",
                    header=None,
                    names=GDELT_V1_ALL_COLS,
                    usecols=KEEP_COLS,
                    dtype=str,
                    na_values=["", "NA", "NULL"],
                    low_memory=False,
                    on_bad_lines="skip",
                )
        return df
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Feature aggregation: DataFrame -> (country -> feature_vector)
# ---------------------------------------------------------------------------

def _safe_num(series: pd.Series, fill: float = 0.0) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(fill).values


def aggregate_day(df: pd.DataFrame, min_events: int = 5) -> dict:
    """
    Aggregate one day's GDELT events into per-country feature vectors.

    Country key: action_geo_country_code (FIPS 2-char) when available,
    else first 3 chars of actor1_country_code (CAMEO).

    Returns dict: { country_code (str) -> np.ndarray shape (7,) }
    """
    df = df.copy()

    # Primary: action_geo_country_code (FIPS 2-char, e.g. "US", "RS")
    df["country"] = df["action_geo_country_code"].where(
        df["action_geo_country_code"].notna()
        & (df["action_geo_country_code"].str.len() == 2),
        other=None,
    )
    # Fallback: actor1_country_code (CAMEO 3-char, e.g. "USA", "RUS")
    mask_missing = df["country"].isna()
    df.loc[mask_missing, "country"] = df.loc[mask_missing, "actor1_country_code"].where(
        df.loc[mask_missing, "actor1_country_code"].notna(), other=None
    )
    df = df.dropna(subset=["country"])
    df = df[df["country"].str.len().between(2, 3)]   # sanity: 2 or 3-char codes

    out = {}
    for cc, grp in df.groupby("country"):
        if len(grp) < min_events:
            continue

        mentions = _safe_num(grp["num_mentions"], 1.0).clip(1)
        total_w  = float(mentions.sum())

        quad = _safe_num(grp["quad_class"]).astype(int)
        protest_w  = float(mentions[np.isin(quad, [3, 4])].sum())
        violence_w = float(mentions[quad == 4].sum())
        protest_score  = protest_w  / total_w
        violence_score = violence_w / total_w

        gold = _safe_num(grp["goldstein_scale"])
        avg_gold       = float(np.average(gold, weights=mentions))
        goldstein_norm = (avg_gold + 10.0) / 20.0       # [0, 1]
        diplo_stress   = max(0.0, -avg_gold) / 10.0     # conflict signal [0, 1]

        tone = _safe_num(grp["avg_tone"])
        avg_tone_ = float(np.average(tone, weights=mentions))
        tone_neg  = min(1.0, max(0.0, -avg_tone_) / 30.0)  # neg tone [0, 1]

        # Terrorism: CAMEO base codes starting with "18"
        bcode    = grp["event_base_code"].astype(str).str[:2]
        terror_w = float(mentions[(bcode == "18").values].sum())
        terror_score = terror_w / total_w

        # Economic stress: CAMEO root codes "16" (sanctions/reduce relations)
        #                  + "17" (coerce/impose embargo) for broader coverage
        rcode   = grp["event_root_code"].astype(str).str[:2]
        econ_w  = float(mentions[rcode.isin(["16", "17"]).values].sum())
        econ_stress = econ_w / total_w

        out[cc] = np.array([
            protest_score,
            violence_score,
            diplo_stress,
            econ_stress,
            terror_score,
            tone_neg,
            goldstein_norm,
        ], dtype=np.float32)

    return out


def percentile_normalize_day(feat_dict: dict) -> dict:
    """
    Replace each raw feature value with its within-day percentile rank [0, 1].

    For each of the 7 features independently, a country at the 95th percentile
    of the global distribution for that day gets a score of 0.95.  This means
    a country like Ukraine with 25% violence-fraction (which is the 97th
    percentile globally) is correctly rated near 1.0 instead of 0.25.

    goldstein_norm is INVERTED first so that high conflict (low goldstein)
    maps to high percentile, consistent with all other risk features.
    """
    if len(feat_dict) < 5:
        return feat_dict   # too few countries to rank meaningfully

    ccs   = list(feat_dict.keys())
    mat   = np.stack([feat_dict[cc] for cc in ccs])   # (N, 7)

    # Invert goldstein_norm (index 6) so low goldstein → high risk percentile
    mat[:, 6] = 1.0 - mat[:, 6]

    ranked = np.zeros_like(mat)
    N = mat.shape[0]
    for f in range(mat.shape[1]):
        col  = mat[:, f]
        order = np.argsort(col)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = (np.arange(N) + 1) / N   # [1/N … 1.0]
        ranked[:, f] = ranks

    return {cc: ranked[i] for i, cc in enumerate(ccs)}


# ---------------------------------------------------------------------------
# Cache: store per-day country features as a parquet file
# ---------------------------------------------------------------------------

def cache_path(d: date, cache_dir: Path) -> Path:
    return cache_dir / f"{d.strftime('%Y%m%d')}_features.parquet"


def load_or_build_cache(d: date, cache_dir: Path, min_events: int) -> Optional[pd.DataFrame]:
    """
    Return a DataFrame with columns [country, f0..f6] for date d.
    Reads from cache if available, otherwise downloads & builds.
    """
    cp = cache_path(d, cache_dir)
    if cp.exists():
        try:
            return pd.read_parquet(cp)
        except Exception:
            cp.unlink(missing_ok=True)

    df_raw = download_and_parse(d)
    if df_raw is None:
        return None

    feat_dict = aggregate_day(df_raw, min_events=min_events)
    if not feat_dict:
        return None

    # Percentile-rank within the day so high-conflict countries stand out
    # even when their raw event-fraction is diluted by large coverage volume
    feat_dict = percentile_normalize_day(feat_dict)

    rows = [[cc] + feat.tolist() for cc, feat in feat_dict.items()]
    cols = ["country"] + [f"f{i}" for i in range(NUM_FEATURES)]
    df_feat = pd.DataFrame(rows, columns=cols)
    df_feat["date"] = d.strftime("%Y-%m-%d")

    try:
        df_feat.to_parquet(cp, index=False)
    except Exception:
        pass  # cache save failure is non-fatal

    return df_feat


# ---------------------------------------------------------------------------
# Build multi-day country feature matrix
# ---------------------------------------------------------------------------

def build_country_timeseries(
    sample_dates: List[date],
    cache_dir: Path,
    min_events: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Downloads / loads data for each sample date.

    Returns dict:
        { country_code -> np.ndarray shape (n_valid_dates, NUM_FEATURES) }
    """
    country_data: dict = {}   # cc -> list of (date_idx, feat)

    total = len(sample_dates)
    failed = 0

    for idx, d in enumerate(sample_dates):
        pct = (idx + 1) / total * 100
        msg = f"  [{idx+1:3d}/{total}] {d}  ({pct:.0f}%)".ljust(50)
        if verbose:
            print(msg, end="\r", flush=True)

        df_feat = load_or_build_cache(d, cache_dir, min_events)
        if df_feat is None:
            failed += 1
            continue

        for _, row in df_feat.iterrows():
            cc = row["country"]
            feat = row[[f"f{i}" for i in range(NUM_FEATURES)]].values.astype(np.float32)
            if cc not in country_data:
                country_data[cc] = []
            country_data[cc].append((idx, feat))

    if verbose:
        print()  # newline after progress

    if verbose and failed > 0:
        print(f"  Warning: {failed}/{total} dates failed to download.")

    # Convert to dense arrays (fill missing dates with zeros)
    result = {}
    for cc, pairs in country_data.items():
        arr = np.zeros((total, NUM_FEATURES), dtype=np.float32)
        for date_idx, feat in pairs:
            arr[date_idx] = feat
        result[cc] = arr

    return result


# ---------------------------------------------------------------------------
# Proxy labels from feature window (mirrors label_generator.py logic)
# ---------------------------------------------------------------------------

def compute_proxy_labels(feat_window: np.ndarray) -> np.ndarray:
    """
    Derive 4-dim proxy label vector from a (seq_len, 7) feature window.

    Uses the tail of the window (last 5 steps or all if shorter) to
    represent the current risk state.  Labels are [0,1] clipped.

    Returns np.ndarray shape (4,): [instability, war, terrorism, financial]
    """
    tail_len = min(5, len(feat_window))
    tail = feat_window[-tail_len:]

    # Column indices: protest(0), violence(1), diplo_stress(2),
    #                 econ_stress(3), terror(4), tone_neg(5), goldstein(6)
    protest   = float(tail[:, 0].mean())
    violence  = float(tail[:, 1].mean())
    diplo     = float(tail[:, 2].mean())
    econ      = float(tail[:, 3].mean())
    terror    = float(tail[:, 4].mean())
    tone_neg  = float(tail[:, 5].mean())
    goldstein = float(tail[:, 6].mean())  # normalized [0,1]

    # Conflict component: inverse of goldstein (higher = worse)
    conflict = 1.0 - goldstein

    instability = 0.4 * protest + 0.3 * violence + 0.2 * diplo + 0.1 * conflict
    war         = 0.5 * violence + 0.3 * diplo    + 0.2 * conflict
    terrorism   = 0.7 * terror  + 0.2 * violence  + 0.1 * tone_neg
    # Financial: amplify econ signal (raw fraction of root-16/17 events is small
    # even for heavily-sanctioned countries; scale by 2.5 then cap at 1.0 so
    # econ_stress > 0.2 → meaningful positive label, > 0.4 → clearly positive)
    econ_amp  = min(1.0, econ * 2.5)
    financial = 0.7 * econ_amp + 0.2 * tone_neg + 0.1 * conflict

    labels = np.array([instability, war, terrorism, financial], dtype=np.float32)
    return labels.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class RealGDELTDataset(Dataset):
    """
    Sliding-window dataset built from real GDELT country time series.

    Each sample: (features [seq_len, 7], mask [seq_len], labels [4])
    """

    def __init__(
        self,
        country_timeseries: dict,
        seq_len: int = 26,
        stride: int = 1,
        min_nonzero_steps: int = 5,
    ):
        self.seq_len = seq_len
        samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for cc, ts in country_timeseries.items():
            T = len(ts)
            if T < seq_len:
                continue
            for start in range(0, T - seq_len + 1, stride):
                window = ts[start : start + seq_len]  # (seq_len, 7)
                # Skip windows that are almost all zeros (country not present)
                nonzero_steps = (window.sum(axis=1) > 0).sum()
                if nonzero_steps < min_nonzero_steps:
                    continue
                mask = (window.sum(axis=1) > 0).astype(np.float32)  # (seq_len,)
                labels = compute_proxy_labels(window)
                samples.append((window, mask, labels))

        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        feat, mask, label = self._samples[idx]
        return (
            torch.from_numpy(feat),
            torch.from_numpy(mask),
            torch.from_numpy(label),
        )


# ---------------------------------------------------------------------------
# Loss functions (same as Phase 2 trainer)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class SmoothedBCE(nn.Module):
    def __init__(self, eps: float = 0.05):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_s = target * (1 - self.eps) + 0.5 * self.eps
        return nn.functional.binary_cross_entropy(pred, target_s)


class MultiTaskLoss(nn.Module):
    """Uncertainty-weighted multi-task loss (Kendal & Gal)."""

    def __init__(self):
        super().__init__()
        # log(sigma^2) for each task -- learned uncertainty weights
        self.log_var = nn.Parameter(torch.zeros(4))
        self.focal   = FocalLoss(alpha=0.25, gamma=2.0)
        self.sbce    = SmoothedBCE(eps=0.05)

    def forward(
        self,
        preds: dict,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        preds:   dict with keys instability/war/terrorism/financial  each (B,)
        targets: (B, 4)
        Returns: (total_loss, per_task_losses (4,))
        """
        losses = torch.stack([
            self.sbce (preds["instability"], targets[:, 0]),
            self.sbce (preds["war"],         targets[:, 1]),
            self.focal(preds["terrorism"],   targets[:, 2]),
            self.sbce (preds["financial"],   targets[:, 3]),
        ])  # (4,)

        # Uncertainty weighting: L_i / (2*sigma_i^2) + log(sigma_i)
        precision = torch.exp(-self.log_var)
        weighted  = precision * losses + 0.5 * self.log_var
        total     = weighted.sum()

        return total, losses.detach()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    country_ts: dict,
    seq_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build dataset
    dataset = RealGDELTDataset(country_ts, seq_len=seq_len, stride=1, min_nonzero_steps=5)
    n = len(dataset)
    if n < 10:
        raise RuntimeError(
            f"Not enough samples ({n}) to train. "
            "Try --interval 7 or --min-events 3 to increase data."
        )

    n_train = max(int(0.8 * n), n - 200)
    n_val   = n - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=device.type == "cuda")

    # Model
    model = HybridRiskTransformer(
        num_features=NUM_FEATURES,
        d_model=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        seq_len=seq_len,
    ).to(device)

    criterion = MultiTaskLoss().to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
    )

    history = {"train_loss": [], "val_loss": [], "val_auc": [], "task_losses": []}

    print(f"\n  Dataset  : {n:,} windows  (train={n_train:,}  val={n_val:,})")
    print(f"  Model    : {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Batches/epoch: {len(train_loader)}")
    print(f"  Device   : {device}")
    print()
    print("  " + "-" * 72)
    header = f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>9}  " \
             f"{'Inst AUC':>8}  {'War AUC':>7}  {'Terr AUC':>8}  {'Fin AUC':>7}"
    print(header)
    print("  " + "-" * 72)

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        criterion.train()
        train_losses = []
        task_loss_sum = torch.zeros(4)

        for feat, mask, labels in train_loader:
            feat   = feat.to(device)
            mask   = mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(feat, mask)
            loss, tl = criterion(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            task_loss_sum += tl.cpu()

        mean_train_loss = np.mean(train_losses)
        mean_task_loss  = (task_loss_sum / max(1, len(train_loader))).numpy()

        # --- Validation ---
        model.eval()
        val_losses = []
        all_preds  = [[] for _ in range(4)]
        all_labels = [[] for _ in range(4)]

        with torch.no_grad():
            for feat, mask, labels in val_loader:
                feat   = feat.to(device)
                mask   = mask.to(device)
                labels = labels.to(device)

                preds = model(feat, mask)
                loss, _ = criterion(preds, labels)
                val_losses.append(loss.item())

                task_keys = ["instability", "war", "terrorism", "financial"]
                for t, key in enumerate(task_keys):
                    all_preds [t].extend(preds[key].cpu().numpy().tolist())
                    all_labels[t].extend(labels[:, t].cpu().numpy().tolist())

        mean_val_loss = np.mean(val_losses)

        # AUC per task (binarize at 0.5 since labels are continuous)
        aucs = []
        for t in range(4):
            p  = np.array(all_preds[t])
            lb = (np.array(all_labels[t]) >= 0.5).astype(int)
            try:
                if lb.sum() > 0 and (1 - lb).sum() > 0:
                    aucs.append(roc_auc_score(lb, p))
                else:
                    aucs.append(float("nan"))
            except Exception:
                aucs.append(float("nan"))

        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(mean_val_loss)
        history["val_auc"].append(aucs)
        history["task_losses"].append(mean_task_loss.tolist())

        auc_strs = [f"{a:.4f}" if not math.isnan(a) else "  N/A " for a in aucs]
        print(
            f"  {epoch:>5}  {mean_train_loss:>10.4f}  {mean_val_loss:>9.4f}  "
            f"{auc_strs[0]:>8}  {auc_strs[1]:>7}  {auc_strs[2]:>8}  {auc_strs[3]:>7}"
        )

    print("  " + "-" * 72)

    # Final evaluation pass
    history["final_preds"]  = all_preds
    history["final_labels"] = all_labels
    history["model"] = model

    return history


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_metrics(history: dict) -> None:
    print()
    print("=" * 72)
    print("  FINAL EVALUATION (validation set, last epoch)")
    print("=" * 72)

    preds  = history["final_preds"]
    labels = history["final_labels"]

    rows = []
    for t, name in enumerate(TASK_NAMES):
        p  = np.array(preds[t])
        lb = np.array(labels[t])
        lb_bin = (lb >= 0.5).astype(int)

        mae = float(np.mean(np.abs(p - lb)))
        mse = float(np.mean((p - lb) ** 2))

        try:
            if lb_bin.sum() > 0 and (1 - lb_bin).sum() > 0:
                auc = roc_auc_score(lb_bin, p)
            else:
                auc = float("nan")
        except Exception:
            auc = float("nan")

        # Composite risk for this task: level accuracy
        pred_level  = np.digitize(p,  [0.25, 0.50, 0.75])  # 0-3
        label_level = np.digitize(lb, [0.25, 0.50, 0.75])
        level_acc   = float((pred_level == label_level).mean())

        rows.append((name, mae, mse, auc, level_acc))

    col_w = [14, 8, 8, 8, 10]
    fmt_h = "  {:<14}  {:>8}  {:>8}  {:>8}  {:>10}"
    fmt_r = "  {:<14}  {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>10.4f}"

    print(fmt_h.format("Task", "MAE", "MSE", "AUC", "Level Acc"))
    print("  " + "-" * 56)
    for row in rows:
        name, mae, mse, auc, lacc = row
        auc_s = f"{auc:.4f}" if not math.isnan(auc) else "   N/A"
        print(f"  {name:<14}  {mae:>8.4f}  {mse:>8.4f}  {auc_s:>8}  {lacc:>10.4f}")

    print("  " + "-" * 56)

    # Composite risk score
    all_p  = np.stack(preds,  axis=1)   # (N, 4)
    all_lb = np.stack(labels, axis=1)   # (N, 4)
    w = np.array([0.4, 0.3, 0.2, 0.1])
    comp_p  = all_p  @ w
    comp_lb = all_lb @ w

    comp_mae   = float(np.mean(np.abs(comp_p - comp_lb)))
    comp_mse   = float(np.mean((comp_p - comp_lb) ** 2))
    comp_level_p  = np.digitize(comp_p,  [0.25, 0.50, 0.75])
    comp_level_lb = np.digitize(comp_lb, [0.25, 0.50, 0.75])
    comp_level_acc = float((comp_level_p == comp_level_lb).mean())

    try:
        corr = float(np.corrcoef(comp_p, comp_lb)[0, 1])
    except Exception:
        corr = float("nan")

    print(f"  {'COMPOSITE RISK':<14}  {comp_mae:>8.4f}  {comp_mse:>8.4f}  {'N/A':>8}  {comp_level_acc:>10.4f}")
    print(f"\n  Composite Pearson r : {corr:.4f}")

    # Training curve summary
    print()
    print("  Training loss  : " + "  ".join(f"{v:.4f}" for v in history["train_loss"]))
    print("  Val loss       : " + "  ".join(f"{v:.4f}" for v in history["val_loss"]))

    mean_aucs = [
        np.nanmean([history["val_auc"][e][t] for e in range(len(history["val_auc"]))])
        for t in range(4)
    ]
    print("  Mean val AUCs  : " + "  ".join(
        f"{TASK_NAMES[t]}={v:.4f}" for t, v in enumerate(mean_aucs)
    ))
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train HybridRiskTransformer on real GDELT data"
    )
    p.add_argument("--start",      default="2023-01-01",
                   help="Start date YYYY-MM-DD  (default: 2023-01-01)")
    p.add_argument("--end",        default="2024-12-31",
                   help="End date   YYYY-MM-DD  (default: 2024-12-31)")
    p.add_argument("--interval",   type=int, default=14,
                   help="Download 1 file every N days (default: 14 = bi-weekly)")
    p.add_argument("--seq-len",    type=int, default=0,
                   help="Sequence length (0 = auto: total_dates // 4)")
    p.add_argument("--epochs",     type=int, default=2,
                   help="Training epochs (default: 2)")
    p.add_argument("--batch",      type=int, default=64,
                   help="Batch size (default: 64)")
    p.add_argument("--lr",         type=float, default=3e-4,
                   help="Peak learning rate (default: 3e-4)")
    p.add_argument("--min-events", type=int, default=5,
                   help="Min events per country per day to include (default: 5)")
    p.add_argument("--cache-dir",  default="data/real_cache",
                   help="Cache directory for per-day aggregated features")
    p.add_argument("--cache-only", action="store_true",
                   help="Skip download; use only already-cached files")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # ---- setup ----
    start_dt = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_dt   = datetime.strptime(args.end,   "%Y-%m-%d").date()

    cache_dir = PROJECT_ROOT / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- sample dates ----
    sample_dates: List[date] = []
    cur = start_dt
    while cur <= end_dt:
        sample_dates.append(cur)
        cur += timedelta(days=args.interval)

    n_dates  = len(sample_dates)
    seq_len  = args.seq_len if args.seq_len > 0 else max(4, n_dates // 4)
    seq_len  = min(seq_len, n_dates)

    print()
    print("=" * 72)
    print("  GLDT  --  Real Data Training (GDELT v1)")
    print("=" * 72)
    print(f"  Date range  : {start_dt}  to  {end_dt}")
    print(f"  Sample dates: {n_dates}  (every {args.interval} days)")
    print(f"  Seq length  : {seq_len}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch}")
    print(f"  Cache dir   : {cache_dir}")
    print(f"  Device      : {device}")
    print()

    if args.cache_only:
        print("  --cache-only: skipping downloads, using existing cache files only.")
        # Override download by patching download_and_parse to always return None
        import scripts.train_real_data as _self
        _self.download_and_parse = lambda d, **kw: None

    # ---- build time series ----
    print("  Building country feature time series ...")
    t0 = time.monotonic()
    country_ts = build_country_timeseries(
        sample_dates,
        cache_dir=cache_dir,
        min_events=args.min_events,
        verbose=True,
    )
    elapsed = time.monotonic() - t0

    n_countries = len(country_ts)
    print(f"  Countries with data : {n_countries}")
    print(f"  Time elapsed        : {elapsed:.1f}s")

    if n_countries == 0:
        print()
        print("  ERROR: No data loaded. Check your internet connection or try:")
        print("    python scripts/train_real_data.py --interval 7")
        sys.exit(1)

    # Show top-10 most data-rich countries
    coverage = {cc: int((ts.sum(axis=1) > 0).sum()) for cc, ts in country_ts.items()}
    top10 = sorted(coverage.items(), key=lambda x: -x[1])[:10]
    print()
    print("  Top countries by coverage (non-zero days):")
    for cc, days in top10:
        bar = "#" * int(days / n_dates * 30)
        print(f"    {cc:4s}  {days:3d}/{n_dates}  {bar}")

    # ---- train ----
    print()
    history = train(
        country_ts=country_ts,
        seq_len=seq_len,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )

    # ---- report ----
    report_metrics(history)

    # ---- save model ----
    model_path = PROJECT_ROOT / "models" / "real_data_model.pt"
    torch.save(history["model"].state_dict(), model_path)
    print(f"\n  Model saved to: {model_path}")
    print()


if __name__ == "__main__":
    main()
