"""
Standalone Phase 2 training on synthetic GDELT-like data.
No database required - generates realistic synthetic country risk windows.

Run:
    python scripts/train_synthetic.py --epochs 5

Synthetic data mirrors the real schema:
  features:  (B, T=90, F=7)  - 7 country daily feature scores (0-1)
  labels:    (B, 4)          - instability, war, terrorism, financial (0-1)

Country populations:
  - 40 stable  countries  ? low labels
  - 25 elevated countries ? mid labels
  - 20 unstable countries ? high labels
  - 15 volatile countries ? mixed/spiking labels
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.risk_model import HybridRiskTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_synthetic")

# -----------------------------------------------------------------------------
# Synthetic Dataset
# -----------------------------------------------------------------------------

FEATURE_NAMES = [
    "protest_score",
    "violence_score",
    "diplomatic_stress",
    "economic_stress",
    "terrorism_score",
    "avg_sentiment",
    "avg_goldstein",
]
NUM_FEATURES = 7
NUM_LABELS   = 4
SEQ_LEN      = 90


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _make_country_profile(stability: str, rng: np.random.Generator) -> dict:
    """
    Return base_mean (F,), base_std (F,), label_mean (4,) for a country regime.

    Feature order: protest, violence, diplomatic_stress, economic, terrorism,
                   sentiment (inverted - higher = negative), goldstein (inverted)
    """
    if stability == "stable":
        base_mean  = np.array([0.05, 0.03, 0.15, 0.08, 0.02, 0.30, 0.25])
        base_std   = np.array([0.02, 0.01, 0.04, 0.03, 0.01, 0.05, 0.05])
        label_mean = np.array([0.12, 0.08, 0.05, 0.10])

    elif stability == "elevated":
        base_mean  = np.array([0.18, 0.12, 0.35, 0.22, 0.08, 0.50, 0.55])
        base_std   = np.array([0.06, 0.04, 0.08, 0.06, 0.03, 0.08, 0.08])
        label_mean = np.array([0.38, 0.28, 0.18, 0.32])

    elif stability == "unstable":
        base_mean  = np.array([0.35, 0.28, 0.58, 0.40, 0.22, 0.68, 0.70])
        base_std   = np.array([0.10, 0.08, 0.12, 0.10, 0.07, 0.10, 0.10])
        label_mean = np.array([0.65, 0.55, 0.40, 0.58])

    else:  # volatile
        base_mean  = np.array([0.25, 0.20, 0.45, 0.30, 0.15, 0.60, 0.62])
        base_std   = np.array([0.15, 0.12, 0.18, 0.14, 0.10, 0.15, 0.15])
        label_mean = np.array([0.50, 0.45, 0.35, 0.45])

    return {"mean": base_mean, "std": base_std, "label_mean": label_mean}


def generate_synthetic_dataset(
    n_countries: int = 100,
    windows_per_country: int = 20,
    seq_len: int = SEQ_LEN,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate (N, T, F) feature tensor and (N, 4) label tensor.

    Features are AR(1) time series per dimension (realistic temporal correlation).
    Labels have noise added to the country-level mean.
    """
    rng = np.random.default_rng(seed)

    # Country regimes
    regimes = (
        ["stable"]   * 40 +
        ["elevated"] * 25 +
        ["unstable"] * 20 +
        ["volatile"] * 15
    )
    # Subsample to n_countries
    if n_countries < len(regimes):
        idx = rng.choice(len(regimes), n_countries, replace=False)
        regimes = [regimes[i] for i in idx]
    else:
        regimes = regimes[:n_countries]

    all_features = []
    all_labels   = []

    for regime in regimes:
        profile = _make_country_profile(regime, rng)
        base_mean  = profile["mean"]      # (F,)
        base_std   = profile["std"]       # (F,)
        label_mean = profile["label_mean"]  # (4,)

        for _ in range(windows_per_country):
            # AR(1) process per feature: x_t = phi * x_{t-1} + noise
            phi   = rng.uniform(0.3, 0.8, size=NUM_FEATURES)
            noise = rng.normal(0, base_std, size=(seq_len, NUM_FEATURES))
            window = np.zeros((seq_len, NUM_FEATURES))
            window[0] = base_mean + noise[0]

            for t in range(1, seq_len):
                window[t] = phi * window[t - 1] + (1 - phi) * base_mean + noise[t]

            # Random spike events for volatile/unstable countries (30% chance)
            if regime in ("unstable", "volatile") and rng.random() < 0.30:
                spike_start = rng.integers(0, seq_len - 7)
                spike_len   = rng.integers(3, 10)
                spike_mag   = rng.uniform(0.2, 0.5, size=NUM_FEATURES)
                window[spike_start:spike_start + spike_len] += spike_mag

            # Clip to [0, 1]
            window = np.clip(window, 0.0, 1.0)

            # Label: country mean + small noise
            label_noise = rng.normal(0, 0.05, size=4)
            label = np.clip(label_mean + label_noise, 0.0, 1.0)

            all_features.append(window)
            all_labels.append(label)

    features = np.array(all_features, dtype=np.float32)  # (N, T, F)
    labels   = np.array(all_labels,   dtype=np.float32)  # (N, 4)

    logger.info(
        "Synthetic dataset: %d samples | features %s | labels %s",
        len(features), features.shape, labels.shape,
    )
    return features, labels


class SyntheticRiskDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features)   # (N, T, F)
        self.labels   = torch.from_numpy(labels)     # (N, 4)
        # All timesteps are valid (no padding in synthetic data)
        self.masks    = torch.ones(len(features), features.shape[1])

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "features": self.features[idx],
            "labels":   self.labels[idx],
            "mask":     self.masks[idx],
        }


# -----------------------------------------------------------------------------
# Loss (mirrors Phase2Trainer)
# -----------------------------------------------------------------------------

class SmoothedBCE(nn.Module):
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.s = smoothing

    def forward(self, pred, target):
        t = target * (1 - self.s) + 0.5 * self.s
        return nn.functional.binary_cross_entropy(pred, t)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = nn.functional.binary_cross_entropy(pred, target, reduction="none")
        p_t = pred * target + (1 - pred) * (1 - target)
        return (self.alpha * (1 - p_t) ** self.gamma * bce).mean()


class MultiTaskLoss(nn.Module):
    TASK_NAMES = ["instability", "war", "terrorism", "financial"]

    def __init__(self):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(4))
        self.losses   = nn.ModuleList([
            SmoothedBCE(0.05),   # instability
            SmoothedBCE(0.05),   # war
            FocalLoss(0.25, 2),  # terrorism  ? focal
            SmoothedBCE(0.05),   # financial
        ])

    def forward(self, preds: dict, targets: torch.Tensor):
        keys  = ["instability", "war", "terrorism", "financial"]
        tasks = torch.stack([
            self.losses[i](preds[keys[i]], targets[:, i])
            for i in range(4)
        ])
        precisions = torch.exp(-self.log_vars)
        total = (precisions * tasks + self.log_vars).sum()
        return total, {k: tasks[i].item() for i, k in enumerate(keys)}


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_metrics(preds_all, targets_all):
    """Compute MSE and AUC (where possible) per task."""
    from sklearn.metrics import roc_auc_score, mean_squared_error
    task_names = ["instability", "war", "terrorism", "financial"]
    results = {}
    for i, name in enumerate(task_names):
        p = preds_all[:, i]
        t = targets_all[:, i]
        mse = float(mean_squared_error(t, p))
        t_bin = (t >= 0.5).astype(int)
        auc = None
        if t_bin.sum() > 0 and t_bin.sum() < len(t_bin):
            try:
                auc = float(roc_auc_score(t_bin, p))
            except Exception:
                pass
        results[name] = {"mse": mse, "auc": auc}
    return results


def mae(preds_all, targets_all):
    return float(np.mean(np.abs(preds_all - targets_all)))


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def train(epochs: int = 5, batch_size: int = 64, seed: int = 42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # -- Data --
    features, labels = generate_synthetic_dataset(
        n_countries=100,
        windows_per_country=25,
        seq_len=SEQ_LEN,
        seed=seed,
    )
    dataset = SyntheticRiskDataset(features, labels)

    n      = len(dataset)
    n_test = max(int(n * 0.15), 1)
    n_val  = max(int(n * 0.15), 1)
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )
    loader_args = dict(batch_size=batch_size, num_workers=0,
                       pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_args)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_args)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_args)

    logger.info(
        "Split - train: %d  val: %d  test: %d  (total: %d)",
        len(train_ds), len(val_ds), len(test_ds), n,
    )

    # -- Model --
    model = HybridRiskTransformer(
        num_features=NUM_FEATURES,
        d_model=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        seq_len=SEQ_LEN,
    ).to(device)
    logger.info("Model parameters: %d", model.parameter_count())

    criterion = MultiTaskLoss().to(device)
    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=3e-4, weight_decay=1e-4,
    )
    total_steps = epochs * len(train_loader)
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, total_steps=total_steps, pct_start=0.1,
    )

    # -- Training loop --
    best_val_loss = float("inf")
    history       = []

    print()
    print("=" * 72)
    print(f"  Phase 2 Risk Model - Synthetic Training  ({epochs} epochs)")
    print("=" * 72)
    print(f"  {'Epoch':<6}  {'Train Loss':<12} {'Val Loss':<12} {'Val MAE':<10} {'LR':<12}  {'Time'}")
    print("-" * 72)

    for epoch in range(1, epochs + 1):
        t0 = time.monotonic()

        # - Train -
        model.train()
        total_train = 0.0
        for batch in train_loader:
            feats   = batch["features"].to(device)
            tgts    = batch["labels"].to(device)
            masks   = batch["mask"].to(device)
            optimizer.zero_grad()
            preds   = model(feats, masks)
            loss, _ = criterion(preds, tgts)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_train += loss.item()

        train_loss = total_train / len(train_loader)

        # - Val -
        model.eval()
        total_val  = 0.0
        all_preds  = []
        all_tgts   = []
        with torch.no_grad():
            for batch in val_loader:
                feats = batch["features"].to(device)
                tgts  = batch["labels"].to(device)
                masks = batch["mask"].to(device)
                preds = model(feats, masks)
                loss, _ = criterion(preds, tgts)
                total_val += loss.item()

                stacked = torch.stack([
                    preds["instability"], preds["war"],
                    preds["terrorism"],   preds["financial"],
                ], dim=1)
                all_preds.append(stacked.cpu().numpy())
                all_tgts.append(tgts.cpu().numpy())

        val_loss  = total_val / len(val_loader)
        preds_arr = np.concatenate(all_preds,  axis=0)
        tgts_arr  = np.concatenate(all_tgts,   axis=0)
        val_mae   = mae(preds_arr, tgts_arr)

        elapsed   = time.monotonic() - t0
        cur_lr    = optimizer.param_groups[0]["lr"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_marker   = " ?"
        else:
            best_marker   = ""

        print(
            f"  {epoch:<6}  {train_loss:<12.4f} {val_loss:<12.4f} "
            f"{val_mae:<10.4f} {cur_lr:<12.2e}  {elapsed:.1f}s{best_marker}"
        )

        metrics = compute_metrics(preds_arr, tgts_arr)
        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_mae":    val_mae,
            "metrics":    metrics,
        })

    print("-" * 72)

    # -- Per-task metrics (last val epoch) --
    last   = history[-1]["metrics"]
    best_e = min(history, key=lambda x: x["val_loss"])

    print()
    print("=" * 72)
    print("  Per-Task Validation Metrics (final epoch)")
    print("-" * 72)
    print(f"  {'Task':<20}  {'MSE':>8}  {'MAE':>8}  {'AUC':>8}")
    print("-" * 72)
    task_names = ["instability", "war", "terrorism", "financial"]
    for name in task_names:
        m   = last[name]
        mse = m["mse"]
        auc = m["auc"]
        # Per-task MAE
        i   = task_names.index(name)
        t_mae = float(np.mean(np.abs(preds_arr[:, i] - tgts_arr[:, i])))
        auc_s = f"{auc:.4f}" if auc is not None else "  N/A  "
        print(f"  {name:<20}  {mse:>8.4f}  {t_mae:>8.4f}  {auc_s:>8}")
    print("-" * 72)
    print(f"  Composite risk MAE (weighted):  {mae(preds_arr, tgts_arr):.4f}")
    print()

    # -- Risk score accuracy --
    print("  Risk Score Quality (composite = 0.4.I + 0.3.W + 0.2.T + 0.1.F)")
    print("-" * 72)
    risk_pred   = (
        0.40 * preds_arr[:, 0]
        + 0.30 * preds_arr[:, 1]
        + 0.20 * preds_arr[:, 2]
        + 0.10 * preds_arr[:, 3]
    )
    risk_true   = (
        0.40 * tgts_arr[:, 0]
        + 0.30 * tgts_arr[:, 1]
        + 0.20 * tgts_arr[:, 2]
        + 0.10 * tgts_arr[:, 3]
    )
    risk_mae    = float(np.mean(np.abs(risk_pred - risk_true)))
    risk_mse    = float(np.mean((risk_pred - risk_true) ** 2))
    risk_corr   = float(np.corrcoef(risk_pred, risk_true)[0, 1])

    print(f"  MAE:         {risk_mae:.4f}")
    print(f"  MSE:         {risk_mse:.4f}")
    print(f"  Pearson r:   {risk_corr:.4f}")

    # Risk-level accuracy (5 buckets: LOW / MOD / ELEV / HIGH / CRIT)
    def risk_bucket(s):
        if s >= 0.80: return 4
        if s >= 0.65: return 3
        if s >= 0.50: return 2
        if s >= 0.35: return 1
        return 0
    buckets_pred = np.array([risk_bucket(s) for s in risk_pred])
    buckets_true = np.array([risk_bucket(s) for s in risk_true])
    exact_acc    = float(np.mean(buckets_pred == buckets_true))
    within1_acc  = float(np.mean(np.abs(buckets_pred - buckets_true) <= 1))
    print(f"  Level accuracy (exact):   {exact_acc:.1%}")
    print(f"  Level accuracy (?1 band): {within1_acc:.1%}")

    # -- Test set --
    print()
    print("-" * 72)
    model.eval()
    test_preds, test_tgts = [], []
    with torch.no_grad():
        for batch in test_loader:
            feats = batch["features"].to(device)
            tgts  = batch["labels"].to(device)
            masks = batch["mask"].to(device)
            preds = model(feats, masks)
            stacked = torch.stack([
                preds["instability"], preds["war"],
                preds["terrorism"],   preds["financial"],
            ], dim=1)
            test_preds.append(stacked.cpu().numpy())
            test_tgts.append(tgts.cpu().numpy())

    test_preds = np.concatenate(test_preds)
    test_tgts  = np.concatenate(test_tgts)
    test_mae_v = mae(test_preds, test_tgts)
    test_risk_p = (0.40*test_preds[:,0] + 0.30*test_preds[:,1] +
                   0.20*test_preds[:,2] + 0.10*test_preds[:,3])
    test_risk_t = (0.40*test_tgts[:,0]  + 0.30*test_tgts[:,1]  +
                   0.20*test_tgts[:,2]  + 0.10*test_tgts[:,3])
    print(f"  Test set MAE:           {test_mae_v:.4f}")
    print(f"  Test risk score MAE:    {float(np.mean(np.abs(test_risk_p - test_risk_t))):.4f}")
    print(f"  Test Pearson r:         {float(np.corrcoef(test_risk_p, test_risk_t)[0,1]):.4f}")
    print("=" * 72)
    print(f"  Best checkpoint at epoch {best_e['epoch']}  (val_loss={best_e['val_loss']:.4f})")
    print("=" * 72)
    print()

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
