# -*- coding: utf-8 -*-
"""
Evaluate HybridRiskTransformer on a held-out test split.

Metrics
-------
Per task (instability, war, terrorism, financial):
    AUC-ROC, AUC-PR, MAE, MSE, F1 at threshold 0.5

Composite risk_score:
    MAE vs label-weighted composite (0.4*I + 0.3*W + 0.2*T + 0.1*F)
    Skill score vs naive mean-label baseline

MC-Dropout confidence calibration:
    ECE (expected calibration error) binned by predicted confidence
    Coverage at 1-sigma and 2-sigma error thresholds

Data sources (auto-detected, or override with flags)
-----------------------------------------------------
1. DB mode (default): MultiTaskRiskDataset from country_multitask_labels.
2. Parquet mode (--parquet-cache or fallback when DB labels absent):
   Loads bi-weekly feature snapshots from data/real_cache/*.parquet and
   derives proxy labels via the same compute_proxy_labels() used by
   train_real_data.py. Works with the real_data_model.pt checkpoint.

Checkpoint formats supported
-----------------------------
- Full format (HybridRiskTransformer.save): dict with "state_dict" + "config".
- Raw state_dict (train_real_data.py output): OrderedDict; seq_len inferred
  from the positional-encoding buffer shape.

Usage:
    python scripts/eval_risk_transformer.py
    python scripts/eval_risk_transformer.py --model-path models/real_data_model.pt
    python scripts/eval_risk_transformer.py --model-path models/checkpoints/risk_model_best.pt \\
        --test-start 2025-06-01 --test-end 2026-03-22
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("eval_risk_transformer")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "configs" / "config.yaml") as f:
    _cfg = yaml.safe_load(f)
_db = _cfg["database"]
DSN = (
    f"postgresql://{_db['user']}:{_db['password']}"
    f"@{_db['host']}:{_db['port']}/{_db['name']}"
)

DEFAULT_MODEL_PATH  = str(ROOT / "models" / "checkpoints" / "risk_model_best.pt")
DEFAULT_TEST_START  = date(2025, 6, 1)
DEFAULT_TEST_END    = date(2026, 3, 22)
DEFAULT_PARQUET_DIR = ROOT / "data" / "real_cache"
RESULTS_PATH        = ROOT / "evaluation" / "results" / "transformer_eval.json"

TASKS = ["instability", "war", "terrorism", "financial"]
RISK_WEIGHTS = {"instability": 0.40, "war": 0.30, "terrorism": 0.20, "financial": 0.10}


# ---------------------------------------------------------------------------
# Model loading (supports both checkpoint formats)
# ---------------------------------------------------------------------------

def load_model(path: str, device: str) -> tuple["HybridRiskTransformer", int]:
    """
    Load HybridRiskTransformer from either format.
    Returns (model, seq_len).
    """
    from models.risk_model import HybridRiskTransformer

    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        # Full format written by HybridRiskTransformer.save()
        cfg    = ckpt["config"]
        model  = HybridRiskTransformer(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        seq_len = cfg.get("seq_len", _cfg["model"]["sequence_length"])
    else:
        # Raw state_dict from train_real_data.py
        # Infer seq_len from positional-encoding buffer: pe shape is (1, seq_len+10, d_model)
        pe       = ckpt["pos_enc.pe"]          # shape (1, max_len, d_model)
        seq_len  = int(pe.shape[1]) - 10
        d_model  = int(pe.shape[2])
        in_feats = int(ckpt["input_proj.0.weight"].shape[1])
        model    = HybridRiskTransformer(
            num_features=in_feats,
            d_model=d_model,
            seq_len=seq_len,
        )
        model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()
    logger.info(
        "Loaded HybridRiskTransformer  params=%d  seq_len=%d",
        model.parameter_count(), seq_len,
    )
    return model, seq_len


# ---------------------------------------------------------------------------
# Parquet-based dataset (mirrors train_real_data.py RealGDELTDataset)
# ---------------------------------------------------------------------------

def _compute_proxy_labels(window: np.ndarray) -> np.ndarray:
    """Derive 4-dim proxy labels from a (seq_len, 7) feature window."""
    tail = window[-min(5, len(window)):]
    protest, violence, diplo, econ, terror, tone_neg, goldstein = (
        tail[:, i].mean() for i in range(7)
    )
    conflict    = 1.0 - goldstein
    instability = 0.4 * protest + 0.3 * violence + 0.2 * diplo + 0.1 * conflict
    war         = 0.5 * violence + 0.3 * diplo    + 0.2 * conflict
    terrorism   = 0.7 * terror  + 0.2 * violence  + 0.1 * tone_neg
    econ_amp    = min(1.0, econ * 2.5)
    financial   = 0.7 * econ_amp + 0.2 * tone_neg + 0.1 * conflict
    return np.clip([instability, war, terrorism, financial], 0.0, 1.0).astype(np.float32)


class ParquetWindowDataset(Dataset):
    """
    Sliding-window dataset built from ALL bi-weekly parquet snapshot files.

    All files are used to build per-country time series (so seq_len is
    always satisfiable).  Only windows whose END index falls in the last
    `test_fraction` of the time series are returned -- this is the held-out
    test set.
    """

    def __init__(
        self,
        all_files: list[Path],
        seq_len: int,
        test_fraction: float = 0.20,
        stride: int = 1,
    ):
        feat_cols = [f"f{i}" for i in range(7)]

        # Build per-country time series from ALL files (chronological order)
        country_ts: dict[str, list[np.ndarray]] = {}
        for f in sorted(all_files):
            df = pd.read_parquet(f)
            for _, row in df.iterrows():
                c = str(row["country"])
                vals = row[feat_cols].values.astype(np.float32)
                country_ts.setdefault(c, []).append(vals)

        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for ts in country_ts.values():
            arr = np.stack(ts)          # (T, 7)
            T   = len(arr)
            if T < seq_len:
                continue
            # Only windows ending in the last test_fraction of snapshots
            test_start_idx = max(seq_len - 1, int(T * (1.0 - test_fraction)))
            for end in range(test_start_idx, T, stride):
                window = arr[end - seq_len + 1 : end + 1]   # (seq_len, 7)
                mask   = (window.sum(axis=1) > 0).astype(np.float32)
                labels = _compute_proxy_labels(window)
                self.samples.append((window, mask, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        window, mask, labels = self.samples[idx]
        return {
            "features": torch.from_numpy(window),
            "mask":     torch.from_numpy(mask),
            "labels":   torch.from_numpy(labels),
        }


def build_parquet_loader(
    cache_dir: Path,
    seq_len: int,
    batch_size: int,
    test_fraction: float = 0.20,
) -> DataLoader:
    """
    Use ALL parquet files for building time series; return only the last
    `test_fraction` of windows per country as the evaluation set.
    """
    all_files = sorted(cache_dir.glob("*_features.parquet"))
    if not all_files:
        raise FileNotFoundError(f"No parquet files found in {cache_dir}")

    n_test = max(1, int(len(all_files) * test_fraction))
    logger.info(
        "Parquet cache: %d total files; last %d (~%.0f%%) form test windows",
        len(all_files), n_test, test_fraction * 100,
    )

    ds = ParquetWindowDataset(
        all_files, seq_len=seq_len, test_fraction=test_fraction, stride=1
    )
    logger.info("Parquet dataset: %d test windows  (seq_len=%d)", len(ds), seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _auc_roc(labels: np.ndarray, preds: np.ndarray, threshold: float = 0.5) -> float:
    from sklearn.metrics import roc_auc_score
    binary = (labels >= threshold).astype(int)
    if binary.max() == binary.min():
        return float("nan")
    return float(roc_auc_score(binary, preds))


def _auc_pr(labels: np.ndarray, preds: np.ndarray, threshold: float = 0.5) -> float:
    from sklearn.metrics import average_precision_score
    binary = (labels >= threshold).astype(int)
    if binary.max() == binary.min():
        return float("nan")
    return float(average_precision_score(binary, preds))


def _f1(labels: np.ndarray, preds: np.ndarray, threshold: float = 0.5) -> float:
    from sklearn.metrics import f1_score
    binary_preds = (preds >= threshold).astype(int)
    binary_labels = (labels >= threshold).astype(int)
    return float(f1_score(binary_labels, binary_preds, zero_division=0))


def _ece(confidences: np.ndarray, errors: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: |confidence - accuracy| weighted by bin size."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        # "accuracy" = fraction of samples with small error (< 0.1)
        bin_acc  = float((errors[mask] < 0.10).mean())
        ece += float(mask.mean()) * abs(bin_conf - bin_acc)
    return ece


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model: "HybridRiskTransformer",
    loader: DataLoader,
    device: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Single deterministic forward pass over the loader.

    Returns:
        preds:  {task: np.ndarray shape (N,)}  -- mean predictions
        labels: np.ndarray shape (N, 4)         -- ground-truth labels
    """
    from models.risk_model import HybridRiskTransformer  # noqa: F401

    model.eval()
    all_preds  = {t: [] for t in TASKS}
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)   # (B, T, F)
            mask     = batch["mask"].to(device)        # (B, T)
            lab      = batch["labels"].cpu().numpy()   # (B, 4)

            out = model(features, mask)
            for i, t in enumerate(TASKS):
                all_preds[t].append(out[t].cpu().numpy())
            all_labels.append(lab)

    return (
        {t: np.concatenate(v) for t, v in all_preds.items()},
        np.concatenate(all_labels, axis=0),
    )


def run_mc_inference(
    model: "HybridRiskTransformer",
    loader: DataLoader,
    device: str,
    n_passes: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    MC-Dropout inference.

    Returns:
        preds:       {task: mean pred (N,)}
        confidences: (N,)   1 - Var(risk_score across passes)
        labels:      (N, 4)
    """
    all_preds_mc  = {t: [] for t in TASKS}
    all_conf      = []
    all_labels    = []

    for batch in loader:
        features = batch["features"].to(device)
        mask     = batch["mask"].to(device)
        lab      = batch["labels"].cpu().numpy()

        out = model.predict_with_confidence(features, mask, n_passes=n_passes)
        for t in TASKS:
            all_preds_mc[t].append(out[t].cpu().numpy())
        all_conf.append(out["confidence"].cpu().numpy())
        all_labels.append(lab)

    return (
        {t: np.concatenate(v) for t, v in all_preds_mc.items()},
        np.concatenate(all_conf),
        np.concatenate(all_labels, axis=0),
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def compute_metrics(
    preds: dict[str, np.ndarray],
    labels: np.ndarray,
) -> dict:
    """Compute per-task and composite metrics."""
    results = {}

    # Per-task
    for i, task in enumerate(TASKS):
        p = preds[task]
        l = labels[:, i]
        results[task] = {
            "n":       int(len(p)),
            "auc_roc": round(_auc_roc(l, p), 4),
            "auc_pr":  round(_auc_pr(l, p), 4),
            "mae":     round(float(np.mean(np.abs(p - l))), 4),
            "mse":     round(float(np.mean((p - l) ** 2)), 4),
            "f1_0.5":  round(_f1(l, p, 0.5), 4),
        }

    # Composite risk score
    composite_pred  = sum(RISK_WEIGHTS[t] * preds[t] for t in TASKS)
    composite_label = sum(RISK_WEIGHTS[t] * labels[:, i] for i, t in enumerate(TASKS))

    baseline_pred   = np.full_like(composite_pred, composite_label.mean())
    baseline_mae    = float(np.mean(np.abs(baseline_pred - composite_label)))
    model_mae       = float(np.mean(np.abs(composite_pred - composite_label)))
    skill           = (baseline_mae - model_mae) / baseline_mae if baseline_mae > 0 else 0.0

    results["composite_risk"] = {
        "n":             int(len(composite_pred)),
        "mae":           round(model_mae, 4),
        "mse":           round(float(np.mean((composite_pred - composite_label) ** 2)), 4),
        "baseline_mae":  round(baseline_mae, 4),
        "skill_score":   round(skill, 4),
    }

    return results


def print_report(
    metrics: dict,
    mc_metrics: dict | None,
    test_start: date,
    test_end: date,
    model_path: str,
    n_samples: int,
) -> None:
    SEP  = "=" * 70

    print("\n" + SEP)
    print("  GEOPULSE  --  HYBRID RISK TRANSFORMER EVALUATION")
    print(SEP)
    print(f"  Model      : {model_path}")
    print(f"  Test range : {test_start}  ->  {test_end}")
    print(f"  Samples    : {n_samples}")
    print()

    # Per-task table
    print("-- Per-Task Metrics " + "-" * 51)
    hdr = f"  {'Task':<14} {'AUC-ROC':>8} {'AUC-PR':>8} {'MAE':>8} {'MSE':>8} {'F1@0.5':>8}"
    print(hdr)
    print("  " + "-" * 64)
    for task in TASKS:
        m = metrics[task]
        auc_roc = f"{m['auc_roc']:.4f}" if not (isinstance(m['auc_roc'], float) and m['auc_roc'] != m['auc_roc']) else "  n/a"
        auc_pr  = f"{m['auc_pr']:.4f}"  if not (isinstance(m['auc_pr'],  float) and m['auc_pr']  != m['auc_pr'])  else "  n/a"
        print(f"  {task:<14} {auc_roc:>8} {auc_pr:>8} {m['mae']:>8.4f} {m['mse']:>8.4f} {m['f1_0.5']:>8.4f}")

    print()
    print("-- Composite Risk Score " + "-" * 47)
    cr = metrics["composite_risk"]
    skill_sign = "+" if cr["skill_score"] >= 0 else ""
    print(f"  MAE (model)       {cr['mae']:.4f}")
    print(f"  MAE (mean-label)  {cr['baseline_mae']:.4f}")
    print(f"  Skill score       {skill_sign}{cr['skill_score']:.1%}  vs naive mean baseline")
    print()

    if mc_metrics:
        print("-- MC-Dropout Confidence Calibration " + "-" * 33)
        print(f"  ECE (expected calibration error)  {mc_metrics['ece']:.4f}")
        print(f"  Mean confidence                   {mc_metrics['mean_confidence']:.4f}")
        print(f"  Coverage @ |err| < 0.10           {mc_metrics['coverage_0.10']:.1%}")
        print(f"  Coverage @ |err| < 0.20           {mc_metrics['coverage_0.20']:.1%}")
        print()

    print("-- Interpretation " + "-" * 52)
    best_task  = min(TASKS, key=lambda t: metrics[t]["mae"])
    worst_task = max(TASKS, key=lambda t: metrics[t]["mae"])
    avg_auc    = np.nanmean([metrics[t]["auc_roc"] for t in TASKS])

    if cr["skill_score"] > 0.05:
        print(f"  [+] Model beats naive baseline by {cr['skill_score']:.1%} on composite risk.")
    elif cr["skill_score"] > 0:
        print("  [~] Marginal improvement over baseline -- consider more training data.")
    else:
        print("  [-] Model worse than naive baseline -- check label quality or retrain.")

    print(f"  Mean AUC-ROC across tasks : {avg_auc:.4f}")
    print(f"  Best task  : {best_task}   (MAE {metrics[best_task]['mae']:.4f})")
    print(f"  Worst task : {worst_task}  (MAE {metrics[worst_task]['mae']:.4f})")
    print(SEP + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HybridRiskTransformer")
    parser.add_argument("--model-path",   default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-start",   default=str(DEFAULT_TEST_START))
    parser.add_argument("--test-end",     default=str(DEFAULT_TEST_END))
    parser.add_argument("--batch-size",   type=int, default=128)
    parser.add_argument("--mc-passes",    type=int, default=30,
                        help="MC-Dropout forward passes for confidence eval (0 = skip)")
    parser.add_argument("--stride",       type=int, default=14,
                        help="Days between test windows in DB mode (default 14)")
    parser.add_argument("--parquet-cache", default=None,
                        help="Path to parquet cache dir; auto-used when DB labels are absent")
    parser.add_argument("--output",       default=str(RESULTS_PATH))
    args = parser.parse_args()

    test_start = date.fromisoformat(args.test_start)
    test_end   = date.fromisoformat(args.test_end)

    # ------------------------------------------------------------------
    # Resolve model path: fall back to real_data_model.pt if default missing
    # ------------------------------------------------------------------
    model_path = Path(args.model_path)
    if not model_path.exists():
        fallback = ROOT / "models" / "real_data_model.pt"
        if fallback.exists():
            logger.warning(
                "Checkpoint not found at %s -- falling back to %s",
                model_path, fallback,
            )
            model_path = fallback
        else:
            logger.error(
                "No checkpoint found at %s or %s. Train first with:\n"
                "  python scripts/train_real_data.py --cache-only",
                model_path, fallback,
            )
            sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, seq_len = load_model(str(model_path), device)

    # ------------------------------------------------------------------
    # Build test loader: DB -> parquet fallback -> error
    # ------------------------------------------------------------------
    loader = None
    data_source = "unknown"

    # Try DB first (unless parquet-cache explicitly given)
    if args.parquet_cache is None:
        try:
            from models.multitask_dataset import MultiTaskRiskDataset, multitask_collate_fn
            logger.info("Building DB test dataset %s -> %s ...", test_start, test_end)
            test_ds = MultiTaskRiskDataset(
                dsn=DSN,
                start_date=test_start,
                end_date=test_end,
                seq_len=seq_len,
                stride=args.stride,
            )
            if len(test_ds) > 0:
                loader = DataLoader(
                    test_ds, batch_size=args.batch_size,
                    shuffle=False, num_workers=0,
                    collate_fn=multitask_collate_fn,
                )
                data_source = "db"
                logger.info("DB test set: %d samples", len(test_ds))
            else:
                logger.warning("DB multitask_labels empty -- switching to parquet cache.")
        except Exception as exc:
            logger.warning("DB dataset failed (%s) -- switching to parquet cache.", exc)

    if loader is None:
        cache_dir = Path(args.parquet_cache) if args.parquet_cache else DEFAULT_PARQUET_DIR
        if not cache_dir.exists():
            logger.error("Parquet cache not found at %s", cache_dir)
            sys.exit(1)
        loader = build_parquet_loader(cache_dir, seq_len, args.batch_size)
        data_source = f"parquet:{cache_dir}"
        logger.info("Using parquet cache: %s", cache_dir)

    n_samples = len(loader.dataset)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Deterministic evaluation
    # ------------------------------------------------------------------
    logger.info("Running deterministic inference ...")
    preds, labels = run_inference(model, loader, device)
    metrics = compute_metrics(preds, labels)

    # ------------------------------------------------------------------
    # MC-Dropout calibration
    # ------------------------------------------------------------------
    mc_metrics = None
    if args.mc_passes > 0:
        logger.info("Running MC-Dropout inference (%d passes) ...", args.mc_passes)
        mc_preds, confidences, mc_labels = run_mc_inference(model, loader, device, args.mc_passes)

        composite_pred  = sum(RISK_WEIGHTS[t] * mc_preds[t] for t in TASKS)
        composite_label = sum(RISK_WEIGHTS[t] * mc_labels[:, i] for i, t in enumerate(TASKS))
        errors          = np.abs(composite_pred - composite_label)

        mc_metrics = {
            "ece":              round(_ece(confidences, errors), 4),
            "mean_confidence":  round(float(confidences.mean()), 4),
            "coverage_0.10":    round(float((errors < 0.10).mean()), 4),
            "coverage_0.20":    round(float((errors < 0.20).mean()), 4),
        }
        metrics["mc_calibration"] = mc_metrics

    # ------------------------------------------------------------------
    # Report + save
    # ------------------------------------------------------------------
    print_report(metrics, mc_metrics, test_start, test_end, str(model_path), n_samples)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_path":  str(model_path),
        "data_source": data_source,
        "test_start":  str(test_start),
        "test_end":    str(test_end),
        "n_samples":   n_samples,
        "metrics":     metrics,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
