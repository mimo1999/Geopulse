"""
Evaluation metrics for GeoPulse walk-forward backtesting.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HorizonMetrics:
    horizon_days: int
    n_samples: int
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float
    ci_coverage: float          # fraction of actuals inside [lower, upper]
    skill_score: float          # (MAE_baseline - MAE_model) / MAE_baseline


@dataclass
class BacktestMetrics:
    model_name: str
    n_countries: int
    n_folds: int
    horizons: list[HorizonMetrics]
    overall_mae: float
    overall_rmse: float
    overall_skill_score: float
    per_tier: dict = field(default_factory=dict)  # {tier: {mae, rmse, n}}


def compute_horizon_metrics(
    actuals: np.ndarray,          # shape (N,)
    predictions: np.ndarray,      # shape (N,)
    baselines: np.ndarray,        # shape (N,)  — naive carry-forward
    lowers: np.ndarray,           # shape (N,)  — CI lower bound
    uppers: np.ndarray,           # shape (N,)  — CI upper bound
    prev_actuals: np.ndarray,     # shape (N,)  — value at forecast origin (for direction)
    horizon_days: int,
) -> HorizonMetrics:
    mask = ~np.isnan(actuals) & ~np.isnan(predictions)
    a = actuals[mask]
    p = predictions[mask]
    b = baselines[mask]
    lo = lowers[mask]
    hi = uppers[mask]
    prev = prev_actuals[mask]

    n = len(a)
    if n == 0:
        return HorizonMetrics(horizon_days, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    errors = np.abs(a - p)
    mae  = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))

    # MAPE — skip near-zero actuals
    nonzero = a > 0.01
    mape = float(np.mean(np.abs((a[nonzero] - p[nonzero]) / a[nonzero]))) * 100 if nonzero.any() else np.nan

    # Directional accuracy: did model predict same direction as actual change?
    actual_dir = np.sign(a - prev)
    pred_dir   = np.sign(p - prev)
    dir_acc = float(np.mean(actual_dir == pred_dir))

    # CI coverage
    ci_cov = float(np.mean((a >= lo) & (a <= hi)))

    # Skill score vs naive baseline
    mae_base = float(np.mean(np.abs(a - b)))
    skill = (mae_base - mae) / mae_base if mae_base > 0 else 0.0

    return HorizonMetrics(
        horizon_days=horizon_days,
        n_samples=n,
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        mape=round(mape, 2) if not np.isnan(mape) else None,
        directional_accuracy=round(dir_acc, 4),
        ci_coverage=round(ci_cov, 4),
        skill_score=round(skill, 4),
    )


def risk_tier(score: float) -> str:
    if score >= 0.80: return "CRITICAL"
    if score >= 0.65: return "HIGH"
    if score >= 0.50: return "ELEVATED"
    if score >= 0.35: return "MODERATE"
    return "LOW"


def compute_per_tier_metrics(
    actuals: np.ndarray,
    predictions: np.ndarray,
    origin_scores: np.ndarray,
) -> dict:
    tiers: dict[str, dict] = {}
    for a, p, o in zip(actuals, predictions, origin_scores):
        if np.isnan(a) or np.isnan(p):
            continue
        t = risk_tier(float(o))
        if t not in tiers:
            tiers[t] = {"actuals": [], "preds": []}
        tiers[t]["actuals"].append(a)
        tiers[t]["preds"].append(p)

    result = {}
    for t, data in tiers.items():
        a = np.array(data["actuals"])
        p = np.array(data["preds"])
        errs = np.abs(a - p)
        result[t] = {
            "n":    len(errs),
            "mae":  round(float(np.mean(errs)), 4),
            "rmse": round(float(np.sqrt(np.mean((a - p) ** 2))), 4),
        }
    return result
