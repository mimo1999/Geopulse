"""
Conformal interval calibration for GeoPulse forecasts.

Problem
-------
MC-Dropout produces *epistemic* uncertainty only — the variance of the model's
mean prediction under dropout.  Empirically this gives sigma ~ 0.026 while the
true forecast residual spread needs sigma ~ 0.20, so the +-1.28 sigma bands cover
only ~13% of actuals instead of the nominal 80%.

Fix
---
Split-conformal calibration.  Given a held-out set of residuals
r_i = |actual_i - predicted_i|, the (1 - alpha) conformal quantile

    q = Quantile( {r_i}, ceil((n+1)(1-alpha)) / n )

defines a band  predicted +- q  that is guaranteed (under exchangeability) to
cover the true value with probability >= 1 - alpha on future data.

We fit one quantile per horizon step (residual spread can differ by horizon),
with an optional per-tier breakdown for conditional coverage diagnostics.

This requires NO retraining — it consumes the residuals already produced by the
walk-forward backtester.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from evaluation.metrics import risk_tier

logger = logging.getLogger("evaluation.calibration")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """
    Split-conformal quantile with finite-sample correction.

    Args:
        residuals: 1-D array of nonconformity scores (|actual - predicted|).
        alpha:     Miscoverage rate (e.g. 0.20 for an 80% interval).

    Returns:
        Half-width q such that  pred +- q  has >= (1 - alpha) coverage.
    """
    residuals = residuals[~np.isnan(residuals)]
    n = len(residuals)
    if n == 0:
        return float("nan")
    # Rank of the conformal quantile with the (n+1) finite-sample correction.
    rank = math.ceil((n + 1) * (1 - alpha))
    rank = min(rank, n)                      # clip — if rank > n, use the max
    return float(np.sort(residuals)[rank - 1])


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

@dataclass
class ConformalCalibrator:
    """
    Holds per-horizon (and optional per-tier) conformal half-widths.

    Usage::

        cal = ConformalCalibrator(alpha=0.20)
        cal.fit(fold_log_calibration)      # list of prediction dicts
        lo, hi = cal.interval(pred=0.42, horizon_days=28)
        cal.save("evaluation/results/conformal_quantiles.json")
    """

    alpha: float = 0.20
    q_by_horizon: dict[int, float] = field(default_factory=dict)
    q_by_horizon_tier: dict[str, float] = field(default_factory=dict)  # key "h:tier"
    q_global: float = float("nan")
    n_calibration: int = 0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, fold_log: list[dict]) -> "ConformalCalibrator":
        """Fit conformal quantiles from a list of prediction records.

        Each record needs: predicted, actual, horizon_days, origin_risk.
        """
        rows = [p for p in fold_log if p.get("actual") is not None]
        self.n_calibration = len(rows)
        if not rows:
            logger.warning("No calibration rows with actuals")
            return self

        resid = np.array([abs(p["actual"] - p["predicted"]) for p in rows])
        horizons = np.array([p["horizon_days"] for p in rows])
        tiers = np.array([risk_tier(float(p["origin_risk"])) for p in rows])

        self.q_global = conformal_quantile(resid, self.alpha)

        for h in sorted(set(horizons.tolist())):
            mask = horizons == h
            self.q_by_horizon[int(h)] = conformal_quantile(resid[mask], self.alpha)
            for t in set(tiers[mask].tolist()):
                tmask = mask & (tiers == t)
                if tmask.sum() >= 30:    # only trust per-tier with enough samples
                    self.q_by_horizon_tier[f"{int(h)}:{t}"] = conformal_quantile(
                        resid[tmask], self.alpha
                    )

        logger.info(
            "Calibrated on %d residuals: global q=%.4f, per-horizon=%s",
            self.n_calibration, self.q_global,
            {h: round(q, 4) for h, q in self.q_by_horizon.items()},
        )
        return self

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def half_width(
        self,
        horizon_days: int,
        origin_risk: Optional[float] = None,
    ) -> float:
        """Return the conformal half-width for a horizon (and tier if given)."""
        if origin_risk is not None:
            key = f"{horizon_days}:{risk_tier(float(origin_risk))}"
            if key in self.q_by_horizon_tier:
                return self.q_by_horizon_tier[key]
        if horizon_days in self.q_by_horizon:
            return self.q_by_horizon[horizon_days]
        return self.q_global

    def interval(
        self,
        pred: float,
        horizon_days: int,
        origin_risk: Optional[float] = None,
    ) -> tuple[float, float]:
        """Return calibrated (lower, upper) bounds, clamped to [0, 1]."""
        q = self.half_width(horizon_days, origin_risk)
        lo = max(0.0, pred - q)
        hi = min(1.0, pred + q)
        return lo, hi

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "coverage_target": round(1 - self.alpha, 3),
            "n_calibration": self.n_calibration,
            "q_global": round(self.q_global, 5),
            "q_by_horizon": {str(k): round(v, 5) for k, v in self.q_by_horizon.items()},
            "q_by_horizon_tier": {k: round(v, 5) for k, v in self.q_by_horizon_tier.items()},
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Conformal quantiles saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ConformalCalibrator":
        with open(path) as f:
            d = json.load(f)
        cal = cls(alpha=d["alpha"])
        cal.q_global = d["q_global"]
        cal.q_by_horizon = {int(k): v for k, v in d["q_by_horizon"].items()}
        cal.q_by_horizon_tier = dict(d.get("q_by_horizon_tier", {}))
        cal.n_calibration = d.get("n_calibration", 0)
        return cal


# ---------------------------------------------------------------------------
# Evaluation: chronological split-conformal to prove out-of-sample coverage
# ---------------------------------------------------------------------------

def evaluate_coverage(
    fold_log: list[dict],
    calibrator: ConformalCalibrator,
) -> dict:
    """Measure coverage and mean interval width of calibrated bands."""
    rows = [p for p in fold_log if p.get("actual") is not None]
    if not rows:
        return {}

    covered, widths = [], []
    by_horizon: dict[int, list[bool]] = {}
    by_tier: dict[str, list[bool]] = {}

    for p in rows:
        lo, hi = calibrator.interval(p["predicted"], p["horizon_days"], p["origin_risk"])
        ok = lo <= p["actual"] <= hi
        covered.append(ok)
        widths.append(hi - lo)
        by_horizon.setdefault(p["horizon_days"], []).append(ok)
        by_tier.setdefault(risk_tier(float(p["origin_risk"])), []).append(ok)

    return {
        "n": len(rows),
        "coverage": round(float(np.mean(covered)), 4),
        "mean_width": round(float(np.mean(widths)), 4),
        "coverage_by_horizon": {
            h: round(float(np.mean(v)), 4) for h, v in sorted(by_horizon.items())
        },
        "coverage_by_tier": {
            t: round(float(np.mean(v)), 4) for t, v in by_tier.items()
        },
    }
