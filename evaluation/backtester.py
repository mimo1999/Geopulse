"""
Walk-forward backtester for GeoPulse risk models.

Strategy
--------
Data:   83 bi-weekly feature snapshots across 317 countries (2023-01-01 → 2026-03-22).
Folds:  Expanding window.  First 30 snapshots form the burn-in; then we step
        forward one snapshot at a time — 53 test folds in total.
Predict: At pivot date T, produce 4-step ahead forecasts (≈ 14, 28, 42, 56 days).
Actual:  The `risk_score` already stored in country_daily_features for those dates.
Baseline: Naive carry-forward — predict the risk_score at T for all horizons.

Outputs
-------
BacktestResults dataclass with per-horizon and per-tier metrics, plus the full
fold-level prediction log for further analysis.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import psycopg2
import psycopg2.extras

from evaluation.metrics import (
    BacktestMetrics,
    HorizonMetrics,
    compute_horizon_metrics,
    compute_per_tier_metrics,
)

logger = logging.getLogger("evaluation.backtester")

HORIZON_DAYS = [14, 28, 42, 56]
BURN_IN_FOLDS = 30          # minimum history before we start evaluating
MIN_COUNTRIES_PER_FOLD = 10 # skip folds with very sparse coverage


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FoldPrediction:
    country: str
    pivot_date: date
    origin_risk: float          # risk_score at T
    horizon_days: int
    target_date: date
    predicted: float
    lower: float
    upper: float
    actual: Optional[float]     # None if the snapshot doesn't exist in DB
    baseline: float             # carry-forward = origin_risk


@dataclass
class BacktestResults:
    run_date: str
    n_countries: int
    n_folds: int
    n_predictions: int
    metrics: BacktestMetrics
    fold_log: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("Results saved to %s", path)


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class WalkForwardBacktester:
    """
    Runs walk-forward evaluation of the EscalationForecasterEngine
    against all countries that have sufficient history.

    Parameters
    ----------
    dsn :           PostgreSQL connection string.
    model_path :    Path to forecaster checkpoint (.pt).
    burn_in :       Number of snapshot dates to skip before evaluating.
    countries :     Restrict to specific country codes; None = all.
    """

    def __init__(
        self,
        dsn: str,
        model_path: Optional[str] = None,
        burn_in: int = BURN_IN_FOLDS,
        countries: Optional[list[str]] = None,
    ):
        self._dsn        = dsn
        self._model_path = model_path
        self._burn_in    = burn_in
        self._countries  = countries
        self._engine     = None   # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BacktestResults:
        t0 = time.monotonic()
        logger.info("Loading snapshot dates and feature data…")

        snapshot_dates = self._fetch_snapshot_dates()
        logger.info("Found %d bi-weekly snapshot dates (%s to %s)",
                    len(snapshot_dates), snapshot_dates[0], snapshot_dates[-1])

        countries = self._countries or self._fetch_countries(snapshot_dates)
        logger.info("Evaluating %d countries", len(countries))

        # Pre-load all features once to avoid per-fold DB round-trips
        feature_matrix = self._load_all_features(countries, snapshot_dates)
        logger.info("Feature matrix loaded: %d entries across %d countries",
                    len(feature_matrix), len(countries))

        # Lazy-init the forecaster engine
        self._init_engine()

        fold_log: list[FoldPrediction] = []
        test_dates = snapshot_dates[self._burn_in:]
        logger.info("Running %d test folds…", len(test_dates))

        for fold_idx, pivot_date in enumerate(test_dates):
            if (fold_idx + 1) % 5 == 0:
                logger.info("  fold %d/%d  %s", fold_idx + 1, len(test_dates), pivot_date)

            # --- Build a batch tensor for all countries present at this pivot ---
            present = [
                c for c in countries
                if feature_matrix.get((c, pivot_date), {}).get("risk_score") is not None
            ]
            if not present:
                continue

            batch_np = np.stack([
                self._build_feature_array(c, pivot_date, snapshot_dates, feature_matrix)
                for c in present
            ])  # shape (B, 90, 7)

            # Run batched forecast
            batch_steps = self._run_forecast_batch(present, pivot_date, batch_np)
            # batch_steps: list of (country, list[step_dict])

            for country, steps in batch_steps:
                origin_risk = feature_matrix[(country, pivot_date)]["risk_score"]
                for step in steps:
                    actual = feature_matrix.get((country, step["target_date"]), {}).get("risk_score")
                    fold_log.append(FoldPrediction(
                        country=country,
                        pivot_date=pivot_date,
                        origin_risk=float(origin_risk),
                        horizon_days=step["horizon_days"],
                        target_date=step["target_date"],
                        predicted=step["risk_score"],
                        lower=step["lower_bound"],
                        upper=step["upper_bound"],
                        actual=actual,
                        baseline=float(origin_risk),
                    ))

        elapsed = time.monotonic() - t0
        logger.info("Backtesting complete in %.1fs — %d predictions", elapsed, len(fold_log))

        metrics = self._aggregate(fold_log, countries, test_dates)
        results = BacktestResults(
            run_date=str(date.today()),
            n_countries=len(countries),
            n_folds=len(test_dates),
            n_predictions=len([p for p in fold_log if p.actual is not None]),
            metrics=metrics,
            fold_log=[self._pred_to_dict(p) for p in fold_log if p.actual is not None],
        )
        return results

    # ------------------------------------------------------------------
    # Forecasting helpers
    # ------------------------------------------------------------------

    def _init_engine(self) -> None:
        from inference.escalation_forecaster import EscalationForecasterEngine
        self._engine = EscalationForecasterEngine(
            dsn=self._dsn,
            model_path=self._model_path,
            seq_len=90,
            mc_passes=5,
        )

    def _run_forecast_batch(
        self,
        countries: list[str],
        pivot: date,
        batch_np: np.ndarray,  # (B, seq_len, features)
    ) -> list[tuple[str, list[dict]]]:
        """Run one batched forward pass for all countries in a fold."""
        from inference.escalation_forecaster import HORIZON_DAYS as HD
        import torch

        if self._engine._model is not None:
            tensor = torch.from_numpy(batch_np)           # (B, T, F)
            B, T, _ = tensor.shape
            mask = torch.ones(B, T)
            out = self._engine._model.predict_with_confidence(tensor, mask, n_passes=self._engine._mc_passes)
            H = self._engine._model._horizon
            results = []
            for b, country in enumerate(countries):
                steps = []
                for h in range(H):
                    offset = HD[h] if h < len(HD) else (h + 1) * 14
                    steps.append({
                        "horizon_days": offset,
                        "target_date":  pivot + timedelta(days=offset),
                        "risk_score":   float(out["risk_score"][b, h].item()),
                        "lower_bound":  float(out["lower_bound"][b, h].item()),
                        "upper_bound":  float(out["upper_bound"][b, h].item()),
                    })
                results.append((country, steps))
            return results
        else:
            # Trend extrapolation fallback — per-country, no batching needed
            return [
                (countries[b], [{
                    "horizon_days": s.step * 14,
                    "target_date":  s.target_date,
                    "risk_score":   s.risk_score,
                    "lower_bound":  s.lower_bound,
                    "upper_bound":  s.upper_bound,
                } for s in self._engine._trend_extrapolation(batch_np[b], pivot)])
                for b in range(len(countries))
            ]

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    def _build_feature_array(
        self,
        country: str,
        pivot: date,
        snapshot_dates: list[date],
        feature_matrix: dict,
    ) -> np.ndarray:
        """Build a (90, 7) float32 array for the 90-day window ending at pivot."""
        from models.dataset import FEATURE_COLUMNS
        seq_len = 90
        start = pivot - timedelta(days=seq_len - 1)
        arr = np.zeros((seq_len, len(FEATURE_COLUMNS)), dtype=np.float32)
        last = None
        for i in range(seq_len):
            d = start + timedelta(days=i)
            row = feature_matrix.get((country, d))
            if row:
                vals = [float(row.get(c, 0) or 0) for c in FEATURE_COLUMNS]
                arr[i] = vals
                last = vals
            elif last is not None:
                arr[i] = last   # forward-fill
        return arr

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _fetch_snapshot_dates(self) -> list[date]:
        with contextlib.closing(psycopg2.connect(self._dsn)) as conn:
            with conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT feature_date FROM country_daily_features "
                    "ORDER BY feature_date ASC"
                )
                return [r[0] for r in cur.fetchall()]

    def _fetch_countries(self, snapshot_dates: list[date]) -> list[str]:
        """Return countries present in at least half the snapshot dates."""
        min_obs = max(self._burn_in, len(snapshot_dates) // 2)
        with contextlib.closing(psycopg2.connect(self._dsn)) as conn:
            with conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT country, COUNT(*) as n FROM country_daily_features "
                    "GROUP BY country HAVING COUNT(*) >= %s ORDER BY n DESC",
                    (min_obs,),
                )
                return [r[0] for r in cur.fetchall()]

    def _load_all_features(
        self,
        countries: list[str],
        snapshot_dates: list[date],
    ) -> dict:
        """
        Load all feature rows into a dict keyed by (country, date)
        for O(1) lookup during fold iteration.
        """
        from models.dataset import FEATURE_COLUMNS
        cols = ", ".join(FEATURE_COLUMNS + ["risk_score"])
        with contextlib.closing(
            psycopg2.connect(self._dsn, cursor_factory=psycopg2.extras.RealDictCursor)
        ) as conn:
            with conn, conn.cursor() as cur:
                cur.execute(
                    f"SELECT country, feature_date, {cols} FROM country_daily_features "
                    f"WHERE country = ANY(%s) ORDER BY feature_date",
                    (countries,),
                )
                rows = cur.fetchall()

        matrix = {}
        for row in rows:
            matrix[(row["country"], row["feature_date"])] = dict(row)
        return matrix

    # ------------------------------------------------------------------
    # Metric aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        fold_log: list[FoldPrediction],
        countries: list[str],
        test_dates: list[date],
    ) -> BacktestMetrics:
        horizon_metrics = []
        all_actuals_arrs, all_preds_arrs, all_baselines_arrs, all_origins_arrs = [], [], [], []

        for h in HORIZON_DAYS:
            preds = [p for p in fold_log if p.horizon_days == h and p.actual is not None]
            if not preds:
                continue
            actuals   = np.array([p.actual       for p in preds])
            predicted = np.array([p.predicted    for p in preds])
            baselines = np.array([p.baseline     for p in preds])
            lowers    = np.array([p.lower        for p in preds])
            uppers    = np.array([p.upper        for p in preds])
            origins   = np.array([p.origin_risk  for p in preds])

            hm = compute_horizon_metrics(
                actuals, predicted, baselines, lowers, uppers, origins, h
            )
            horizon_metrics.append(hm)
            all_actuals_arrs.append(actuals)
            all_preds_arrs.append(predicted)
            all_baselines_arrs.append(baselines)
            all_origins_arrs.append(origins)

        if not all_actuals_arrs:
            overall_mae = overall_rmse = baseline_mae = float("nan")
            overall_skill = 0.0
            per_tier = {}
        else:
            a_all = np.concatenate(all_actuals_arrs)
            p_all = np.concatenate(all_preds_arrs)
            b_all = np.concatenate(all_baselines_arrs)
            o_all = np.concatenate(all_origins_arrs)
            diff  = a_all - p_all
            overall_mae  = float(np.mean(np.abs(diff)))
            overall_rmse = float(np.sqrt(np.mean(diff ** 2)))
            baseline_mae = float(np.mean(np.abs(a_all - b_all)))
            overall_skill = (baseline_mae - overall_mae) / baseline_mae if baseline_mae > 0 else 0.0
            per_tier = compute_per_tier_metrics(a_all, p_all, o_all)

        return BacktestMetrics(
            model_name="EscalationForecaster-LSTM-v1",
            n_countries=len(countries),
            n_folds=len(test_dates),
            horizons=horizon_metrics,
            overall_mae=round(overall_mae, 4),
            overall_rmse=round(overall_rmse, 4),
            overall_skill_score=round(overall_skill, 4),
            per_tier=per_tier,
        )

    @staticmethod
    def _pred_to_dict(p: FoldPrediction) -> dict:
        return {
            "country":      p.country,
            "pivot_date":   str(p.pivot_date),
            "origin_risk":  round(p.origin_risk, 4),
            "horizon_days": p.horizon_days,
            "target_date":  str(p.target_date),
            "predicted":    round(p.predicted, 4),
            "lower":        round(p.lower, 4),
            "upper":        round(p.upper, 4),
            "actual":       round(p.actual, 4) if p.actual is not None else None,
            "baseline":     round(p.baseline, 4),
            "error":        round(abs(p.actual - p.predicted), 4) if p.actual is not None else None,
        }
