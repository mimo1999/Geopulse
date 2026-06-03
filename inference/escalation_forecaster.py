"""
Phase 3: Runtime escalation forecasting engine.

Wraps EscalationForecaster for API and pipeline use.
Loads country features from country_daily_features table,
runs multi-step ahead inference, and persists results to
country_escalation_forecasts.

Each call to forecast() produces H=4 bi-weekly step predictions
(≈ 14, 28, 42, 56 days ahead).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import psycopg2
import psycopg2.extras
import torch

from models.forecaster import EscalationForecaster
from models.dataset import FEATURE_COLUMNS, NUM_FEATURES

logger = logging.getLogger("inference.escalation_forecaster")

HORIZON_DAYS = [14, 28, 42, 56]   # target offset in days per step


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class ForecastStep:
    step: int           # 1-indexed horizon step
    target_date: date
    risk_score: float
    instability: float
    war_probability: float
    terrorism_risk: float
    financial_stress: float
    confidence: float
    variance: float
    lower_bound: float
    upper_bound: float

    def to_dict(self) -> dict:
        return {
            "step":              self.step,
            "target_date":       str(self.target_date),
            "risk_score":        round(self.risk_score, 4),
            "instability":       round(self.instability, 4),
            "war_probability":   round(self.war_probability, 4),
            "terrorism_risk":    round(self.terrorism_risk, 4),
            "financial_stress":  round(self.financial_stress, 4),
            "confidence":        round(self.confidence, 4),
            "variance":          round(self.variance, 4),
            "lower_bound":       round(self.lower_bound, 4),
            "upper_bound":       round(self.upper_bound, 4),
        }


@dataclass
class ForecastResult:
    country:       str
    forecast_date: date
    steps:         list[ForecastStep]
    model_version: str = "v0.3-forecaster"

    def to_dict(self) -> dict:
        return {
            "country":       self.country,
            "forecast_date": str(self.forecast_date),
            "horizon_steps": len(self.steps),
            "forecasts":     [s.to_dict() for s in self.steps],
            "model_version": self.model_version,
        }

    @property
    def risk_trajectory(self) -> list[float]:
        return [s.risk_score for s in self.steps]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EscalationForecasterEngine:
    """
    High-level forecasting engine for pipeline and API use.

    Usage::

        engine = EscalationForecasterEngine(dsn=..., model_path=...)
        result = engine.forecast("UA")
    """

    _FETCH_FEATURES_SQL = """
        SELECT feature_date, {cols}
        FROM country_daily_features
        WHERE country = %s
          AND feature_date >= %s
          AND feature_date <= %s
        ORDER BY feature_date ASC
    """

    _UPSERT_FORECAST_SQL = """
        INSERT INTO country_escalation_forecasts (
            country, forecast_date, horizon_step, target_date,
            risk_score, instability, war_probability, terrorism_risk,
            financial_stress, confidence, variance, lower_bound, upper_bound,
            model_version
        ) VALUES (
            %(country)s, %(forecast_date)s, %(horizon_step)s, %(target_date)s,
            %(risk_score)s, %(instability)s, %(war_probability)s, %(terrorism_risk)s,
            %(financial_stress)s, %(confidence)s, %(variance)s, %(lower_bound)s,
            %(upper_bound)s, %(model_version)s
        )
        ON CONFLICT (country, forecast_date, horizon_step)
        DO UPDATE SET
            target_date      = EXCLUDED.target_date,
            risk_score       = EXCLUDED.risk_score,
            instability      = EXCLUDED.instability,
            war_probability  = EXCLUDED.war_probability,
            terrorism_risk   = EXCLUDED.terrorism_risk,
            financial_stress = EXCLUDED.financial_stress,
            confidence       = EXCLUDED.confidence,
            variance         = EXCLUDED.variance,
            lower_bound      = EXCLUDED.lower_bound,
            upper_bound      = EXCLUDED.upper_bound,
            created_at       = NOW()
    """

    _FETCH_FORECAST_SQL = """
        SELECT horizon_step, target_date, risk_score, instability,
               war_probability, terrorism_risk, financial_stress,
               confidence, variance, lower_bound, upper_bound
        FROM country_escalation_forecasts
        WHERE country = %s AND forecast_date = %s
        ORDER BY horizon_step ASC
    """

    def __init__(
        self,
        dsn: str,
        model_path: Optional[str] = None,
        seq_len: int = 90,
        mc_passes: int = 30,
        device: str = "cpu",
    ):
        self._dsn       = dsn
        self._seq_len   = seq_len
        self._mc_passes = mc_passes
        self._device    = device
        self._model: Optional[EscalationForecaster] = None

        if model_path and Path(model_path).exists():
            logger.info("Loading forecaster from %s", model_path)
            self._model = EscalationForecaster.load(model_path, device)
            logger.info(
                "Forecaster loaded: horizon=%d, params=%d",
                self._model._horizon,
                self._model.parameter_count(),
            )
        else:
            logger.warning(
                "Forecaster model not found at %s — "
                "forecast() will fall back to trend extrapolation.",
                model_path,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forecast(
        self,
        country: str,
        as_of: Optional[date] = None,
        persist: bool = True,
    ) -> ForecastResult:
        """
        Produce multi-step ahead forecasts for a country.

        Args:
            country:  Country code (e.g. "UA", "PK").
            as_of:    Reference date (default: today).
            persist:  Write results to country_escalation_forecasts.

        Returns:
            ForecastResult with per-step predictions.
        """
        if as_of is None:
            as_of = date.today()

        features = self._load_features(country, as_of)

        if self._model is not None:
            steps = self._neural_forecast(features, as_of)
        else:
            steps = self._trend_extrapolation(features, as_of)

        result = ForecastResult(country=country, forecast_date=as_of, steps=steps)

        if persist:
            self._persist(result)

        return result

    def forecast_all(
        self,
        as_of: Optional[date] = None,
        persist: bool = True,
    ) -> list[ForecastResult]:
        """Forecast all countries that have recent feature data."""
        if as_of is None:
            as_of = date.today()

        countries = self._fetch_active_countries(as_of)
        logger.info("Forecasting %d countries as of %s", len(countries), as_of)

        results = []
        for c in countries:
            try:
                r = self.forecast(c, as_of, persist=persist)
                results.append(r)
            except Exception as exc:
                logger.warning("Forecast failed for %s: %s", c, exc)

        return results

    def fetch_stored_forecast(
        self,
        country: str,
        forecast_date: Optional[date] = None,
    ) -> Optional[ForecastResult]:
        """Retrieve a previously persisted forecast from the DB."""
        if forecast_date is None:
            forecast_date = date.today()
        try:
            conn = psycopg2.connect(self._dsn)
            with conn, conn.cursor() as cur:
                cur.execute(self._FETCH_FORECAST_SQL, (country, forecast_date))
                rows = cur.fetchall()
            conn.close()
        except Exception as exc:
            logger.warning("Fetch forecast error for %s: %s", country, exc)
            return None

        if not rows:
            return None

        steps = []
        for row in rows:
            (step, tgt_date, risk, inst, war, terror, fin,
             conf, var, lo, hi) = row
            steps.append(ForecastStep(
                step=step,
                target_date=tgt_date,
                risk_score=float(risk or 0),
                instability=float(inst or 0),
                war_probability=float(war or 0),
                terrorism_risk=float(terror or 0),
                financial_stress=float(fin or 0),
                confidence=float(conf or 0),
                variance=float(var or 0),
                lower_bound=float(lo or 0),
                upper_bound=float(hi or 0),
            ))

        return ForecastResult(country=country, forecast_date=forecast_date, steps=steps)

    # ------------------------------------------------------------------
    # Scoring implementations
    # ------------------------------------------------------------------

    def _neural_forecast(
        self,
        features: np.ndarray,
        as_of: date,
    ) -> list[ForecastStep]:
        tensor = torch.from_numpy(features).unsqueeze(0).to(self._device)
        T = tensor.size(1)
        mask   = torch.ones(1, T, device=self._device)

        out = self._model.predict_with_confidence(tensor, mask, n_passes=self._mc_passes)  # type: ignore[union-attr]

        H = self._model._horizon  # type: ignore[union-attr]
        steps = []
        for h in range(H):
            step_offset = HORIZON_DAYS[h] if h < len(HORIZON_DAYS) else (h + 1) * 14
            steps.append(ForecastStep(
                step=h + 1,
                target_date=as_of + timedelta(days=step_offset),
                risk_score=float(out["risk_score"][0, h].item()),
                instability=float(out["instability"][0, h].item()),
                war_probability=float(out["war"][0, h].item()),
                terrorism_risk=float(out["terrorism"][0, h].item()),
                financial_stress=float(out["financial"][0, h].item()),
                confidence=float(out["confidence"][0, h].item()),
                variance=float(out["variance"][0, h].item()),
                lower_bound=float(out["lower_bound"][0, h].item()),
                upper_bound=float(out["upper_bound"][0, h].item()),
            ))
        return steps

    def _trend_extrapolation(
        self,
        features: np.ndarray,
        as_of: date,
        horizon: int = 4,
    ) -> list[ForecastStep]:
        """
        Fallback when no trained forecaster is available.
        Uses linear trend from last 30 days of features to project forward.
        """
        last_30 = features[-30:] if features.shape[0] >= 30 else features

        # Simple heuristic risk from features (same as RiskScorer heuristic)
        protest    = float(np.mean(last_30[:, 0]))
        violence   = float(np.mean(last_30[:, 1]))
        diplo      = float(np.mean(last_30[:, 2]))
        economic   = float(np.mean(last_30[:, 3]))
        terror     = float(np.mean(last_30[:, 4]))
        tone_neg   = float(np.mean(last_30[:, 5]))

        instability = min(0.5 * violence + 0.5 * protest, 1.0)
        war         = min(0.4 * violence + 0.4 * diplo, 1.0)
        terrorism   = min(terror * 1.2, 1.0)
        financial   = min(0.7 * economic + 0.3 * tone_neg, 1.0)
        base_risk   = (0.4 * instability + 0.3 * war + 0.2 * terrorism + 0.1 * financial)

        # Simple linear trend over last 30 days
        trend = float(np.polyfit(np.arange(last_30.shape[0]),
                                  last_30[:, 1], 1)[0]) if last_30.shape[0] > 1 else 0.0

        steps = []
        for h in range(horizon):
            step_offset = HORIZON_DAYS[h] if h < len(HORIZON_DAYS) else (h + 1) * 14
            # Project trend forward
            projected_risk = float(np.clip(base_risk + trend * step_offset * 0.5, 0, 1))
            steps.append(ForecastStep(
                step=h + 1,
                target_date=as_of + timedelta(days=step_offset),
                risk_score=projected_risk,
                instability=instability,
                war_probability=war,
                terrorism_risk=terrorism,
                financial_stress=financial,
                confidence=0.40,
                variance=0.05,
                lower_bound=float(np.clip(projected_risk - 0.08, 0, 1)),
                upper_bound=float(np.clip(projected_risk + 0.08, 0, 1)),
            ))
        return steps

    # ------------------------------------------------------------------
    # Feature loading
    # ------------------------------------------------------------------

    def _load_features(self, country: str, as_of: date) -> np.ndarray:
        # Anchor to the latest available data if as_of is ahead of the DB
        conn = psycopg2.connect(self._dsn)
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT MAX(feature_date) FROM country_daily_features WHERE country = %s",
                    (country,),
                )
                row = cur.fetchone()
                latest_for_country = row[0] if row and row[0] else as_of
                effective_date = min(as_of, latest_for_country)
                start = effective_date - timedelta(days=self._seq_len - 1)
                cols  = ", ".join(FEATURE_COLUMNS)
                cur.execute(
                    self._FETCH_FEATURES_SQL.format(cols=cols),
                    (country, start, effective_date),
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        by_date = {row[0]: list(row[1:]) for row in rows}
        matrix  = np.zeros((self._seq_len, NUM_FEATURES), dtype=np.float32)
        last_valid = None

        for i in range(self._seq_len):
            d = start + timedelta(days=i)
            if d in by_date:
                vals = [v if v is not None else 0.0 for v in by_date[d]]
                matrix[i] = vals
                last_valid = vals
            elif last_valid is not None:
                matrix[i] = last_valid

        return matrix

    def _fetch_active_countries(self, as_of: date) -> list[str]:
        """
        Return countries that have feature data around the latest available date.
        Uses the MAX(feature_date) in the table so the engine works even when
        the feature pipeline hasn't run in a few months.
        """
        conn = psycopg2.connect(self._dsn)
        try:
            with conn, conn.cursor() as cur:
                # Find the latest available date
                cur.execute("SELECT MAX(feature_date) FROM country_daily_features")
                row = cur.fetchone()
                latest = row[0] if row else None
                if latest is None:
                    return []
                since = latest - timedelta(days=30)  # within 30 days of latest snapshot
                cur.execute(
                    "SELECT DISTINCT country FROM country_daily_features "
                    "WHERE feature_date >= %s AND feature_date <= %s",
                    (since, latest),
                )
                return [r[0] for r in cur.fetchall()]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self, result: ForecastResult) -> None:
        rows = []
        for step in result.steps:
            rows.append({
                "country":        result.country,
                "forecast_date":  result.forecast_date,
                "horizon_step":   step.step,
                "target_date":    step.target_date,
                "risk_score":     step.risk_score,
                "instability":    step.instability,
                "war_probability":step.war_probability,
                "terrorism_risk": step.terrorism_risk,
                "financial_stress": step.financial_stress,
                "confidence":     step.confidence,
                "variance":       step.variance,
                "lower_bound":    step.lower_bound,
                "upper_bound":    step.upper_bound,
                "model_version":  result.model_version,
            })
        try:
            conn = psycopg2.connect(self._dsn)
            with conn, conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, self._UPSERT_FORECAST_SQL, rows)
            conn.close()
        except Exception as exc:
            logger.warning("Failed to persist forecast for %s: %s", result.country, exc)
