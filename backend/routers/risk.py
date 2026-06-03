"""
Phase 2 Risk router — enhanced MCP endpoint with SHAP + multi-task scores.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/risk", tags=["Risk"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RiskScoreRequest(BaseModel):
    country: str = Field(..., example="Pakistan")
    as_of: Optional[date] = None
    include_attributions: bool = False
    include_spillover: bool = False


class RiskScoreResponseV2(BaseModel):
    country: str
    risk_score: float
    confidence: float
    trend: str
    level: str
    instability: float
    war_probability: float
    terrorism_risk: float
    financial_stress: float
    major_drivers: list[str]
    advisory: str
    prediction_date: str
    # Phase 2 additions
    attributions: Optional[dict[str, float]] = None
    top_features: Optional[list[dict]] = None
    spillover_neighbors: Optional[list[dict]] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/score", response_model=RiskScoreResponseV2)
def get_risk_score_v2(req: RiskScoreRequest, request_state=None):
    """
    Enhanced MCP endpoint (Phase 2).
    Optionally includes feature attributions (IG) and spillover neighbors.
    """
    from fastapi import Request
    from starlette.requests import Request as StarletteRequest

    # Get scorer from app state
    import inspect
    frame = inspect.currentframe()
    # Use module-level access pattern — scorer injected at route registration
    scorer = _scorer_ref.get("scorer")
    if scorer is None:
        raise HTTPException(500, "Scorer not initialized")

    try:
        pred = scorer.score(req.country, as_of=req.as_of)
    except Exception as exc:
        raise HTTPException(500, str(exc))

    response = RiskScoreResponseV2(
        country=pred.country,
        risk_score=pred.risk_score,
        confidence=pred.confidence,
        trend=pred.trend,
        level=pred.advisory.level,
        instability=pred.instability,
        war_probability=pred.war_probability,
        terrorism_risk=pred.terrorism_risk,
        financial_stress=pred.financial_stress,
        major_drivers=pred.advisory.major_drivers,
        advisory=pred.advisory.advisory_text,
        prediction_date=str(pred.prediction_date),
    )

    if req.include_attributions and _attr_engine_ref.get("engine"):
        engine = _attr_engine_ref["engine"]
        attr_dict = engine.fetch_attributions(
            req.country,
            pred.prediction_date,
        )
        response.attributions = attr_dict
        if attr_dict:
            sorted_attrs = sorted(
                attr_dict.items(), key=lambda x: abs(x[1]), reverse=True
            )
            response.top_features = [
                {"feature": k, "attribution": round(v, 6)}
                for k, v in sorted_attrs[:3]
            ]

    if req.include_spillover and _spillover_ref.get("analyzer"):
        analyzer = _spillover_ref["analyzer"]
        neighbors = analyzer.fetch_neighbors(req.country, top_n=5)
        response.spillover_neighbors = neighbors

    return response


# Global refs (set by main.py on startup)
_scorer_ref:      dict = {}
_attr_engine_ref: dict = {}
_spillover_ref:   dict = {}


def init_router(scorer, attr_engine=None, spillover_analyzer=None):
    _scorer_ref["scorer"]       = scorer
    _attr_engine_ref["engine"]  = attr_engine
    _spillover_ref["analyzer"]  = spillover_analyzer
