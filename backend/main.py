"""
FastAPI backend — Risk Inference Engine + MCP Endpoint.

Phase 1 routes:
    POST /riskscore              MCP-compatible risk score endpoint
    GET  /countries              List tracked countries
    GET  /country/{code}/timeline  Historical risk timeline
    GET  /global/heatmap         All countries' latest risk scores
    GET  /health                 Health check
    POST /ingest/trigger         Manually trigger ingestion run

Phase 2 additions:
    POST /riskscore              Enhanced: attributions + spillover optional fields
    GET  /country/{code}/events  Event clusters for country drilldown
    GET  /country/{code}/spillover  Top spillover neighbors
    GET  /country/{code}/attributions  Feature attributions (IG)
    GET  /country/{code}/labels  Ground-truth proxy labels
    POST /analyze/spillover      Trigger spillover network computation
    POST /analyze/labels         Trigger label generation for a date range
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
import yaml
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("backend.main")

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    cfg_path = Path("configs/config.yaml")
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {}


_cfg = _load_config()
_db_cfg = _cfg.get("database", {})

DATABASE_URL = os.getenv(
    "DATABASE_SYNC_URL",
    (
        f"postgresql://{_db_cfg.get('user', 'gldt')}:"
        f"{_db_cfg.get('password', 'gldt_secret')}@"
        f"{_db_cfg.get('host', 'localhost')}:"
        f"{_db_cfg.get('port', 5432)}/"
        f"{_db_cfg.get('name', 'gdelt_risk')}"
    ),
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/checkpoints/run_phase1_best.pt")

# Country code → display name
try:
    from data.country_codes import code_to_name as _code_to_name
except ImportError:
    def _code_to_name(code: str) -> str:  # type: ignore[misc]
        return code


# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------

def _get_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Backend starting — DB: %s", DATABASE_URL.split("@")[-1])

    from inference.risk_scorer import RiskScorer
    scorer = RiskScorer(
        dsn=DATABASE_URL,
        model_path=MODEL_PATH if Path(MODEL_PATH).exists() else None,
    )
    app.state.scorer = scorer

    # Phase 2: attribution engine + spillover analyzer
    app.state.attr_engine = None
    app.state.spillover   = None
    try:
        from inference.spillover import SpilloverAnalyzer
        app.state.spillover = SpilloverAnalyzer(dsn=DATABASE_URL)
        logger.info("Spillover analyzer ready")
    except Exception as exc:
        logger.warning("Spillover unavailable: %s", exc)

    if scorer._model is not None:
        try:
            from inference.explainer import AttributionEngine
            app.state.attr_engine = AttributionEngine(
                model=scorer._model,
                dsn=DATABASE_URL,
                model_version="v0.2",
            )
            logger.info("Attribution engine ready")
        except Exception as exc:
            logger.warning("Attribution engine unavailable: %s", exc)

    # Phase 3: escalation forecaster
    app.state.forecaster = None
    try:
        from inference.escalation_forecaster import EscalationForecasterEngine
        forecaster_path = os.getenv(
            "FORECASTER_PATH",
            "models/checkpoints/forecaster_v1_best.pt",
        )
        app.state.forecaster = EscalationForecasterEngine(
            dsn=DATABASE_URL,
            model_path=forecaster_path if Path(forecaster_path).exists() else None,
        )
        logger.info("Escalation forecaster ready")
    except Exception as exc:
        logger.warning("Forecaster unavailable: %s", exc)

    # Phase 3: GNN spillover engine
    app.state.gnn_engine = None
    try:
        from inference.gnn_spillover import GNNSpilloverEngine
        app.state.gnn_engine = GNNSpilloverEngine(dsn=DATABASE_URL)
        logger.info("GNN spillover engine ready")
    except Exception as exc:
        logger.warning("GNN engine unavailable: %s", exc)

    # Phase 3: RAG advisory engine
    app.state.rag_engine = None
    try:
        from inference.rag_engine import RAGEngine
        app.state.rag_engine = RAGEngine(dsn=DATABASE_URL)
        logger.info("RAG advisory engine ready")
    except Exception as exc:
        logger.warning("RAG engine unavailable: %s", exc)

    yield
    logger.info("Backend shutdown")


app = FastAPI(
    title="GLDT Risk Intelligence API",
    description="Geopolitical risk analytics and escalation monitoring powered by GDELT.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cfg.get("api", {}).get("cors_origins", ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class RiskScoreRequest(BaseModel):
    country: str = Field(..., example="Pakistan", description="Country name or ISO code")
    as_of: Optional[date] = Field(None, description="Reference date (default: today)")


class RiskScoreResponse(BaseModel):
    country: str
    name: str = ""
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


class CountryTimelineEntry(BaseModel):
    date: str
    risk_score: Optional[float]
    protest_score: Optional[float]
    violence_score: Optional[float]
    diplomatic_stress: Optional[float]
    avg_sentiment: Optional[float]


class HeatmapEntry(BaseModel):
    country: str
    risk_score: float
    confidence: Optional[float]
    trend: Optional[str]
    level: str


class IngestionRequest(BaseModel):
    target_date: Optional[date] = None
    backfill_days: int = Field(default=1, ge=1, le=365)


class V2IngestionRequest(BaseModel):
    mode: str = Field(
        default="latest",
        description="'latest' = process most recent 15-min file; "
                    "'catchup' = register all master-list files and process oldest pending batch",
    )
    catchup_batch: int = Field(
        default=20, ge=1, le=200,
        description="Number of pending files to process when mode='catchup'",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    try:
        conn = _get_conn()
        with conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
        return {"status": "ok", "db": "connected", "timestamp": datetime.utcnow().isoformat()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}")


@app.post("/riskscore", response_model=RiskScoreResponse, tags=["MCP"])
def get_risk_score(req: RiskScoreRequest):
    """
    MCP-compatible endpoint — returns structured risk score for a country.

    This is the primary endpoint for downstream AI agents and dashboards.
    """
    scorer = app.state.scorer
    try:
        pred = scorer.score(req.country, as_of=req.as_of)
    except Exception as exc:
        logger.error("Scoring error for %s: %s", req.country, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return RiskScoreResponse(
        country=pred.country,
        name=_code_to_name(pred.country),
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


@app.get("/countries", tags=["Data"])
def list_countries():
    """List all countries with recent event data."""
    try:
        conn = _get_conn()
        with conn, conn.cursor() as cur:
            cur.execute("""
                SELECT country, MAX(feature_date) AS last_date,
                       COUNT(*) AS day_count
                FROM country_daily_features
                GROUP BY country
                ORDER BY country
            """)
            rows = cur.fetchall()
        countries = []
        for r in rows:
            d = dict(r)
            d["name"] = _code_to_name(d["country"])
            countries.append(d)
        return {"countries": countries, "total": len(rows)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/country/{country}/timeline", tags=["Data"])
def get_country_timeline(
    country: str,
    days: int = Query(default=90, ge=7, le=365),
):
    """Return historical risk timeline for a country."""
    try:
        conn = _get_conn()
        with conn, conn.cursor() as cur:
            # Anchor to the most recent data in DB rather than today's date,
            # so windows stay valid even when ingestion is behind schedule.
            cur.execute(
                "SELECT MAX(feature_date) FROM country_daily_features WHERE country = %s",
                (country,),
            )
            row = cur.fetchone()
            max_date = row["max"] if row and row["max"] else None
            since = (max_date or date.today()) - timedelta(days=days)

            cur.execute("""
                SELECT feature_date, risk_score, protest_score,
                       violence_score, diplomatic_stress, economic_stress,
                       terrorism_score, avg_sentiment, confidence
                FROM country_daily_features
                WHERE country = %s AND feature_date >= %s
                ORDER BY feature_date ASC
            """, (country, since))
            rows = cur.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail=f"No data for country: {country}")

        return {
            "country": country,
            "days": days,
            "timeline": [dict(r) for r in rows],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/global/heatmap", tags=["Data"])
def get_global_heatmap():
    """
    Latest risk score per country — used to render the world heatmap.
    """
    try:
        conn = _get_conn()
        with conn, conn.cursor() as cur:
            cur.execute("""
                SELECT
                    l.country,
                    l.risk_score,
                    l.confidence,
                    l.feature_date,
                    p.trend
                FROM latest_country_risk l
                LEFT JOIN LATERAL (
                    SELECT trend
                    FROM country_risk_predictions
                    WHERE country = l.country
                    ORDER BY prediction_time DESC
                    LIMIT 1
                ) p ON TRUE
                WHERE l.risk_score IS NOT NULL
                ORDER BY l.risk_score DESC
            """)
            rows = cur.fetchall()

        from advisory.rule_engine import classify_risk
        return {
            "as_of": str(date.today()),
            "countries": [
                {
                    **dict(r),
                    "name": _code_to_name(r["country"]),
                    "level": classify_risk(r["risk_score"]) if r["risk_score"] else "UNKNOWN",
                    "feature_date": str(r["feature_date"]),
                }
                for r in rows
            ],
            "total": len(rows),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/country/{country}/predictions", tags=["Predictions"])
def get_predictions(
    country: str,
    limit: int = Query(default=30, ge=1, le=365),
):
    """Return model predictions history for a country."""
    try:
        conn = _get_conn()
        with conn, conn.cursor() as cur:
            cur.execute("""
                SELECT prediction_time, risk_score, instability_score,
                       war_probability, terrorism_risk, financial_stress,
                       confidence, trend, advisory, model_version
                FROM country_risk_predictions
                WHERE country = %s
                ORDER BY prediction_time DESC
                LIMIT %s
            """, (country, limit))
            rows = cur.fetchall()
        return {"country": country, "predictions": [dict(r) for r in rows]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest/trigger", tags=["Admin"])
def trigger_ingestion(req: IngestionRequest, background_tasks: BackgroundTasks):
    """Manually trigger a GDELT ingestion run (runs in background)."""

    def _run_ingestion():
        from ingestion.ingestion_pipeline import IngestionPipeline, PipelineConfig
        cfg = PipelineConfig(dsn=DATABASE_URL)
        with IngestionPipeline(cfg) as pipeline:
            if req.backfill_days > 1:
                pipeline.backfill(days=req.backfill_days)
            else:
                target = req.target_date or (date.today() - timedelta(days=1))
                pipeline.ingest_date(target)

    background_tasks.add_task(_run_ingestion)
    return {
        "status": "triggered",
        "target_date": str(req.target_date or "yesterday"),
        "backfill_days": req.backfill_days,
    }


@app.post("/ingest/v2/trigger", tags=["Admin"])
def trigger_v2_ingestion(req: V2IngestionRequest, background_tasks: BackgroundTasks):
    """
    Trigger GDELT 2.0 15-minute file ingestion (runs in background).

    Modes:
    - **latest**: Download and ingest the single most recent 15-min export file
      from `lastupdate.txt`. Suitable for running as a 15-minute cron job.
    - **catchup**: Register *all* files from the GDELT 2.0 master list into the
      cursor table, then process the oldest `catchup_batch` pending files.
      Use this once on first setup or after a gap in ingestion.
    """
    mode = req.mode.lower()
    if mode not in ("latest", "catchup"):
        raise HTTPException(status_code=400, detail="mode must be 'latest' or 'catchup'")

    def _run_v2():
        from ingestion.gdelt_v2 import GDELTV2Processor
        processor = GDELTV2Processor(dsn=DATABASE_URL, raw_data_dir="data/raw/v2")
        if mode == "latest":
            results = processor.process_latest()
        else:
            results = processor.run_catchup(batch_size=req.catchup_batch)
        ok  = sum(1 for r in results if r.status == "success")
        ins = sum(r.events_inserted for r in results if r.status == "success")
        logger.info("v2 ingest (%s): %d/%d files OK, %d events inserted", mode, ok, len(results), ins)

    background_tasks.add_task(_run_v2)
    return {
        "status": "triggered",
        "mode": mode,
        "catchup_batch": req.catchup_batch if mode == "catchup" else None,
    }


@app.get("/ingestion/runs", tags=["Admin"])
def get_ingestion_runs(limit: int = Query(default=20, ge=1, le=100)):
    """Return recent ingestion audit log entries."""
    try:
        conn = _get_conn()
        with conn, conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM ingestion_runs
                ORDER BY run_time DESC LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
        return {"runs": [dict(r) for r in rows]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/ingestion/v2/status", tags=["Admin"])
def get_v2_ingestion_status():
    """
    Return GDELT v2 cursor statistics — how many files are pending/done/error
    and total events inserted.
    """
    try:
        conn = _get_conn()
        with conn, conn.cursor() as cur:
            cur.execute("""
                SELECT
                    status,
                    COUNT(*)            AS file_count,
                    SUM(events_inserted) AS total_events,
                    MAX(file_timestamp) AS latest_file_ts,
                    MAX(processed_at)   AS last_processed_at
                FROM gdelt_v2_cursor
                GROUP BY status
                ORDER BY status
            """)
            rows = cur.fetchall()

            cur.execute("SELECT COUNT(*) AS total FROM gdelt_events")
            total_row = cur.fetchone()

        return {
            "gdelt_events_total": total_row["total"] if total_row else 0,
            "cursor_summary": [dict(r) for r in rows],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ===========================================================================
# Phase 2 Routes
# ===========================================================================

@app.get("/country/{country}/events", tags=["Phase 2 — Events"])
def get_country_events(
    country: str,
    days: int = Query(default=30, ge=1, le=180),
):
    """
    Return pre-aggregated event clusters for country drilldown.
    Shows protest / military / terrorism / sanctions / diplomatic event groups.
    """
    try:
        conn = _get_conn()
        with conn, conn.cursor() as cur:
            # Anchor window to latest available data, not today's calendar date.
            cur.execute(
                "SELECT MAX(cluster_date) FROM event_clusters WHERE country = %s",
                (country,),
            )
            row = cur.fetchone()
            max_date = row["max"] if row and row["max"] else None
            since = (max_date or date.today()) - timedelta(days=days)

            cur.execute("""
                SELECT cluster_date, category, event_count,
                       total_mentions, avg_goldstein, avg_tone,
                       max_intensity, top_actor_pairs
                FROM event_clusters
                WHERE country = %s AND cluster_date >= %s
                ORDER BY cluster_date DESC, total_mentions DESC
            """, (country, since))
            rows = cur.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail=f"No event data for {country}")

        return {
            "country": country,
            "days":    days,
            "events":  [dict(r) for r in rows],
            "total":   len(rows),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/country/{country}/spillover", tags=["Phase 2 — Spillover"])
def get_country_spillover(
    country: str,
    top_n: int = Query(default=5, ge=1, le=20),
):
    """
    Return top spillover neighbors for a country.
    Shows risk correlation + bilateral event co-occurrence.
    """
    spillover = app.state.spillover
    if spillover is None:
        raise HTTPException(status_code=503, detail="Spillover analyzer not available")

    neighbors = spillover.fetch_neighbors(country, top_n=top_n)
    return {
        "country":   country,
        "neighbors": neighbors,
        "total":     len(neighbors),
    }


@app.get("/country/{country}/attributions", tags=["Phase 2 — Explainability"])
def get_country_attributions(
    country: str,
    target_date: Optional[date] = None,
    target_head: str = Query(default="risk_score"),
):
    """
    Return feature attributions (Integrated Gradients) for a country prediction.
    Shows which features most influenced the risk score.
    """
    if target_date is None:
        target_date = date.today()

    attr_engine = app.state.attr_engine
    if attr_engine is None:
        return {
            "country":      country,
            "date":         str(target_date),
            "attributions": None,
            "message":      "Model not trained yet — attributions unavailable",
        }

    attr = attr_engine.fetch_attributions(country, target_date)
    if attr is None:
        try:
            scorer = app.state.scorer
            feat_matrix = scorer._load_features(country, target_date)
            import numpy as np
            mask = np.ones(feat_matrix.shape[0], dtype=np.float32)
            result = attr_engine.explain_and_save(
                country=country,
                features=feat_matrix,
                mask=mask,
                target_date=target_date,
                target_head=target_head,
            )
            return result.to_dict()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return {
        "country":      country,
        "date":         str(target_date),
        "target":       target_head,
        "attributions": attr,
    }


@app.get("/country/{country}/labels", tags=["Phase 2 — Labels"])
def get_country_labels(
    country: str,
    days: int = Query(default=90, ge=7, le=365),
):
    """Return ground-truth proxy labels for model training inspection."""
    try:
        conn = _get_conn()
        with conn, conn.cursor() as cur:
            # Anchor window to latest available data.
            cur.execute(
                "SELECT MAX(label_date) FROM country_multitask_labels WHERE country = %s",
                (country,),
            )
            row = cur.fetchone()
            max_date = row["max"] if row and row["max"] else None
            since = (max_date or date.today()) - timedelta(days=days)

            cur.execute("""
                SELECT label_date, instability_label, war_label,
                       terrorism_label, financial_label, event_count
                FROM country_multitask_labels
                WHERE country = %s AND label_date >= %s
                ORDER BY label_date DESC
            """, (country, since))
            rows = cur.fetchall()
        return {"country": country, "labels": [dict(r) for r in rows]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class SpilloverTriggerRequest(BaseModel):
    as_of: Optional[date] = None
    window_days: int = Field(default=90, ge=30, le=365)


@app.post("/analyze/spillover", tags=["Phase 2 — Admin"])
def trigger_spillover(req: SpilloverTriggerRequest, background_tasks: BackgroundTasks):
    """Trigger spillover network computation (runs in background)."""

    def _run():
        from inference.spillover import SpilloverAnalyzer
        analyzer = SpilloverAnalyzer(dsn=DATABASE_URL, window_days=req.window_days)
        n = analyzer.compute_and_save(as_of=req.as_of)
        logger.info("Spillover computation done: %d pairs", n)

    background_tasks.add_task(_run)
    return {"status": "triggered", "as_of": str(req.as_of or date.today())}


class LabelTriggerRequest(BaseModel):
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    days: int = Field(default=30, ge=1, le=365)


@app.post("/analyze/labels", tags=["Phase 2 — Admin"])
def trigger_label_generation(req: LabelTriggerRequest, background_tasks: BackgroundTasks):
    """Trigger proxy label generation for a date range."""

    def _run():
        from preprocessing.label_generator import LabelGenerator
        gen = LabelGenerator(dsn=DATABASE_URL)
        end   = req.end_date or (date.today() - timedelta(days=1))
        start = req.start_date or (end - timedelta(days=req.days - 1))
        results = gen.compute_labels_range(start, end)
        total = sum(results.values())
        logger.info("Label generation done: %d rows", total)

    background_tasks.add_task(_run)
    return {"status": "triggered", "days": req.days}


@app.post("/analyze/clusters", tags=["Phase 2 — Admin"])
def trigger_event_clustering(background_tasks: BackgroundTasks):
    """Trigger event cluster computation for the last 7 days."""

    def _run():
        from preprocessing.event_clusterer import EventClusterer
        clusterer = EventClusterer(dsn=DATABASE_URL)
        end   = date.today() - timedelta(days=1)
        start = end - timedelta(days=6)
        total = clusterer.compute_range(start, end)
        logger.info("Event clustering done: %d rows", total)

    background_tasks.add_task(_run)
    return {"status": "triggered"}


# ===========================================================================
# Phase 3 Routes — Escalation Forecasting, GNN, RAG Advisories
# ===========================================================================

# ---------------------------------------------------------------------------
# Pydantic schemas (Phase 3)
# ---------------------------------------------------------------------------

class ForecastStep(BaseModel):
    step: int
    target_date: str
    risk_score: float
    instability: float
    war_probability: float
    terrorism_risk: float
    financial_stress: float
    confidence: float
    variance: float
    lower_bound: float
    upper_bound: float


class ForecastResponse(BaseModel):
    country: str
    forecast_date: str
    horizon_steps: int
    forecasts: list[ForecastStep]
    model_version: str


class GNNInfluenceResponse(BaseModel):
    country: str
    contagion_score: float
    risk_amplification: float
    network_adjusted_risk: float
    top_influencers: list[dict]


class RAGAdvisoryResponse(BaseModel):
    country: str
    name: str = ""
    advisory: str
    retrieved_contexts: list[dict]
    rag_confidence: float
    level: str


class ForecastTriggerRequest(BaseModel):
    country: Optional[str] = None
    as_of:   Optional[date] = None


class GNNTriggerRequest(BaseModel):
    as_of: Optional[date] = None


class CorpusRebuildRequest(BaseModel):
    country: Optional[str] = None
    days: int = Field(default=90, ge=7, le=365)


# ---------------------------------------------------------------------------
# Forecasting routes
# ---------------------------------------------------------------------------

@app.get("/country/{country}/forecast", tags=["Phase 3 — Forecast"])
def get_country_forecast(
    country: str,
    as_of: Optional[date] = None,
):
    """
    Multi-step ahead risk forecast for a country.
    Returns 4 bi-weekly predictions (≈ 14, 28, 42, 56 days ahead).
    Falls back to trend extrapolation if the forecaster model is not trained.
    """
    forecaster = app.state.forecaster
    if forecaster is None:
        raise HTTPException(status_code=503, detail="Forecaster not available")

    try:
        result = forecaster.forecast(country, as_of=as_of, persist=True)
        return result.to_dict()
    except Exception as exc:
        logger.error("Forecast error for %s: %s", country, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/global/escalation_alerts", tags=["Phase 3 — Forecast"])
def get_escalation_alerts(
    min_risk: float = Query(default=0.60, ge=0.0, le=1.0),
    horizon_step: int = Query(default=1, ge=1, le=4),
):
    """
    Countries predicted to exceed min_risk at the given horizon step.
    Sorted by predicted risk descending.
    """
    try:
        conn = _get_conn()
        with conn, conn.cursor() as cur:
            cur.execute("""
                SELECT f.country, f.risk_score AS predicted_risk,
                       f.confidence, f.target_date,
                       l.risk_score AS current_risk
                FROM country_escalation_forecasts f
                LEFT JOIN latest_country_risk l ON l.country = f.country
                WHERE f.forecast_date = (
                    SELECT MAX(forecast_date) FROM country_escalation_forecasts
                )
                  AND f.horizon_step  = %s
                  AND f.risk_score   >= %s
                ORDER BY f.risk_score DESC
                LIMIT 50
            """, (horizon_step, min_risk))
            rows = cur.fetchall()
        conn.close()

        alerts = []
        for row in rows:
            pred  = row["predicted_risk"]
            conf  = row["confidence"]
            tgt_date = row["target_date"]
            curr  = row["current_risk"]
            delta = round(float(pred) - float(curr or 0), 4)
            alerts.append({
                "country":        row["country"],
                "name":           _code_to_name(row["country"]),
                "predicted_risk": round(float(pred), 4),
                "current_risk":   round(float(curr or 0), 4),
                "delta":          delta,
                "horizon_step":   horizon_step,
                "confidence":     round(float(conf or 0), 4),
                "target_date":    str(tgt_date),
                "risk_score":     round(float(pred), 4),   # alias for Streamlit compat
            })

        return {"alerts": alerts, "total": len(alerts), "min_risk": min_risk}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/forecast/trigger", tags=["Phase 3 — Admin"])
def trigger_forecast(req: ForecastTriggerRequest, background_tasks: BackgroundTasks):
    """Trigger forecast computation for all countries (or one) in background."""

    def _run():
        forecaster = app.state.forecaster
        if forecaster is None:
            return
        if req.country:
            forecaster.forecast(req.country, as_of=req.as_of, persist=True)
        else:
            results = forecaster.forecast_all(as_of=req.as_of, persist=True)
            logger.info("Forecast complete: %d countries", len(results))

    background_tasks.add_task(_run)
    return {
        "status":  "triggered",
        "country": req.country or "all",
        "as_of":   str(req.as_of or date.today()),
    }


# ---------------------------------------------------------------------------
# GNN routes
# ---------------------------------------------------------------------------

@app.get("/country/{country}/gnn_influence", tags=["Phase 3 — GNN"])
def get_gnn_influence(country: str):
    """
    GNN-based enrichment for a country: contagion score, risk amplification,
    network-adjusted risk, and top influencing neighbours.
    """
    gnn = app.state.gnn_engine
    if gnn is None:
        raise HTTPException(status_code=503, detail="GNN engine not available")

    enrichment = gnn.fetch_country_enrichment(country)
    if not enrichment:
        # Run on-demand for this single country
        try:
            results = gnn.enrich(persist=True)
            enrichment = results.get(country, {})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    if not enrichment:
        raise HTTPException(status_code=404, detail=f"No GNN data for {country}")

    influencers = gnn.get_top_influencers(country, top_n=5)
    return GNNInfluenceResponse(
        country=country,
        contagion_score=enrichment.get("contagion_score", 0),
        risk_amplification=enrichment.get("risk_amplification", 0),
        network_adjusted_risk=enrichment.get("network_adjusted_risk", 0),
        top_influencers=influencers,
    )


@app.get("/global/gnn_network", tags=["Phase 3 — GNN"])
def get_gnn_network(
    min_weight: float = Query(default=0.20, ge=0.0, le=1.0),
):
    """
    Full node+edge list for GNN network visualisation.
    Returns nodes (country + risk scores + contagion) and edges (spillover weights).
    """
    try:
        conn = _get_conn()
        with conn, conn.cursor() as cur:
            # Nodes
            cur.execute("""
                SELECT g.country, g.contagion_score, g.risk_amplification,
                       g.network_adjusted_risk, l.risk_score
                FROM gnn_node_embeddings g
                JOIN latest_country_risk l ON l.country = g.country
                WHERE g.computed_date = (SELECT MAX(computed_date) FROM gnn_node_embeddings)
            """)
            node_rows = cur.fetchall()

            # Edges
            cur.execute("""
                SELECT country_a, country_b, spillover_weight
                FROM country_spillover
                WHERE computed_date = (SELECT MAX(computed_date) FROM country_spillover)
                  AND spillover_weight >= %s
            """, (min_weight,))
            edge_rows = cur.fetchall()

        conn.close()

        nodes = [
            {
                "country":               r["country"],
                "name":                  _code_to_name(r["country"]),
                "contagion_score":        round(float(r["contagion_score"] or 0), 4),
                "risk_amplification":     round(float(r["risk_amplification"] or 0), 4),
                "network_adjusted_risk":  round(float(r["network_adjusted_risk"] or 0), 4),
                "risk_score":             round(float(r["risk_score"] or 0), 4),
            }
            for r in node_rows
        ]
        edges = [
            {
                "source":       r["country_a"],
                "source_name":  _code_to_name(r["country_a"]),
                "target":       r["country_b"],
                "target_name":  _code_to_name(r["country_b"]),
                "weight":       round(float(r["spillover_weight"]), 4),
            }
            for r in edge_rows
        ]

        return {"nodes": nodes, "edges": edges}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/analyze/gnn", tags=["Phase 3 — Admin"])
def trigger_gnn(req: GNNTriggerRequest, background_tasks: BackgroundTasks):
    """Trigger GNN enrichment computation (background task)."""

    def _run():
        gnn = app.state.gnn_engine
        if gnn is None:
            return
        results = gnn.enrich(as_of=req.as_of, persist=True)
        logger.info("GNN enrichment complete: %d countries", len(results))

    background_tasks.add_task(_run)
    return {"status": "triggered", "as_of": str(req.as_of or date.today())}


# ---------------------------------------------------------------------------
# RAG Advisory routes
# ---------------------------------------------------------------------------

@app.get("/country/{country}/rag_advisory", tags=["Phase 3 — RAG"])
def get_rag_advisory(
    country: str,
    include_retrieved: bool = Query(default=True),
):
    """
    RAG-enhanced advisory for a country.
    Enriches the rule-based advisory with retrieved historical analogues.
    Includes forecast trajectory context if available.
    """
    rag = app.state.rag_engine
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not available")

    # Get base scores from scorer
    scorer = app.state.scorer
    try:
        pred = scorer.score(country)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scoring error: {exc}")

    # Get forecast trajectory if available
    forecast_trajectory = None
    forecaster = app.state.forecaster
    if forecaster is not None:
        try:
            fc = forecaster.fetch_stored_forecast(country)
            if fc:
                forecast_trajectory = fc.risk_trajectory
        except Exception:
            pass

    try:
        advisory, retrieved = rag.generate(
            country=country,
            risk_score=pred.risk_score,
            confidence=pred.confidence,
            trend=pred.trend,
            instability=pred.instability,
            war=pred.war_probability,
            terrorism=pred.terrorism_risk,
            financial=pred.financial_stress,
            forecast_trajectory=forecast_trajectory,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    from advisory.rule_engine import classify_risk
    return RAGAdvisoryResponse(
        country=country,
        name=_code_to_name(country),
        advisory=advisory.advisory_text,
        retrieved_contexts=retrieved if include_retrieved else [],
        rag_confidence=float(advisory.confidence),
        level=advisory.level,
    )


@app.get("/advisory/corpus/stats", tags=["Phase 3 — RAG"])
def get_corpus_stats():
    """Return advisory corpus statistics."""
    rag = app.state.rag_engine
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not available")
    return rag.get_corpus_stats()


@app.get("/advisory/corpus", tags=["Phase 3 — RAG"])
def browse_corpus(
    risk_level: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100),
):
    """Browse advisory corpus entries."""
    try:
        conn = _get_conn()
        with conn, conn.cursor() as cur:
            if risk_level:
                cur.execute("""
                    SELECT id, situation_type, risk_level, text, tags, source
                    FROM advisory_corpus
                    WHERE risk_level = %s
                    ORDER BY created_at DESC LIMIT %s
                """, (risk_level.upper(), limit))
            else:
                cur.execute("""
                    SELECT id, situation_type, risk_level, text, tags, source
                    FROM advisory_corpus
                    ORDER BY created_at DESC LIMIT %s
                """, (limit,))
            rows = cur.fetchall()
        conn.close()
        return {
            "entries": [
                {"id": r["id"], "situation_type": r["situation_type"], "risk_level": r["risk_level"],
                 "text": r["text"], "tags": r["tags"] or [], "source": r["source"]}
                for r in rows
            ],
            "total": len(rows),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/advisory/corpus/rebuild", tags=["Phase 3 — RAG"])
def rebuild_corpus(req: CorpusRebuildRequest, background_tasks: BackgroundTasks):
    """Rebuild advisory corpus from event_clusters + seed entries (background)."""

    def _run():
        rag = app.state.rag_engine
        if rag is None:
            return
        n = rag.rebuild_corpus()
        rag.build_corpus_from_clusters(country=req.country, days=req.days, persist=True)
        logger.info("Advisory corpus rebuilt: %d entries", n)

    background_tasks.add_task(_run)
    return {"status": "triggered", "days": req.days}
