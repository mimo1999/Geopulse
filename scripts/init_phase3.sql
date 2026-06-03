-- ============================================================
-- GLDT Phase 3 — PostgreSQL Migration
-- Run AFTER init_local_pg.sql (Phase 1+2 schema must exist)
--   psql -U postgres -d gdelt_risk -f scripts/init_phase3.sql
-- ============================================================

-- ---------------------------------------------------------------------------
-- ESCALATION FORECASTS
-- Multi-step ahead risk trajectory predictions per country.
-- One row per (country, forecast_date, horizon_step).
-- horizon_step 1 = next bi-weekly period (~14 days),
--              2 = ~28 days, 3 = ~42 days, 4 = ~56 days.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS country_escalation_forecasts (
    id                  BIGSERIAL   PRIMARY KEY,
    country             TEXT        NOT NULL,
    forecast_date       DATE        NOT NULL,       -- when the forecast was generated
    horizon_step        SMALLINT    NOT NULL,        -- 1..4
    target_date         DATE        NOT NULL,       -- the forecasted date

    risk_score          FLOAT       NOT NULL,
    instability         FLOAT,
    war_probability     FLOAT,
    terrorism_risk      FLOAT,
    financial_stress    FLOAT,

    confidence          FLOAT,
    variance            FLOAT,
    lower_bound         FLOAT,      -- 10th-percentile (MC estimate)
    upper_bound         FLOAT,      -- 90th-percentile (MC estimate)

    model_version       TEXT        DEFAULT 'v0.3-forecaster',
    created_at          TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (country, forecast_date, horizon_step)
);

CREATE INDEX IF NOT EXISTS idx_forecasts_country_date
    ON country_escalation_forecasts (country, forecast_date DESC);
CREATE INDEX IF NOT EXISTS idx_forecasts_target
    ON country_escalation_forecasts (target_date DESC, risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_forecasts_escalation
    ON country_escalation_forecasts (risk_score DESC, horizon_step);

-- ---------------------------------------------------------------------------
-- GNN NODE EMBEDDINGS
-- Per-country graph embeddings computed by RiskGNN.
-- Contagion score = how much risk the country is absorbing from neighbors.
-- Risk amplification = delta vs raw model prediction.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gnn_node_embeddings (
    country             TEXT        NOT NULL,
    computed_date       DATE        NOT NULL,

    embedding           FLOAT[]     NOT NULL,   -- 8-dim node embedding
    contagion_score     FLOAT       DEFAULT 0.0,
    risk_amplification  FLOAT       DEFAULT 0.0,
    network_adjusted_risk FLOAT,

    computed_at         TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (country, computed_date)
);

CREATE INDEX IF NOT EXISTS idx_gnn_date
    ON gnn_node_embeddings (computed_date DESC, contagion_score DESC);

-- ---------------------------------------------------------------------------
-- ADVISORY CORPUS
-- Curated + auto-generated situation descriptions for RAG retrieval.
-- tfidf_vector populated by the corpus builder script.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS advisory_corpus (
    id                  BIGSERIAL   PRIMARY KEY,
    situation_type      TEXT        NOT NULL,    -- e.g. 'military_escalation_high'
    risk_level          TEXT        NOT NULL,    -- CRITICAL/HIGH/ELEVATED/MODERATE/LOW
    text                TEXT        NOT NULL,    -- 2-3 sentence situation description
    tags                TEXT[]      DEFAULT '{}',
    tfidf_vector        FLOAT[],                 -- populated by corpus builder
    source              TEXT        DEFAULT 'manual',  -- 'manual'|'event_cluster'|'ollama'
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_corpus_level_type
    ON advisory_corpus (risk_level, situation_type);

-- ---------------------------------------------------------------------------
-- GRANTS
-- ---------------------------------------------------------------------------
GRANT ALL PRIVILEGES ON ALL TABLES    IN SCHEMA public TO gldt;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO gldt;
ALTER DEFAULT PRIVILEGES IN SCHEMA PUBLIC
    GRANT ALL ON TABLES    TO gldt;
ALTER DEFAULT PRIVILEGES IN SCHEMA PUBLIC
    GRANT ALL ON SEQUENCES TO gldt;
