-- ============================================================
-- GLDT Project — PostgreSQL Init Script
-- Extensions + Schema + Partitioning
-- ============================================================

-- ---------------------
-- Extensions
-- ---------------------
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;      -- fast text search
CREATE EXTENSION IF NOT EXISTS btree_gin;    -- GIN on composite keys

-- ---------------------
-- ENUM types
-- ---------------------
CREATE TYPE quad_class_enum AS ENUM (
    'verbal_cooperation',
    'material_cooperation',
    'verbal_conflict',
    'material_conflict'
);

-- ============================================================
-- RAW GDELT EVENTS  (partitioned by month on event_date)
-- ============================================================
CREATE TABLE IF NOT EXISTS gdelt_events (
    global_event_id     BIGINT          NOT NULL,
    event_date          DATE            NOT NULL,
    actor1_code         TEXT,
    actor1_name         TEXT,
    actor1_country      TEXT,
    actor1_type1        TEXT,
    actor2_code         TEXT,
    actor2_name         TEXT,
    actor2_country      TEXT,
    actor2_type1        TEXT,
    event_code          INT,
    event_base_code     INT,
    event_root_code     INT,
    quad_class          SMALLINT,       -- 1 VerbCoop 2 MatCoop 3 VerbConfl 4 MatConfl
    goldstein           FLOAT,          -- -10 to +10
    num_mentions        INT,
    num_sources         INT,
    num_articles        INT,
    avg_tone            FLOAT,
    action_geo_country  TEXT,
    latitude            FLOAT,
    longitude           FLOAT,
    source_url          TEXT,
    ingested_at         TIMESTAMPTZ     DEFAULT NOW(),
    PRIMARY KEY (global_event_id, event_date)
) PARTITION BY RANGE (event_date);

-- Monthly partitions pre-created for 2020-01 → 2025-12
DO $$
DECLARE
    y INT;
    m INT;
    start_date DATE;
    end_date   DATE;
    tbl_name   TEXT;
BEGIN
    FOR y IN 2020..2026 LOOP
        FOR m IN 1..12 LOOP
            start_date := make_date(y, m, 1);
            end_date   := start_date + INTERVAL '1 month';
            tbl_name   := format('gdelt_events_%s_%s', y, lpad(m::text, 2, '0'));
            EXECUTE format(
                'CREATE TABLE IF NOT EXISTS %I
                 PARTITION OF gdelt_events
                 FOR VALUES FROM (%L) TO (%L)',
                tbl_name, start_date, end_date
            );
        END LOOP;
    END LOOP;
END
$$;

-- Indexes on partitioned table (inherited by all child partitions)
CREATE INDEX IF NOT EXISTS idx_gdelt_country_date
    ON gdelt_events (action_geo_country, event_date DESC);

CREATE INDEX IF NOT EXISTS idx_gdelt_event_code
    ON gdelt_events (event_code, event_date DESC);

CREATE INDEX IF NOT EXISTS idx_gdelt_quad_class
    ON gdelt_events (quad_class, event_date DESC);

CREATE INDEX IF NOT EXISTS idx_gdelt_geo
    ON gdelt_events USING BRIN (event_date);

-- Convert to TimescaleDB hypertable for further time-series optimization
SELECT create_hypertable(
    'gdelt_events', 'event_date',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- ============================================================
-- COUNTRY DAILY FEATURES  (materialized per country per day)
-- ============================================================
CREATE TABLE IF NOT EXISTS country_daily_features (
    country             TEXT        NOT NULL,
    feature_date        DATE        NOT NULL,

    -- Event counts
    total_events        INT         DEFAULT 0,
    conflict_events     INT         DEFAULT 0,
    cooperation_events  INT         DEFAULT 0,

    -- Derived scores (0–1 normalized)
    protest_score       FLOAT       DEFAULT 0.0,
    violence_score      FLOAT       DEFAULT 0.0,
    diplomatic_stress   FLOAT       DEFAULT 0.0,
    economic_stress     FLOAT       DEFAULT 0.0,
    terrorism_score     FLOAT       DEFAULT 0.0,

    -- Sentiment
    avg_sentiment       FLOAT       DEFAULT 0.0,
    avg_goldstein       FLOAT       DEFAULT 0.0,

    -- Composite risk (populated by inference engine)
    risk_score          FLOAT,
    confidence          FLOAT,

    -- Metadata
    computed_at         TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (country, feature_date)
);

CREATE INDEX IF NOT EXISTS idx_cdf_date_risk
    ON country_daily_features (feature_date DESC, risk_score DESC);

CREATE INDEX IF NOT EXISTS idx_cdf_country
    ON country_daily_features (country, feature_date DESC);

-- TimescaleDB hypertable for efficient time-series queries
SELECT create_hypertable(
    'country_daily_features', 'feature_date',
    partitioning_column => 'country',
    number_partitions   => 4,
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists       => TRUE
);

-- Continuous aggregate: 7-day rolling average per country
CREATE MATERIALIZED VIEW IF NOT EXISTS country_weekly_features
WITH (timescaledb.continuous) AS
SELECT
    country,
    time_bucket('7 days', feature_date)        AS week,
    AVG(protest_score)                          AS avg_protest,
    AVG(violence_score)                         AS avg_violence,
    AVG(diplomatic_stress)                      AS avg_diplo_stress,
    AVG(economic_stress)                        AS avg_econ_stress,
    AVG(terrorism_score)                        AS avg_terror,
    AVG(avg_sentiment)                          AS avg_sentiment,
    AVG(avg_goldstein)                          AS avg_goldstein,
    AVG(risk_score)                             AS avg_risk,
    SUM(total_events)                           AS total_events
FROM country_daily_features
GROUP BY country, week
WITH NO DATA;

SELECT add_continuous_aggregate_policy('country_weekly_features',
    start_offset    => INTERVAL '14 days',
    end_offset      => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists   => TRUE
);

-- ============================================================
-- RISK PREDICTIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS country_risk_predictions (
    id                  BIGSERIAL   PRIMARY KEY,
    country             TEXT        NOT NULL,
    prediction_time     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    forecast_horizon    INT         NOT NULL DEFAULT 30,  -- days ahead

    -- Multi-task outputs (0–1)
    risk_score          FLOAT       NOT NULL,
    instability_score   FLOAT,
    war_probability     FLOAT,
    terrorism_risk      FLOAT,
    financial_stress    FLOAT,

    -- Uncertainty
    confidence          FLOAT,
    prediction_variance FLOAT,

    -- Trend
    trend               TEXT CHECK (trend IN ('increasing','stable','decreasing')),

    -- Advisory text
    advisory            TEXT,

    -- Which model version produced this
    model_version       TEXT        DEFAULT 'v0.1'
);

CREATE INDEX IF NOT EXISTS idx_crp_country_time
    ON country_risk_predictions (country, prediction_time DESC);

SELECT create_hypertable(
    'country_risk_predictions', 'prediction_time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists       => TRUE
);

-- ============================================================
-- EVENT EMBEDDINGS  (pgvector – optional Phase 2)
-- ============================================================
CREATE TABLE IF NOT EXISTS event_embeddings (
    global_event_id     BIGINT      PRIMARY KEY,
    event_date          DATE        NOT NULL,
    embedding           vector(384),        -- all-MiniLM-L6-v2 dim
    source_text         TEXT
);

CREATE INDEX IF NOT EXISTS idx_emb_vector
    ON event_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ============================================================
-- INGESTION AUDIT LOG
-- ============================================================
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id              BIGSERIAL   PRIMARY KEY,
    run_time        TIMESTAMPTZ DEFAULT NOW(),
    source_file     TEXT        NOT NULL,
    events_parsed   INT         DEFAULT 0,
    events_inserted INT         DEFAULT 0,
    events_skipped  INT         DEFAULT 0,
    duration_sec    FLOAT,
    status          TEXT        CHECK (status IN ('success','partial','failed')),
    error_message   TEXT
);

-- ============================================================
-- HELPER VIEWS
-- ============================================================
CREATE OR REPLACE VIEW latest_country_risk AS
SELECT DISTINCT ON (country)
    country,
    feature_date,
    risk_score,
    confidence,
    protest_score,
    violence_score,
    diplomatic_stress,
    economic_stress,
    terrorism_score,
    avg_sentiment
FROM country_daily_features
WHERE risk_score IS NOT NULL
ORDER BY country, feature_date DESC;

CREATE OR REPLACE VIEW global_risk_summary AS
SELECT
    feature_date,
    COUNT(*)                    AS countries_tracked,
    AVG(risk_score)             AS global_avg_risk,
    MAX(risk_score)             AS max_risk,
    PERCENTILE_CONT(0.9)
        WITHIN GROUP (ORDER BY risk_score) AS p90_risk
FROM country_daily_features
WHERE risk_score IS NOT NULL
GROUP BY feature_date
ORDER BY feature_date DESC;

-- Grant access
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO gldt;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO gldt;
