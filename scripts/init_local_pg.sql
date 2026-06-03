-- ============================================================
-- GLDT Project — Local PostgreSQL 18 Init Script
-- Plain PG18 schema (no TimescaleDB / PostGIS / pgvector).
-- Run against the gdelt_risk database as superuser:
--   psql -U postgres -d gdelt_risk -f scripts/init_local_pg.sql
-- ============================================================

-- ---------------------
-- Extensions (standard only)
-- ---------------------
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- ---------------------
-- Application user
-- ---------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'gldt') THEN
        CREATE ROLE gldt LOGIN PASSWORD 'gldt';
    END IF;
END
$$;

GRANT ALL PRIVILEGES ON DATABASE gdelt_risk TO gldt;

-- ---------------------
-- ENUM types
-- ---------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'quad_class_enum') THEN
        CREATE TYPE quad_class_enum AS ENUM (
            'verbal_cooperation',
            'material_cooperation',
            'verbal_conflict',
            'material_conflict'
        );
    END IF;
END
$$;

-- ============================================================
-- RAW GDELT EVENTS  (range-partitioned by month on event_date)
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
    quad_class          SMALLINT,
    goldstein           FLOAT,
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

-- Monthly partitions 2020-01 → 2026-12
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

CREATE INDEX IF NOT EXISTS idx_gdelt_country_date
    ON gdelt_events (action_geo_country, event_date DESC);
CREATE INDEX IF NOT EXISTS idx_gdelt_event_code
    ON gdelt_events (event_code, event_date DESC);
CREATE INDEX IF NOT EXISTS idx_gdelt_quad_class
    ON gdelt_events (quad_class, event_date DESC);

-- ============================================================
-- COUNTRY DAILY FEATURES
-- ============================================================
CREATE TABLE IF NOT EXISTS country_daily_features (
    country             TEXT        NOT NULL,
    feature_date        DATE        NOT NULL,

    total_events        INT         DEFAULT 0,
    conflict_events     INT         DEFAULT 0,
    cooperation_events  INT         DEFAULT 0,

    protest_score       FLOAT       DEFAULT 0.0,
    violence_score      FLOAT       DEFAULT 0.0,
    diplomatic_stress   FLOAT       DEFAULT 0.0,
    economic_stress     FLOAT       DEFAULT 0.0,
    terrorism_score     FLOAT       DEFAULT 0.0,

    avg_sentiment       FLOAT       DEFAULT 0.0,
    avg_goldstein       FLOAT       DEFAULT 0.0,

    risk_score          FLOAT,
    confidence          FLOAT,

    computed_at         TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (country, feature_date)
);

CREATE INDEX IF NOT EXISTS idx_cdf_date_risk
    ON country_daily_features (feature_date DESC, risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_cdf_country
    ON country_daily_features (country, feature_date DESC);

-- Weekly roll-up view (replaces TimescaleDB continuous aggregate)
CREATE OR REPLACE VIEW country_weekly_features AS
SELECT
    country,
    DATE_TRUNC('week', feature_date)::DATE  AS week,
    AVG(protest_score)                       AS avg_protest,
    AVG(violence_score)                      AS avg_violence,
    AVG(diplomatic_stress)                   AS avg_diplo_stress,
    AVG(economic_stress)                     AS avg_econ_stress,
    AVG(terrorism_score)                     AS avg_terror,
    AVG(avg_sentiment)                       AS avg_sentiment,
    AVG(avg_goldstein)                       AS avg_goldstein,
    AVG(risk_score)                          AS avg_risk,
    SUM(total_events)                        AS total_events
FROM country_daily_features
GROUP BY country, DATE_TRUNC('week', feature_date);

-- ============================================================
-- RISK PREDICTIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS country_risk_predictions (
    id                  BIGSERIAL   PRIMARY KEY,
    country             TEXT        NOT NULL,
    prediction_time     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    forecast_horizon    INT         NOT NULL DEFAULT 30,

    risk_score          FLOAT       NOT NULL,
    instability_score   FLOAT,
    war_probability     FLOAT,
    terrorism_risk      FLOAT,
    financial_stress    FLOAT,

    confidence          FLOAT,
    prediction_variance FLOAT,

    trend               TEXT CHECK (trend IN ('increasing','stable','decreasing')),
    advisory            TEXT,
    model_version       TEXT        DEFAULT 'v0.1'
);

CREATE INDEX IF NOT EXISTS idx_crp_country_time
    ON country_risk_predictions (country, prediction_time DESC);

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
-- Phase 2: MULTI-TASK PROXY LABELS
-- ============================================================
CREATE TABLE IF NOT EXISTS country_multitask_labels (
    country             TEXT        NOT NULL,
    label_date          DATE        NOT NULL,

    instability_label   FLOAT       NOT NULL DEFAULT 0.0,
    war_label           FLOAT       NOT NULL DEFAULT 0.0,
    terrorism_label     FLOAT       NOT NULL DEFAULT 0.0,
    financial_label     FLOAT       NOT NULL DEFAULT 0.0,

    label_version       TEXT        NOT NULL DEFAULT 'v1',
    event_count         INT         DEFAULT 0,
    computed_at         TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (country, label_date)
);

CREATE INDEX IF NOT EXISTS idx_labels_date
    ON country_multitask_labels (label_date DESC);
CREATE INDEX IF NOT EXISTS idx_labels_war
    ON country_multitask_labels (war_label DESC, label_date DESC);

-- ============================================================
-- Phase 2: FEATURE ATTRIBUTIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS feature_attributions (
    id                  BIGSERIAL   PRIMARY KEY,
    country             TEXT        NOT NULL,
    attribution_date    DATE        NOT NULL,
    method              TEXT        NOT NULL DEFAULT 'integrated_gradients',
    model_version       TEXT        NOT NULL DEFAULT 'v0.1',

    protest_attr        FLOAT,
    violence_attr       FLOAT,
    diplomatic_attr     FLOAT,
    economic_attr       FLOAT,
    terrorism_attr      FLOAT,
    sentiment_attr      FLOAT,
    goldstein_attr      FLOAT,

    target_head         TEXT        DEFAULT 'risk_score',
    computed_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_attr_country_date
    ON feature_attributions (country, attribution_date DESC);

-- ============================================================
-- Phase 2: COUNTRY SPILLOVER NETWORK
-- ============================================================
CREATE TABLE IF NOT EXISTS country_spillover (
    country_a           TEXT        NOT NULL,
    country_b           TEXT        NOT NULL,
    computed_date       DATE        NOT NULL,

    risk_correlation    FLOAT,
    cooccurrence_count  INT         DEFAULT 0,
    cooccurrence_score  FLOAT       DEFAULT 0.0,
    is_adjacent         BOOLEAN     DEFAULT FALSE,
    spillover_weight    FLOAT,

    PRIMARY KEY (country_a, country_b, computed_date),
    CHECK (country_a <= country_b)
);

CREATE INDEX IF NOT EXISTS idx_spillover_a
    ON country_spillover (country_a, spillover_weight DESC);
CREATE INDEX IF NOT EXISTS idx_spillover_b
    ON country_spillover (country_b, spillover_weight DESC);

-- ============================================================
-- Phase 2: EVENT CLUSTERS
-- ============================================================
CREATE TABLE IF NOT EXISTS event_clusters (
    country             TEXT        NOT NULL,
    cluster_date        DATE        NOT NULL,
    category            TEXT        NOT NULL,
    event_count         INT         DEFAULT 0,
    total_mentions      INT         DEFAULT 0,
    avg_goldstein       FLOAT,
    avg_tone            FLOAT,
    max_intensity       FLOAT,
    top_actor_pairs     JSONB,
    computed_at         TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (country, cluster_date, category)
);

CREATE INDEX IF NOT EXISTS idx_clusters_country_date
    ON event_clusters (country, cluster_date DESC);

-- ============================================================
-- Phase 2: ACTOR REGISTRY
-- ============================================================
CREATE TABLE IF NOT EXISTS actor_registry (
    actor_code          TEXT        PRIMARY KEY,
    actor_name          TEXT,
    actor_country       TEXT,
    actor_type          TEXT,
    mention_count       BIGINT      DEFAULT 0,
    last_seen           DATE,
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_actor_country
    ON actor_registry (actor_country, mention_count DESC);

-- ============================================================
-- Phase 2: GDELT V2 INGESTION CURSOR
-- ============================================================
CREATE TABLE IF NOT EXISTS gdelt_v2_cursor (
    id                  BIGSERIAL   PRIMARY KEY,
    file_url            TEXT        UNIQUE NOT NULL,
    file_timestamp      TIMESTAMPTZ NOT NULL,
    status              TEXT        NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending','processing','done','error')),
    events_inserted     INT         DEFAULT 0,
    processed_at        TIMESTAMPTZ,
    error_message       TEXT
);

CREATE INDEX IF NOT EXISTS idx_v2_cursor_status
    ON gdelt_v2_cursor (status, file_timestamp);

-- ============================================================
-- Phase 2: MODEL REGISTRY
-- ============================================================
CREATE TABLE IF NOT EXISTS model_registry (
    id                  SERIAL      PRIMARY KEY,
    version             TEXT        UNIQUE NOT NULL,
    phase               INT         NOT NULL DEFAULT 2,
    checkpoint_path     TEXT,
    train_loss          FLOAT,
    val_loss            FLOAT,
    test_loss           FLOAT,
    num_params          INT,
    hyperparams         JSONB,
    trained_at          TIMESTAMPTZ DEFAULT NOW(),
    is_active           BOOLEAN     DEFAULT FALSE
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

CREATE OR REPLACE VIEW labeled_countries AS
SELECT
    l.country,
    MIN(l.label_date)       AS first_label,
    MAX(l.label_date)       AS last_label,
    COUNT(*)                AS label_days,
    AVG(l.war_label)        AS avg_war_label,
    AVG(l.terrorism_label)  AS avg_terror_label
FROM country_multitask_labels l
GROUP BY l.country
HAVING COUNT(*) >= 30
ORDER BY label_days DESC;

CREATE OR REPLACE VIEW country_top_neighbors AS
SELECT
    country_a AS country,
    country_b AS neighbor,
    spillover_weight,
    risk_correlation,
    computed_date
FROM (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY country_a
        ORDER BY spillover_weight DESC
    ) AS rn
    FROM country_spillover
    WHERE computed_date = (SELECT MAX(computed_date) FROM country_spillover)
) ranked
WHERE rn <= 5
UNION ALL
SELECT
    country_b AS country,
    country_a AS neighbor,
    spillover_weight,
    risk_correlation,
    computed_date
FROM (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY country_b
        ORDER BY spillover_weight DESC
    ) AS rn
    FROM country_spillover
    WHERE computed_date = (SELECT MAX(computed_date) FROM country_spillover)
) ranked
WHERE rn <= 5;

-- ============================================================
-- GRANTS
-- ============================================================
GRANT ALL PRIVILEGES ON ALL TABLES    IN SCHEMA public TO gldt;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO gldt;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL ON TABLES    TO gldt;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL ON SEQUENCES TO gldt;
