-- ============================================================
-- GLDT Phase 2 — PostgreSQL Migration
-- Run AFTER init.sql (Phase 1 schema must already exist)
-- ============================================================

-- ---------------------------------------------------------------------------
-- MULTI-TASK PROXY LABELS
-- Ground-truth proxy labels derived from GDELT event patterns.
-- One row per (country, date). Labels are 0–1 float probabilities.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS country_multitask_labels (
    country             TEXT        NOT NULL,
    label_date          DATE        NOT NULL,

    -- Binary-ish proxy labels (0–1 continuous)
    instability_label   FLOAT       NOT NULL DEFAULT 0.0,
    war_label           FLOAT       NOT NULL DEFAULT 0.0,
    terrorism_label     FLOAT       NOT NULL DEFAULT 0.0,
    financial_label     FLOAT       NOT NULL DEFAULT 0.0,

    -- Metadata
    label_version       TEXT        NOT NULL DEFAULT 'v1',
    event_count         INT         DEFAULT 0,
    computed_at         TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (country, label_date)
);

CREATE INDEX IF NOT EXISTS idx_labels_date
    ON country_multitask_labels (label_date DESC);

CREATE INDEX IF NOT EXISTS idx_labels_war
    ON country_multitask_labels (war_label DESC, label_date DESC);

SELECT create_hypertable(
    'country_multitask_labels', 'label_date',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists       => TRUE
);

-- ---------------------------------------------------------------------------
-- FEATURE ATTRIBUTIONS (SHAP / Integrated Gradients)
-- Stores per-feature importance scores per country prediction.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS feature_attributions (
    id                  BIGSERIAL   PRIMARY KEY,
    country             TEXT        NOT NULL,
    attribution_date    DATE        NOT NULL,
    method              TEXT        NOT NULL DEFAULT 'integrated_gradients',
    model_version       TEXT        NOT NULL DEFAULT 'v0.1',

    -- Attribution scores per feature (same order as FEATURE_COLUMNS)
    protest_attr        FLOAT,
    violence_attr       FLOAT,
    diplomatic_attr     FLOAT,
    economic_attr       FLOAT,
    terrorism_attr      FLOAT,
    sentiment_attr      FLOAT,
    goldstein_attr      FLOAT,

    -- Target head this attribution is for
    target_head         TEXT        DEFAULT 'risk_score',

    computed_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_attr_country_date
    ON feature_attributions (country, attribution_date DESC);

-- ---------------------------------------------------------------------------
-- COUNTRY SPILLOVER NETWORK
-- Pairwise risk correlation + co-occurrence scores.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS country_spillover (
    country_a           TEXT        NOT NULL,
    country_b           TEXT        NOT NULL,
    computed_date       DATE        NOT NULL,

    -- Correlation of risk_score time series (90-day window)
    risk_correlation    FLOAT,

    -- Co-occurrence: events where both countries appear (actor1+actor2)
    cooccurrence_count  INT         DEFAULT 0,
    cooccurrence_score  FLOAT       DEFAULT 0.0,

    -- Geographic adjacency (1 = neighbors, 0 = not)
    is_adjacent         BOOLEAN     DEFAULT FALSE,

    -- Combined spillover weight (0–1)
    spillover_weight    FLOAT,

    PRIMARY KEY (country_a, country_b, computed_date),
    CHECK (country_a < country_b)   -- avoid duplicates (A,B) and (B,A)
);

CREATE INDEX IF NOT EXISTS idx_spillover_a
    ON country_spillover (country_a, spillover_weight DESC);
CREATE INDEX IF NOT EXISTS idx_spillover_b
    ON country_spillover (country_b, spillover_weight DESC);

-- ---------------------------------------------------------------------------
-- EVENT CLUSTERS
-- Pre-aggregated event summaries per country per date, by category.
-- Used for country drilldown UI without hitting raw gdelt_events.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS event_clusters (
    country             TEXT        NOT NULL,
    cluster_date        DATE        NOT NULL,
    category            TEXT        NOT NULL,   -- 'protest','military','terrorism','sanctions','diplomatic'
    event_count         INT         DEFAULT 0,
    total_mentions      INT         DEFAULT 0,
    avg_goldstein       FLOAT,
    avg_tone            FLOAT,
    max_intensity       FLOAT,      -- max |goldstein| in cluster
    top_actor_pairs     JSONB,      -- [{actor1,actor2,count},...]
    computed_at         TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (country, cluster_date, category)
);

CREATE INDEX IF NOT EXISTS idx_clusters_country_date
    ON event_clusters (country, cluster_date DESC);

SELECT create_hypertable(
    'event_clusters', 'cluster_date',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists       => TRUE
);

-- ---------------------------------------------------------------------------
-- ACTOR REGISTRY
-- Deduplicated actor entities extracted from GDELT events.
-- Used for "related actors" display in the UI.
-- ---------------------------------------------------------------------------
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

-- ---------------------------------------------------------------------------
-- GDELT V2 INGESTION CURSOR
-- Tracks which 15-minute files have been processed.
-- ---------------------------------------------------------------------------
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

-- ---------------------------------------------------------------------------
-- MODEL REGISTRY
-- Track model versions, hyperparams, and eval metrics.
-- ---------------------------------------------------------------------------
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

-- ---------------------------------------------------------------------------
-- HELPER VIEWS (Phase 2)
-- ---------------------------------------------------------------------------

-- Countries with labels ready for training
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

-- Top spillover neighbors for each country
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

-- Grant Phase 2 tables
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO gldt;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO gldt;
