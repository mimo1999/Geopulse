"""
Seed the gdelt_risk database from the parquet feature cache.

Run this after schema init (init_local_pg.sql + init_phase3.sql) to populate
country_daily_features and country_risk_predictions from the bi-weekly
parquet snapshots in data/real_cache/.

Also computes a correlation-based country_spillover table.

Usage:
    python scripts/seed_db_from_cache.py [--cache-dir data/real_cache] [--dsn ...]
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed_db")


# ---------------------------------------------------------------------------
# Risk computation (mirrors forecaster_dataset.py)
# ---------------------------------------------------------------------------

def compute_all(row: dict) -> tuple:
    protest  = float(row.get("f0", 0))
    violence = float(row.get("f1", 0))
    diplo    = float(row.get("f2", 0))
    economic = float(row.get("f3", 0))
    terror   = float(row.get("f4", 0))
    goldstein = float(row.get("f5", 0))
    tone     = float(row.get("f6", 0))

    instability = min(0.5 * violence + 0.5 * protest, 1.0)
    war         = min(0.4 * violence + 0.4 * diplo + 0.2 * goldstein, 1.0)
    terrorism   = min(terror * 1.2, 1.0)
    financial   = min(0.7 * economic + 0.3 * tone, 1.0)
    risk        = min(0.40 * instability + 0.30 * war + 0.20 * terrorism + 0.10 * financial, 1.0)
    return protest, violence, diplo, economic, terror, goldstein, tone, instability, war, terrorism, financial, risk


# ---------------------------------------------------------------------------
# Main seed routine
# ---------------------------------------------------------------------------

def seed(cache_dir: str, dsn: str) -> None:
    files = sorted(glob.glob(f"{cache_dir}/*_features.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files found in {cache_dir}")
    logger.info("Found %d parquet snapshots", len(files))

    conn = psycopg2.connect(dsn)

    # ---- 1. country_daily_features ----
    logger.info("Seeding country_daily_features...")
    daily_rows = []
    country_ts: dict = defaultdict(list)

    for fpath in files:
        df = pd.read_parquet(fpath)
        for _, row in df.iterrows():
            r = row.to_dict()
            p, v, d, e, t, g, tone, inst, war, terror, fin, risk = compute_all(r)
            snap_date = r["date"]
            if hasattr(snap_date, "date"):
                snap_date = snap_date.date()
            daily_rows.append((
                r["country"], snap_date,
                p, v, d, e, t, g, tone, risk, 0.75,
            ))
            country_ts[r["country"]].append(risk)

    cur = conn.cursor()
    psycopg2.extras.execute_values(cur, """
        INSERT INTO country_daily_features
            (country, feature_date, protest_score, violence_score, diplomatic_stress,
             economic_stress, terrorism_score, avg_goldstein, avg_sentiment, risk_score, confidence)
        VALUES %s
        ON CONFLICT (country, feature_date) DO UPDATE
          SET risk_score=EXCLUDED.risk_score,
              protest_score=EXCLUDED.protest_score,
              violence_score=EXCLUDED.violence_score,
              diplomatic_stress=EXCLUDED.diplomatic_stress,
              economic_stress=EXCLUDED.economic_stress,
              terrorism_score=EXCLUDED.terrorism_score,
              confidence=EXCLUDED.confidence
    """, daily_rows)
    conn.commit()
    logger.info("Inserted/updated %d daily feature rows", len(daily_rows))

    # ---- 2. country_risk_predictions (latest snapshot only) ----
    logger.info("Seeding country_risk_predictions from latest snapshot...")
    latest_df = pd.read_parquet(files[-1])
    pred_rows = []
    for _, row in latest_df.iterrows():
        r = row.to_dict()
        p, v, d, e, t, g, tone, inst, war, terror, fin, risk = compute_all(r)
        hist = country_ts[r["country"]]
        if len(hist) >= 6:
            delta = np.mean(hist[-3:]) - np.mean(hist[-6:-3])
            trend = "increasing" if delta > 0.02 else "decreasing" if delta < -0.02 else "stable"
        else:
            trend = "stable"
        pred_rows.append((
            r["country"], risk, inst, war, terror, fin,
            0.75, trend, f"Risk {risk:.2f} ({trend})", "v0.3-parquet",
        ))

    psycopg2.extras.execute_values(cur, """
        INSERT INTO country_risk_predictions
            (country, risk_score, instability_score, war_probability, terrorism_risk,
             financial_stress, confidence, trend, advisory, model_version)
        VALUES %s
        ON CONFLICT DO NOTHING
    """, pred_rows)
    conn.commit()
    logger.info("Inserted %d risk prediction rows", len(pred_rows))

    # ---- 3. country_spillover (correlation-based) ----
    logger.info("Computing correlation-based spillover...")
    full_len = len(files)
    countries_full = [c for c, v in country_ts.items() if len(v) == full_len]
    if len(countries_full) >= 2:
        mat = np.array([[country_ts[c][i] for i in range(full_len)] for c in countries_full])
        corr = np.corrcoef(mat)
        THRESHOLD = 0.25
        spill_rows = []
        today = date.today()
        for i, ca in enumerate(countries_full):
            for j, cb in enumerate(countries_full):
                if i >= j:
                    continue
                w = float(corr[i, j])
                if w >= THRESHOLD:
                    spill_rows.append((ca, cb, round(w, 4), today))
        if spill_rows:
            psycopg2.extras.execute_values(cur, """
                INSERT INTO country_spillover (country_a, country_b, spillover_weight, computed_date)
                VALUES %s
                ON CONFLICT DO NOTHING
            """, spill_rows)
            conn.commit()
            logger.info("Inserted %d spillover pairs (threshold=%.2f)", len(spill_rows), THRESHOLD)
    else:
        logger.warning("Not enough countries with full time series for spillover")

    conn.close()
    logger.info("Database seeding complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Seed gdelt_risk DB from parquet cache")
    p.add_argument("--cache-dir", default="data/real_cache")
    p.add_argument("--dsn",       default="postgresql://gldt:gldt_secret@localhost:5432/gdelt_risk")
    args = p.parse_args()
    seed(args.cache_dir, args.dsn)
