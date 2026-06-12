# -*- coding: utf-8 -*-
"""
Evaluate RiskGNN via A/B comparison against carry-forward baseline.

Strategy
--------
For each bi-weekly snapshot date T in the DB (last N dates by default):
  1. Build 12-dim node features from country_daily_features for that date.
  2. Load spillover adjacency from country_spillover.
  3. Run RiskGNN → network_adjusted_risk per country.
  4. Retrieve actual risk_score at T+14 and T+28 from country_daily_features.
  5. Compare MAE(base_risk → actual) vs MAE(network_adjusted_risk → actual).

Note: node features are approximated from country_daily_features for historical
dates (ML sub-scores instability/war/terrorism/financial unavailable historically;
they are proxied by risk_score). Confidence defaults to 0.5. Current-date
evaluation uses the full 12-dim features via the live DB view.

Graph diagnostics
-----------------
  Node count, edge count, density, degree distribution, isolated nodes.

Output distributions
--------------------
  contagion_score and risk_amplification distributions across countries.

Usage:
    python scripts/eval_gnn_spillover.py
    python scripts/eval_gnn_spillover.py --n-dates 30 --horizons 14 28
    python scripts/eval_gnn_spillover.py --model-path models/checkpoints/gnn_best.pt
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("eval_gnn_spillover")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "configs" / "config.yaml") as f:
    _cfg = yaml.safe_load(f)
_db = _cfg["database"]
DSN = (
    f"postgresql://{_db['user']}:{_db['password']}"
    f"@{_db['host']}:{_db['port']}/{_db['name']}"
)

DEFAULT_MODEL_PATH = str(ROOT / "models" / "checkpoints" / "gnn_best.pt")
RESULTS_PATH       = ROOT / "evaluation" / "results" / "gnn_eval.json"

# GNN node feature layout (must match models/gnn.py NODE_FEATURE_DIM = 12)
# [risk_score, instability, war, terrorism, financial, confidence,
#  protest_score, violence_score, diplomatic_stress, economic_stress,
#  terrorism_score, avg_sentiment]
NODE_FEATURE_DIM = 12

# Columns from country_daily_features used to build historical node features.
# First 6 slots (ML sub-scores) are approximated from risk_score.
_HIST_FEATURE_COLS = [
    "risk_score",         # slot 0: also used as proxy for slots 1-5
    "protest_score",      # slot 6
    "violence_score",     # slot 7
    "diplomatic_stress",  # slot 8
    "economic_stress",    # slot 9
    "terrorism_score",    # slot 10
    "avg_sentiment",      # slot 11
]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def fetch_snapshot_dates(n: int) -> list[date]:
    """Return the last `n` distinct feature dates, ascending."""
    with contextlib.closing(psycopg2.connect(DSN)) as conn:
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT feature_date FROM country_daily_features "
                "ORDER BY feature_date DESC LIMIT %s",
                (n,),
            )
            dates = [r[0] for r in cur.fetchall()]
    return sorted(dates)


def fetch_node_features_at(snapshot_date: date) -> dict[str, np.ndarray]:
    """
    Build 12-dim node feature vectors for all countries at a historical date.

    ML sub-scores (instability, war, terrorism, financial, confidence) are not
    stored historically — they are proxied by risk_score (slot 0) with
    confidence defaulting to 0.5. This introduces approximation error but
    keeps the evaluation self-contained without requiring stored model outputs.
    """
    cols = ", ".join(_HIST_FEATURE_COLS)
    with contextlib.closing(psycopg2.connect(DSN)) as conn:
        with conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT country, {cols} FROM country_daily_features "
                f"WHERE feature_date = %s",
                (snapshot_date,),
            )
            rows = cur.fetchall()

    result: dict[str, np.ndarray] = {}
    for row in rows:
        country     = row[0]
        risk_score  = float(row[1] or 0.0)
        protest     = float(row[2] or 0.0)
        violence    = float(row[3] or 0.0)
        diplomatic  = float(row[4] or 0.0)
        economic    = float(row[5] or 0.0)
        terrorism   = float(row[6] or 0.0)
        sentiment   = float(row[7] or 0.0)

        # Slots 0-5: risk_score proxy for ML sub-scores; confidence=0.5
        node_feat = np.array([
            risk_score,   # risk_score
            risk_score,   # instability (proxy)
            risk_score,   # war (proxy)
            risk_score,   # terrorism (proxy)
            risk_score,   # financial (proxy)
            0.5,          # confidence (default)
            protest,
            violence,
            diplomatic,
            economic,
            terrorism,
            sentiment,
        ], dtype=np.float32)

        result[country] = node_feat
    return result


def fetch_spillover() -> list[dict]:
    """Fetch the latest spillover edges from country_spillover."""
    with contextlib.closing(psycopg2.connect(DSN)) as conn:
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT country_a, country_b, spillover_weight "
                "FROM country_spillover "
                "WHERE computed_date = (SELECT MAX(computed_date) FROM country_spillover)"
            )
            rows = cur.fetchall()
    return [{"country_a": r[0], "country_b": r[1], "spillover_weight": r[2]} for r in rows]


def fetch_actual_risks(countries: list[str], target_date: date) -> dict[str, float]:
    """Fetch actual risk_score for all countries at target_date."""
    with contextlib.closing(psycopg2.connect(DSN)) as conn:
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT country, risk_score FROM country_daily_features "
                "WHERE feature_date = %s AND country = ANY(%s) "
                "AND risk_score IS NOT NULL",
                (target_date, countries),
            )
            rows = cur.fetchall()
    return {r[0]: float(r[1]) for r in rows}


# ---------------------------------------------------------------------------
# Graph diagnostics
# ---------------------------------------------------------------------------

def graph_stats(adj: torch.Tensor, countries: list[str]) -> dict:
    N    = len(countries)
    mask = (adj > 0).float()
    # Exclude self-loops from edge count
    off_diag = mask - torch.eye(N)
    n_edges  = int(off_diag.clamp(min=0).sum().item()) // 2  # undirected

    degrees       = off_diag.clamp(min=0).sum(dim=1).numpy()
    isolated      = int((degrees == 0).sum())
    max_possible  = N * (N - 1) // 2
    density       = n_edges / max_possible if max_possible > 0 else 0.0

    return {
        "n_nodes":       N,
        "n_edges":       n_edges,
        "density":       round(density, 4),
        "isolated_nodes": isolated,
        "degree_mean":   round(float(degrees.mean()), 2),
        "degree_max":    int(degrees.max()),
    }


# ---------------------------------------------------------------------------
# A/B evaluation
# ---------------------------------------------------------------------------

def run_ab_eval(
    model: "RiskGNN",
    spillover_rows: list[dict],
    snapshot_dates: list[date],
    horizons: list[int],
    min_edge_weight: float,
) -> dict:
    """
    For each snapshot date, run GNN and compare base_risk vs network_adjusted_risk
    as predictors of actual future risk at each horizon.
    """
    from models.gnn import build_adjacency

    records: dict[int, list[tuple[float, float, float]]] = {h: [] for h in horizons}

    for pivot in snapshot_dates:
        node_data = fetch_node_features_at(pivot)
        if not node_data:
            continue

        countries   = list(node_data.keys())
        country_idx = {c: i for i, c in enumerate(countries)}
        N           = len(countries)

        x = np.zeros((N, NODE_FEATURE_DIM), dtype=np.float32)
        for c, feats in node_data.items():
            x[country_idx[c]] = feats

        adj = build_adjacency(country_idx, spillover_rows, min_edge_weight)
        x_t = torch.from_numpy(x)

        with torch.no_grad():
            out = model(x_t, adj)

        amplif         = out["risk_amplification"].numpy()        # (N,)
        base_risks     = x[:, 0]                                  # risk_score slot
        adjusted_risks = np.clip(base_risks + amplif * 0.15, 0, 1)

        for horizon in horizons:
            target_date = pivot + timedelta(days=horizon)
            actuals     = fetch_actual_risks(countries, target_date)
            if not actuals:
                continue

            for country, actual in actuals.items():
                idx  = country_idx[country]
                base = float(base_risks[idx])
                adj_ = float(adjusted_risks[idx])
                records[horizon].append((base, adj_, actual))

        logger.info("  Processed pivot %s (%d countries)", pivot, N)

    results = {}
    for horizon, recs in records.items():
        if not recs:
            results[f"h{horizon}"] = {"n": 0}
            continue
        arr        = np.array(recs)                # (M, 3)
        base       = arr[:, 0]
        adjusted   = arr[:, 1]
        actuals    = arr[:, 2]

        base_mae   = float(np.mean(np.abs(base - actuals)))
        adj_mae    = float(np.mean(np.abs(adjusted - actuals)))
        skill      = (base_mae - adj_mae) / base_mae if base_mae > 0 else 0.0
        corr_base  = float(np.corrcoef(base, actuals)[0, 1])
        corr_adj   = float(np.corrcoef(adjusted, actuals)[0, 1])

        results[f"h{horizon}"] = {
            "n":               len(recs),
            "base_mae":        round(base_mae, 4),
            "adjusted_mae":    round(adj_mae, 4),
            "gnn_skill_score": round(skill, 4),
            "corr_base":       round(corr_base, 4),
            "corr_adjusted":   round(corr_adj, 4),
        }
    return results


# ---------------------------------------------------------------------------
# Output distribution stats
# ---------------------------------------------------------------------------

def output_distribution(gnn_out: dict, countries: list[str]) -> dict:
    contagion = gnn_out["contagion_score"].numpy()
    amplif    = gnn_out["risk_amplification"].numpy()

    def _stats(arr: np.ndarray) -> dict:
        return {
            "mean": round(float(arr.mean()), 4),
            "std":  round(float(arr.std()),  4),
            "min":  round(float(arr.min()),  4),
            "p25":  round(float(np.percentile(arr, 25)), 4),
            "p50":  round(float(np.percentile(arr, 50)), 4),
            "p75":  round(float(np.percentile(arr, 75)), 4),
            "max":  round(float(arr.max()),  4),
        }

    top5_contagion = sorted(
        zip(countries, contagion.tolist()),
        key=lambda x: x[1], reverse=True,
    )[:5]
    top5_amplified = sorted(
        zip(countries, amplif.tolist()),
        key=lambda x: x[1], reverse=True,
    )[:5]
    top5_dampened  = sorted(
        zip(countries, amplif.tolist()),
        key=lambda x: x[1],
    )[:5]

    return {
        "contagion_score":    _stats(contagion),
        "risk_amplification": _stats(amplif),
        "top5_contagion":     [{"country": c, "score": round(s, 4)} for c, s in top5_contagion],
        "top5_amplified":     [{"country": c, "delta": round(s, 4)} for c, s in top5_amplified],
        "top5_dampened":      [{"country": c, "delta": round(s, 4)} for c, s in top5_dampened],
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    graph: dict,
    dist: dict,
    ab: dict,
    model_path: str,
    n_dates: int,
) -> None:
    SEP = "=" * 70

    print("\n" + SEP)
    print("  GEOPULSE  --  RISK GNN EVALUATION")
    print(SEP)
    print(f"  Model    : {model_path}")
    print(f"  Eval dates evaluated : {n_dates}")
    print()

    print("-- Graph Structure " + "-" * 51)
    print(f"  Nodes          : {graph['n_nodes']}")
    print(f"  Edges          : {graph['n_edges']}")
    print(f"  Density        : {graph['density']:.4f}")
    print(f"  Isolated nodes : {graph['isolated_nodes']}")
    print(f"  Avg degree     : {graph['degree_mean']:.1f}  (max {graph['degree_max']})")
    print()

    print("-- Output Distributions (current run) " + "-" * 32)
    cs = dist["contagion_score"]
    ra = dist["risk_amplification"]
    print(f"  contagion_score   mean={cs['mean']:.4f}  std={cs['std']:.4f}  "
          f"[{cs['min']:.4f}, {cs['max']:.4f}]")
    print(f"  risk_amplif.      mean={ra['mean']:.4f}  std={ra['std']:.4f}  "
          f"[{ra['min']:.4f}, {ra['max']:.4f}]")
    print()
    print("  Top-5 contagion importers:")
    for e in dist["top5_contagion"]:
        print(f"    {e['country']:>4}  {e['score']:.4f}")
    print("  Top-5 risk-amplified countries (network increases risk):")
    for e in dist["top5_amplified"]:
        print(f"    {e['country']:>4}  {e['delta']:+.4f}")
    print("  Top-5 risk-dampened countries (network lowers risk):")
    for e in dist["top5_dampened"]:
        print(f"    {e['country']:>4}  {e['delta']:+.4f}")
    print()

    print("-- A/B Comparison: GNN vs Carry-Forward " + "-" * 30)
    print(f"  {'Horizon':>8}  {'N':>6}  {'Base MAE':>10}  {'GNN MAE':>10}  {'Skill%':>8}  "
          f"{'Corr base':>10}  {'Corr GNN':>10}")
    print("  " + "-" * 70)
    for key in sorted(ab):
        m = ab[key]
        if m.get("n", 0) == 0:
            print(f"  {key:>8}  {'n/a':>6}")
            continue
        horizon_str = key.replace("h", "") + "d"
        skill_sign  = "+" if m["gnn_skill_score"] >= 0 else ""
        print(
            f"  {horizon_str:>8}  {m['n']:>6}  {m['base_mae']:>10.4f}  {m['adjusted_mae']:>10.4f}"
            f"  {skill_sign}{m['gnn_skill_score']:>7.1%}  {m['corr_base']:>10.4f}  {m['corr_adjusted']:>10.4f}"
        )
    print()

    print("-- Interpretation " + "-" * 52)
    for key, m in ab.items():
        if m.get("n", 0) == 0:
            continue
        horizon_str = key.replace("h", "") + "d"
        if m["gnn_skill_score"] > 0.02:
            print(f"  [+] {horizon_str}: GNN enrichment reduces MAE by {m['gnn_skill_score']:.1%}")
        elif m["gnn_skill_score"] > 0:
            print(f"  [~] {horizon_str}: Marginal GNN improvement ({m['gnn_skill_score']:.1%})")
        else:
            print(f"  [-] {horizon_str}: GNN not improving over carry-forward (retrain recommended)")
    print(SEP + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RiskGNN A/B vs carry-forward")
    parser.add_argument("--model-path",      default=DEFAULT_MODEL_PATH)
    parser.add_argument("--n-dates",         type=int, default=20,
                        help="Number of recent bi-weekly snapshot dates to replay (default 20)")
    parser.add_argument("--horizons",        type=int, nargs="+", default=[14, 28],
                        help="Forecast horizons in days (default: 14 28)")
    parser.add_argument("--min-edge-weight", type=float, default=0.15)
    parser.add_argument("--output",          default=str(RESULTS_PATH))
    args = parser.parse_args()

    from models.gnn import RiskGNN, build_adjacency, NODE_FEATURE_DIM as NFD

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model_path = Path(args.model_path)
    if model_path.exists():
        model = RiskGNN.load(str(model_path))
        logger.info("Loaded RiskGNN from %s (%d params)", model_path.name, model.parameter_count())
    else:
        logger.warning(
            "GNN checkpoint not found at %s — using default (untrained) weights. "
            "Train the GNN first with: python scripts/train_gnn.py",
            model_path,
        )
        model = RiskGNN(node_features=NFD, hidden=64, out_features=8, num_layers=2)

    model.eval()

    # ------------------------------------------------------------------
    # Load current-state data for graph stats and output distributions
    # ------------------------------------------------------------------
    logger.info("Fetching current node features for graph diagnostics ...")
    from inference.gnn_spillover import GNNSpilloverEngine

    engine      = GNNSpilloverEngine(dsn=DSN, model_path=args.model_path if model_path.exists() else None)
    node_data   = engine._fetch_node_features()
    spillover_rows = fetch_spillover()

    if not node_data:
        logger.error("No node features available. Ensure the feature pipeline has run.")
        sys.exit(1)

    countries   = list(node_data.keys())
    country_idx = {c: i for i, c in enumerate(countries)}
    N           = len(countries)

    x = np.zeros((N, NODE_FEATURE_DIM), dtype=np.float32)
    for c, feats in node_data.items():
        x[country_idx[c]] = feats

    adj = build_adjacency(country_idx, spillover_rows, args.min_edge_weight)
    x_t = torch.from_numpy(x)
    adj_t = adj

    with torch.no_grad():
        gnn_out = model(x_t, adj_t)

    graph = graph_stats(adj, countries)
    dist  = output_distribution(gnn_out, countries)

    logger.info(
        "Graph: %d nodes, %d edges, density %.4f",
        graph["n_nodes"], graph["n_edges"], graph["density"],
    )

    # ------------------------------------------------------------------
    # Historical A/B evaluation
    # ------------------------------------------------------------------
    logger.info("Fetching %d snapshot dates for A/B replay ...", args.n_dates)
    snapshot_dates = fetch_snapshot_dates(args.n_dates)
    if not snapshot_dates:
        logger.error("No snapshot dates found in country_daily_features.")
        sys.exit(1)

    logger.info(
        "Replaying %d dates (%s -> %s) ...",
        len(snapshot_dates), snapshot_dates[0], snapshot_dates[-1],
    )
    ab_results = run_ab_eval(
        model=model,
        spillover_rows=spillover_rows,
        snapshot_dates=snapshot_dates,
        horizons=args.horizons,
        min_edge_weight=args.min_edge_weight,
    )

    # ------------------------------------------------------------------
    # Report + save
    # ------------------------------------------------------------------
    print_report(graph, dist, ab_results, str(model_path), len(snapshot_dates))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_path":    str(model_path),
        "n_dates":       len(snapshot_dates),
        "date_range":    [str(snapshot_dates[0]), str(snapshot_dates[-1])],
        "graph":         graph,
        "distributions": dist,
        "ab_comparison": ab_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
