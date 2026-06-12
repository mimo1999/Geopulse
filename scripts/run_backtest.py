# -*- coding: utf-8 -*-
"""
Run the walk-forward backtester and print a formatted performance report.

Usage:
    python scripts/run_backtest.py [--countries US RS UP ...]

Results are saved to evaluation/results/backtest_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("run_backtest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "configs" / "config.yaml") as f:
    _cfg = yaml.safe_load(f)
_db = _cfg["database"]
DSN = (
    f"postgresql://{_db['user']}:{_db['password']}"
    f"@{_db['host']}:{_db['port']}/{_db['name']}"
)
MODEL_PATH = str(ROOT / "models" / "checkpoints" / "forecaster_v1_best.pt")
RESULTS_PATH = ROOT / "evaluation" / "results" / "backtest_results.json"


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

TIER_ORDER = ["CRITICAL", "HIGH", "ELEVATED", "MODERATE", "LOW"]


def print_report(results: dict) -> None:
    m   = results["metrics"]
    SEP = "=" * 66

    print("\n" + SEP)
    print("  GEOPULSE  --  WALK-FORWARD BACKTEST REPORT")
    print(SEP)
    print(f"  Model      : {m['model_name']}")
    print(f"  Run date   : {results['run_date']}")
    print(f"  Countries  : {results['n_countries']}")
    print(f"  Test folds : {results['n_folds']}  (bi-weekly, expanding window)")
    print(f"  Predictions: {results['n_predictions']}  (with matched actuals)")
    print()

    print("-- Overall " + "-" * 55)
    print(f"  MAE          {m['overall_mae']:.4f}")
    print(f"  RMSE         {m['overall_rmse']:.4f}")
    skill_pct = m["overall_skill_score"] * 100
    sign = "+" if skill_pct >= 0 else ""
    print(f"  Skill score  {sign}{skill_pct:.1f}%  vs carry-forward baseline")
    print()

    print("-- Per-Horizon (days ahead) " + "-" * 38)
    print(f"  {'Days':>5}  {'N':>6}  {'MAE':>7}  {'RMSE':>7}  "
          f"{'MAPE%':>7}  {'DirAcc':>7}  {'CI Cov':>7}  {'Skill%':>8}")
    print("  " + "-" * 62)
    for h in m["horizons"]:
        skill = (h["skill_score"] or 0) * 100
        mape  = f"{h['mape']:.1f}" if h.get("mape") is not None else "  n/a"
        print(
            f"  {h['horizon_days']:>5}  {h['n_samples']:>6}  "
            f"{h['mae']:>7.4f}  {h['rmse']:>7.4f}  "
            f"{mape:>7}  {h['directional_accuracy']:>7.1%}  "
            f"{h['ci_coverage']:>7.1%}  {skill:>+8.1f}%"
        )
    print()

    print("-- Per Risk Tier (origin score at pivot date) " + "-" * 20)
    print(f"  {'Tier':>10}  {'N':>6}  {'MAE':>7}  {'RMSE':>7}")
    print("  " + "-" * 36)
    tier_data = m.get("per_tier", {})
    for tier in TIER_ORDER:
        if tier in tier_data:
            td = tier_data[tier]
            print(f"  {tier:>10}  {td['n']:>6}  {td['mae']:>7.4f}  {td['rmse']:>7.4f}")
    print()

    print("-- Interpretation " + "-" * 48)
    if m["overall_skill_score"] > 0.05:
        print(f"  [+] Model beats carry-forward baseline by {m['overall_skill_score']*100:.1f}%.")
    elif m["overall_skill_score"] > 0:
        print("  [~] Model marginally beats baseline -- consider more training data.")
    else:
        print("  [-] Baseline beats model -- check feature quality or retrain.")

    best_h  = min(m["horizons"], key=lambda h: h["mae"])
    worst_h = max(m["horizons"], key=lambda h: h["mae"])
    print(f"  Best horizon : {best_h['horizon_days']}d  (MAE {best_h['mae']:.4f})")
    print(f"  Worst horizon: {worst_h['horizon_days']}d  (MAE {worst_h['mae']:.4f})")

    avg_dir = sum(h["directional_accuracy"] for h in m["horizons"]) / len(m["horizons"])
    print(f"  Dir accuracy : {avg_dir:.1%}  (random baseline = 50%)")

    avg_ci = sum(h["ci_coverage"] for h in m["horizons"]) / len(m["horizons"])
    if avg_ci >= 0.75:
        print(f"  CI coverage  : {avg_ci:.1%}  -- well-calibrated 80% intervals")
    else:
        print(f"  CI coverage  : {avg_ci:.1%}  -- intervals too narrow (target >=80%)")

    print(SEP + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GeoPulse walk-forward backtest")
    parser.add_argument(
        "--countries", nargs="*", default=None,
        help="Country codes to include (default: all with sufficient history)",
    )
    parser.add_argument(
        "--burn-in", type=int, default=30,
        help="Number of snapshot dates to skip before testing (default: 30)",
    )
    args = parser.parse_args()

    from evaluation.backtester import WalkForwardBacktester

    backtester = WalkForwardBacktester(
        dsn=DSN,
        model_path=MODEL_PATH,
        burn_in=args.burn_in,
        countries=args.countries,
    )

    logger.info("Starting walk-forward backtest...")
    results = backtester.run()

    results.save(RESULTS_PATH)

    print_report(results.to_dict())
    logger.info("Results written to %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
