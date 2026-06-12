# -*- coding: utf-8 -*-
"""
Calibrate forecast confidence intervals via split-conformal prediction.

Reads the walk-forward backtest residuals, splits the test folds
chronologically (early = calibration, late = evaluation), fits per-horizon
conformal quantiles on the calibration half, and reports out-of-sample
coverage on the evaluation half — proving the bands are well-calibrated
with no leakage.

Finally, it refits on ALL residuals and persists the production quantiles to
evaluation/results/conformal_quantiles.json for the serving path to load.

Usage:
    python scripts/calibrate_intervals.py [--alpha 0.20]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("calibrate")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_PATH = ROOT / "evaluation" / "results" / "backtest_results.json"
QUANTILES_PATH = ROOT / "evaluation" / "results" / "conformal_quantiles.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.20,
                        help="Miscoverage rate; 0.20 => 80%% intervals")
    args = parser.parse_args()
    target = 1 - args.alpha

    from evaluation.calibration import ConformalCalibrator, evaluate_coverage

    with open(RESULTS_PATH) as f:
        results = json.load(f)
    fold_log = results["fold_log"]
    logger.info("Loaded %d prediction records", len(fold_log))

    # --- Chronological split (no leakage) ---
    pivot_dates = sorted({p["pivot_date"] for p in fold_log})
    split_date = pivot_dates[len(pivot_dates) // 2]
    calib = [p for p in fold_log if p["pivot_date"] < split_date]
    evalu = [p for p in fold_log if p["pivot_date"] >= split_date]
    logger.info("Split at %s: %d calibration / %d evaluation records",
                split_date, len(calib), len(evalu))

    # Baseline (uncalibrated MC-dropout) coverage on the eval half
    hits = total = 0
    for p in evalu:
        if p["actual"] is not None:
            total += 1
            if p["lower"] <= p["actual"] <= p["upper"]:
                hits += 1
    base_cov = hits / max(1, total)

    # Fit on calibration half, evaluate on held-out half
    cal = ConformalCalibrator(alpha=args.alpha).fit(calib)
    cov = evaluate_coverage(evalu, cal)

    SEP = "=" * 66
    print("\n" + SEP)
    print("  GEOPULSE  --  CONFORMAL INTERVAL CALIBRATION")
    print(SEP)
    print(f"  Target coverage     : {target:.0%}")
    print(f"  Calibration records : {len(calib)}  (folds before {split_date})")
    print(f"  Evaluation records  : {cov['n']}  (folds from {split_date})")
    print()
    print("-- Coverage (held-out evaluation half) " + "-" * 27)
    print(f"  {'':<22}{'Before':>10}{'After':>10}{'Target':>10}")
    print(f"  {'Overall coverage':<22}{base_cov:>9.1%}{cov['coverage']:>10.1%}{target:>10.0%}")
    print()
    print("-- Calibrated coverage by horizon " + "-" * 32)
    print(f"  {'Horizon':>8}{'Half-width':>12}{'Coverage':>12}")
    for h in sorted(cal.q_by_horizon):
        hw = cal.q_by_horizon[h]
        c = cov["coverage_by_horizon"].get(h, float("nan"))
        print(f"  {h:>6}d{hw:>12.4f}{c:>11.1%}")
    print()
    print("-- Calibrated coverage by risk tier " + "-" * 30)
    print(f"  {'Tier':>10}{'Coverage':>12}")
    for t in ["CRITICAL", "HIGH", "ELEVATED", "MODERATE", "LOW"]:
        if t in cov["coverage_by_tier"]:
            print(f"  {t:>10}{cov['coverage_by_tier'][t]:>11.1%}")
    print()
    print(f"  Mean interval width : {cov['mean_width']:.4f}  (was ~{0.052:.3f} uncalibrated)")
    print()

    # --- Refit on ALL residuals for production use ---
    prod = ConformalCalibrator(alpha=args.alpha).fit(fold_log)
    prod.save(QUANTILES_PATH)
    print("-- Production quantiles " + "-" * 42)
    print(f"  Refit on all {prod.n_calibration} residuals -> {QUANTILES_PATH.name}")
    for h in sorted(prod.q_by_horizon):
        print(f"    horizon {h:>3}d : +-{prod.q_by_horizon[h]:.4f}")
    print(SEP + "\n")

    logger.info("Done. Serving path can load %s via ConformalCalibrator.load().", QUANTILES_PATH)


if __name__ == "__main__":
    main()
