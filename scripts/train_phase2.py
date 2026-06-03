"""
CLI: Train Phase 2 multi-task model.

Requires:
  - PostgreSQL with country_daily_features AND country_multitask_labels populated
  - Run scripts/phase2_setup.py first

Usage:
    python scripts/train_phase2.py
    python scripts/train_phase2.py --epochs 80 --d-model 256
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Train Phase 2 risk model")
    parser.add_argument("--epochs",   type=int,   default=60)
    parser.add_argument("--d-model",  type=int,   default=128)
    parser.add_argument("--batch",    type=int,   default=64)
    parser.add_argument("--lr",       type=float, default=2e-4)
    parser.add_argument("--dropout",  type=float, default=0.15)
    parser.add_argument("--seq-len",  type=int,   default=90)
    parser.add_argument("--workers",  type=int,   default=2)
    parser.add_argument("--config",   default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    db = cfg["database"]
    dsn = (
        f"postgresql://{db['user']}:{db['password']}@"
        f"{db['host']}:{db['port']}/{db['name']}"
    )

    from training.phase2_trainer import Phase2Config, Phase2Trainer

    config = Phase2Config(
        dsn=dsn,
        epochs=args.epochs,
        d_model=args.d_model,
        batch_size=args.batch,
        learning_rate=args.lr,
        dropout=args.dropout,
        seq_len=args.seq_len,
        num_workers=args.workers,
        train_end=date(2023, 6, 30),
        val_end=date(2023, 12, 31),
        test_end=date(2024, 12, 31),
    )

    trainer = Phase2Trainer(config)
    trainer.train()
    print("\nPhase 2 training complete — checkpoint saved to models/checkpoints/")


if __name__ == "__main__":
    main()
