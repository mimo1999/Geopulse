"""
Phase 3: Train EscalationForecaster on the bi-weekly parquet cache.

Self-contained — no database required.
Uses data/real_cache/*.parquet files produced by train_real_data.py.

Usage:
    python scripts/train_forecaster.py [options]

Options:
    --cache-dir          Path to parquet cache directory (default: data/real_cache)
    --epochs             Training epochs (default: 40)
    --batch              Batch size (default: 32)
    --lr                 Learning rate (default: 2e-4)
    --d-model            Transformer hidden dim (default: 128)
    --num-heads          Attention heads (default: 4)
    --num-layers         Transformer layers (default: 2)
    --horizon            Forecast horizon steps (default: 4)
    --context-steps      Context window in bi-weekly steps (default: 6)
    --encoder-weights    Path to HybridRiskTransformer checkpoint for warm-start
    --checkpoint-dir     Output directory for saved models (default: models/checkpoints)
    --run-name           Checkpoint filename prefix (default: forecaster_v1)
    --device             torch device (default: cpu)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.forecaster import EscalationForecaster
from models.forecaster_dataset import (
    DatasetConfig,
    ParquetForecastDataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_forecaster")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ForecasterTrainingConfig:
    cache_dir:        str   = "data/real_cache"
    epochs:           int   = 40
    batch_size:       int   = 32
    lr:               float = 2e-4
    d_model:          int   = 128
    num_heads:        int   = 4
    num_layers:       int   = 2
    horizon:          int   = 4
    context_steps:    int   = 6
    encoder_weights:  str   = ""
    checkpoint_dir:   str   = "models/checkpoints"
    run_name:         str   = "forecaster_v1"
    device:           str   = "cpu"
    # Horizon-weighted loss: closer steps are more important
    horizon_weights: list   = None  # set in __post_init__

    def __post_init__(self):
        if self.horizon_weights is None:
            # Step 1=1.0, step 2=0.8, step 3=0.6, step 4=0.4
            self.horizon_weights = [max(0.4, 1.0 - 0.2 * h) for h in range(self.horizon)]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class ForecastLoss(nn.Module):
    """
    Horizon-weighted MSE + optional smoothed BCE.

    Targets are (B, H, 5): [instability, war, terrorism, financial, risk_score]
    Predictions dict keys: instability (B,H), war, terrorism, financial, risk_score
    """

    def __init__(self, horizon_weights: list[float]):
        super().__init__()
        self.register_buffer(
            "hw",
            torch.tensor(horizon_weights, dtype=torch.float32),
        )

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        labels: torch.Tensor,            # (B, H, 5)
    ) -> torch.Tensor:
        task_keys = ["instability", "war", "terrorism", "financial", "risk_score"]
        total = torch.tensor(0.0, device=labels.device)

        for ti, key in enumerate(task_keys):
            pred  = preds[key]              # (B, H)
            tgt   = labels[:, :, ti]        # (B, H)
            mse   = (pred - tgt) ** 2       # (B, H)
            # Apply horizon weights: (H,) broadcast
            hw    = self.hw.to(labels.device)
            loss  = (mse * hw.unsqueeze(0)).mean()
            total = total + loss

        return total / len(task_keys)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ForecasterTrainer:
    def __init__(self, cfg: ForecasterTrainingConfig):
        self.cfg    = cfg
        self.device = torch.device(cfg.device)

        # Build datasets
        logger.info("Building datasets from %s", cfg.cache_dir)

        train_cfg = DatasetConfig(
            cache_dir=cfg.cache_dir,
            context_steps=cfg.context_steps,
            horizon=cfg.horizon,
            split="train",
        )
        val_cfg = DatasetConfig(
            cache_dir=cfg.cache_dir,
            context_steps=cfg.context_steps,
            horizon=cfg.horizon,
            split="val",
        )
        train_ds = ParquetForecastDataset(train_cfg)
        val_ds   = ParquetForecastDataset(val_cfg)

        if len(train_ds) == 0:
            raise RuntimeError(
                f"No training samples found. "
                f"Check that {cfg.cache_dir} contains *_features.parquet files."
            )

        logger.info("Train: %d samples | Val: %d samples", len(train_ds), len(val_ds))

        self.train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=ParquetForecastDataset.collate_fn,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=ParquetForecastDataset.collate_fn,
        )

        # Build model
        self.model = EscalationForecaster(
            num_features=7,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            horizon=cfg.horizon,
        ).to(self.device)

        # Warm-start encoder from existing model if provided
        if cfg.encoder_weights and Path(cfg.encoder_weights).exists():
            loaded = self.model.load_encoder_weights(cfg.encoder_weights)
            logger.info("Warm-start: loaded %d encoder keys from %s", loaded, cfg.encoder_weights)

        logger.info("Model parameters: %d", self.model.parameter_count())

        # Optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        total_steps    = cfg.epochs * max(1, len(self.train_loader))
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=cfg.lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        self.criterion = ForecastLoss(cfg.horizon_weights)

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        self.best_val_loss = float("inf")
        self.best_path = str(Path(cfg.checkpoint_dir) / f"{cfg.run_name}_best.pt")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        logger.info("Starting training: %d epochs on %s", self.cfg.epochs, self.device)

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss, val_metrics = self._eval_epoch()

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.model.save(self.best_path)

            logger.info(
                "Epoch %3d/%d | train=%.4f | val=%.4f %s",
                epoch, self.cfg.epochs,
                train_loss, val_loss,
                "← BEST" if is_best else "",
            )
            if epoch % 5 == 0:
                metric_str = " | ".join(
                    f"{k}={v:.4f}" for k, v in val_metrics.items()
                )
                logger.info("  Val per-task: %s", metric_str)

        # Save final checkpoint
        final_path = str(Path(self.cfg.checkpoint_dir) / f"{self.cfg.run_name}_final.pt")
        self.model.save(final_path)
        logger.info(
            "\nTraining complete.\n"
            "  Best val loss : %.4f\n"
            "  Best model    : %s\n"
            "  Final model   : %s",
            self.best_val_loss, self.best_path, final_path,
        )

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in self.train_loader:
            context = batch["context"].to(self.device)    # (B, context_steps, 7)
            labels  = batch["labels"].to(self.device)     # (B, horizon, 5)

            self.optimizer.zero_grad()
            preds = self.model(context)
            loss  = self.criterion(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _eval_epoch(self) -> tuple[float, dict[str, float]]:
        self.model.eval()
        total_loss  = 0.0
        n_batches   = 0
        task_losses = {k: 0.0 for k in ["instability", "war", "terrorism", "financial", "risk_score"]}

        for batch in self.val_loader:
            context = batch["context"].to(self.device)
            labels  = batch["labels"].to(self.device)

            preds = self.model(context)
            loss  = self.criterion(preds, labels)
            total_loss += loss.item()
            n_batches  += 1

            # Per-task MSE
            task_names = ["instability", "war", "terrorism", "financial", "risk_score"]
            for ti, key in enumerate(task_names):
                mse = float(((preds[key] - labels[:, :, ti]) ** 2).mean().item())
                task_losses[key] += mse

        n = max(n_batches, 1)
        return total_loss / n, {k: v / n for k, v in task_losses.items()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> ForecasterTrainingConfig:
    p = argparse.ArgumentParser(description="Train EscalationForecaster on parquet cache")
    p.add_argument("--cache-dir",        default="data/real_cache")
    p.add_argument("--epochs",           type=int,   default=40)
    p.add_argument("--batch",            type=int,   default=32,   dest="batch_size")
    p.add_argument("--lr",               type=float, default=2e-4)
    p.add_argument("--d-model",          type=int,   default=128,  dest="d_model")
    p.add_argument("--num-heads",        type=int,   default=4,    dest="num_heads")
    p.add_argument("--num-layers",       type=int,   default=2,    dest="num_layers")
    p.add_argument("--horizon",          type=int,   default=4)
    p.add_argument("--context-steps",    type=int,   default=6,    dest="context_steps")
    p.add_argument("--encoder-weights",  default="",              dest="encoder_weights")
    p.add_argument("--checkpoint-dir",   default="models/checkpoints", dest="checkpoint_dir")
    p.add_argument("--run-name",         default="forecaster_v1", dest="run_name")
    p.add_argument("--device",           default="cpu")
    args = p.parse_args()
    return ForecasterTrainingConfig(**vars(args))


if __name__ == "__main__":
    cfg     = parse_args()
    trainer = ForecasterTrainer(cfg)
    trainer.train()
