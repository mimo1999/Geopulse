"""
PyTorch training loop for HybridRiskTransformer.

Phase 1: binary instability classifier (stable vs unstable).
Phase 2: multi-task prediction (war, terrorism, financial stress).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.risk_model import HybridRiskTransformer
from models.dataset import RiskDataLoader

logger = logging.getLogger("training.trainer")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    # Model
    num_features: int = 7
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    seq_len: int = 90

    # Training
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    early_stopping_patience: int = 7

    # Scheduler
    warmup_epochs: int = 5
    scheduler: str = "cosine"         # "cosine" | "step" | "none"

    # Data
    dsn: str = "postgresql://gldt:gldt_secret@localhost:5432/gdelt_risk"
    train_end: date = field(default_factory=lambda: date(2023, 6, 30))
    val_end:   date = field(default_factory=lambda: date(2023, 12, 31))
    test_end:  date = field(default_factory=lambda: date(2024, 12, 31))
    num_workers: int = 2

    # Output
    checkpoint_dir: str = "models/checkpoints"
    run_name: str = "run_phase1"


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    """
    Combines BCELoss across all output heads with learnable weights.
    Uses uncertainty weighting (Kendall & Gal, 2018):
        L_total = Σ exp(-log_var_i) * L_i + log_var_i
    """

    def __init__(self, n_tasks: int = 4):
        super().__init__()
        # Learnable log-variance per task (initialized to 0)
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
        self.bce = nn.BCELoss(reduction="mean")

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        targets: torch.Tensor,  # (B, n_tasks)
    ) -> tuple[torch.Tensor, dict[str, float]]:
        task_keys = ["instability", "war", "terrorism", "financial"]
        total = torch.tensor(0.0, device=targets.device)
        losses: dict[str, float] = {}

        for i, key in enumerate(task_keys):
            pred = preds[key]                         # (B,)
            tgt  = targets[:, i].float()
            task_loss = self.bce(pred, tgt)

            # Uncertainty weighting
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * task_loss + self.log_vars[i]
            losses[key] = task_loss.item()

        return total, losses


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Manages the full training loop including:
    - LR warmup + cosine schedule
    - Gradient clipping
    - Early stopping
    - Checkpoint saving/loading
    - Basic metrics logging
    """

    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training device: %s", self.device)

        self.model = HybridRiskTransformer(
            num_features=config.num_features,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            seq_len=config.seq_len,
        ).to(self.device)

        logger.info("Model parameters: %d", self.model.parameter_count())

        self.criterion = MultiTaskLoss(n_tasks=4).to(self.device)

        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.criterion.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._best_val_loss = float("inf")
        self._patience_counter = 0

    def train(self) -> None:
        """Run full training loop."""
        loaders = RiskDataLoader(
            dsn=self.cfg.dsn,
            train_end=self.cfg.train_end,
            val_end=self.cfg.val_end,
            test_end=self.cfg.test_end,
            seq_len=self.cfg.seq_len,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
        )

        scheduler = self._build_scheduler(len(loaders.train))

        logger.info("Starting training for %d epochs", self.cfg.epochs)
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.monotonic()

            train_loss = self._train_epoch(loaders.train, scheduler, epoch)
            val_loss   = self._val_epoch(loaders.val)

            elapsed = time.monotonic() - t0
            logger.info(
                "Epoch %d/%d — train=%.4f  val=%.4f  (%.1fs)",
                epoch, self.cfg.epochs, train_loss, val_loss, elapsed,
            )

            improved = self._checkpoint(val_loss, epoch)
            if not improved:
                self._patience_counter += 1
                if self._patience_counter >= self.cfg.early_stopping_patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break
            else:
                self._patience_counter = 0

        # Final evaluation on test set
        best_path = self.checkpoint_dir / f"{self.cfg.run_name}_best.pt"
        if best_path.exists():
            self.model = HybridRiskTransformer.load(str(best_path), str(self.device))
        test_loss = self._val_epoch(loaders.test)
        logger.info("Test loss: %.4f", test_loss)

    def _train_epoch(
        self,
        loader: DataLoader,
        scheduler,
        epoch: int,
    ) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            features = batch["features"].to(self.device)
            labels   = batch["labels"].to(self.device)    # (B, 1) — Phase 1
            mask     = batch["mask"].to(self.device)

            # Expand labels to 4 tasks: use risk_score for all in Phase 1
            # Phase 2: supply ground-truth multi-task labels
            targets = labels.expand(-1, 4)                # (B, 4)

            self.optimizer.zero_grad()
            preds = self.model(features, mask)
            loss, _ = self.criterion(preds, targets)
            loss.backward()

            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.gradient_clip,
            )
            self.optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        for batch in loader:
            features = batch["features"].to(self.device)
            labels   = batch["labels"].to(self.device)
            mask     = batch["mask"].to(self.device)
            targets  = labels.expand(-1, 4)

            preds = self.model(features, mask)
            loss, _ = self.criterion(preds, targets)
            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    def _checkpoint(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            path = self.checkpoint_dir / f"{self.cfg.run_name}_best.pt"
            self.model.save(str(path))
            logger.info("  ✓ New best model saved (val=%.4f)", val_loss)
            return True

        # Save periodic checkpoint
        if epoch % 10 == 0:
            path = self.checkpoint_dir / f"{self.cfg.run_name}_epoch{epoch}.pt"
            self.model.save(str(path))

        return False

    def _build_scheduler(self, steps_per_epoch: int):
        if self.cfg.scheduler == "cosine":
            total_steps = self.cfg.epochs * steps_per_epoch
            warmup_steps = self.cfg.warmup_epochs * steps_per_epoch
            from torch.optim.lr_scheduler import OneCycleLR
            return OneCycleLR(
                self.optimizer,
                max_lr=self.cfg.learning_rate,
                total_steps=total_steps,
                pct_start=warmup_steps / total_steps,
            )
        return None
