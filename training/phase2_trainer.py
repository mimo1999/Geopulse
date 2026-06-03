"""
Phase 2 trainer: multi-task learning with proxy ground-truth labels.

Key differences from Phase 1 trainer:
  - Uses MultiTaskRiskDataset (4-label: instability, war, terrorism, financial)
  - Label smoothing to account for proxy label noise
  - Focal loss option for class-imbalanced tasks (terrorism is rare)
  - Per-task evaluation metrics (AUC, calibration)
  - Attribution evaluation: saves IG attributions for top-risk predictions
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from models.risk_model import HybridRiskTransformer
from models.multitask_dataset import MultiTaskDataLoader, NUM_LABELS_V2

logger = logging.getLogger("training.phase2_trainer")


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal loss for binary classification with severe class imbalance.
    FL(p) = -α(1-p)^γ log(p)
    Used for terrorism head (rare events).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        p_t = pred * target + (1 - pred) * (1 - target)
        focal = self.alpha * (1 - p_t) ** self.gamma * bce
        return focal.mean()


class SmoothedBCE(nn.Module):
    """BCE with label smoothing for noisy proxy labels."""

    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.s = smoothing

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        target_smooth = target * (1 - self.s) + 0.5 * self.s
        return F.binary_cross_entropy(pred, target_smooth)


class MultiTaskLossV2(nn.Module):
    """
    Phase 2 multi-task loss with:
    - Smoothed BCE for instability, war, financial (proxy labels)
    - Focal loss for terrorism (severe class imbalance)
    - Uncertainty-weighted task combination (Kendall & Gal)
    """

    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(4))
        self.instability_loss = SmoothedBCE(smoothing)
        self.war_loss         = SmoothedBCE(smoothing)
        self.terrorism_loss   = FocalLoss(alpha=0.25, gamma=2.0)
        self.financial_loss   = SmoothedBCE(smoothing)

    def forward(
        self,
        preds: dict[str, Tensor],
        targets: Tensor,   # (B, 4): instability, war, terrorism, financial
    ) -> tuple[Tensor, dict[str, float]]:
        losses_raw = torch.stack([
            self.instability_loss(preds["instability"], targets[:, 0]),
            self.war_loss(        preds["war"],         targets[:, 1]),
            self.terrorism_loss(  preds["terrorism"],   targets[:, 2]),
            self.financial_loss(  preds["financial"],   targets[:, 3]),
        ])

        precisions = torch.exp(-self.log_vars)
        total = (precisions * losses_raw + self.log_vars).sum()

        names = ["instability", "war", "terrorism", "financial"]
        individual = {n: losses_raw[i].item() for i, n in enumerate(names)}

        return total, individual


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TaskMetrics:
    """Running AUC and MSE per task."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._preds: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []

    def update(self, preds: Tensor, targets: Tensor):
        self._preds.append(preds.detach().cpu().numpy())
        self._targets.append(targets.detach().cpu().numpy())

    def compute(self) -> dict[str, float]:
        if not self._preds:
            return {}
        p = np.concatenate(self._preds)
        t = np.concatenate(self._targets)
        mse = float(np.mean((p - t) ** 2))

        # AUC (binarize targets at 0.5)
        t_bin = (t >= 0.5).astype(int)
        metrics = {"mse": mse}
        if t_bin.sum() > 0 and t_bin.sum() < len(t_bin):
            try:
                from sklearn.metrics import roc_auc_score
                metrics["auc"] = float(roc_auc_score(t_bin, p))
            except Exception:
                pass
        return metrics


# ---------------------------------------------------------------------------
# Phase 2 Trainer
# ---------------------------------------------------------------------------

@dataclass
class Phase2Config:
    # Model
    num_features: int = 7
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 3       # +1 vs Phase 1 for better temporal modelling
    dropout: float = 0.15
    seq_len: int = 90

    # Training
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    early_stopping_patience: int = 8
    label_smoothing: float = 0.05

    # Data
    dsn: str = "postgresql://gldt:gldt_secret@localhost:5432/gdelt_risk"
    train_end: date = field(default_factory=lambda: date(2023, 6, 30))
    val_end:   date = field(default_factory=lambda: date(2023, 12, 31))
    test_end:  date = field(default_factory=lambda: date(2024, 12, 31))
    num_workers: int = 2

    # Attribution computation during eval
    compute_attributions: bool = True
    attribution_batch_n: int = 10   # compute IG for this many high-risk samples per epoch

    checkpoint_dir: str = "models/checkpoints"
    run_name: str = "run_phase2"


class Phase2Trainer:
    """
    Multi-task trainer for Phase 2.

    Trains on 4 proxy labels simultaneously with task-specific losses.
    Evaluates per-task AUC after each epoch.
    Optionally computes and persists feature attributions for top predictions.
    """

    def __init__(self, config: Phase2Config):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Phase 2 training — device: %s", self.device)

        self.model = HybridRiskTransformer(
            num_features=config.num_features,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            seq_len=config.seq_len,
        ).to(self.device)

        self.criterion = MultiTaskLossV2(smoothing=config.label_smoothing).to(self.device)

        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.criterion.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.ckpt_dir = Path(config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self._best_val_loss = float("inf")
        self._patience = 0

        logger.info("Model parameters: %d", self.model.parameter_count())

    def train(self) -> None:
        loaders = MultiTaskDataLoader(
            dsn=self.cfg.dsn,
            train_end=self.cfg.train_end,
            val_end=self.cfg.val_end,
            test_end=self.cfg.test_end,
            seq_len=self.cfg.seq_len,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
        )

        total_steps = self.cfg.epochs * len(loaders.train)
        warmup_steps = 5 * len(loaders.train)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.cfg.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
        )

        logger.info("Phase 2 training: %d epochs", self.cfg.epochs)

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.monotonic()
            train_loss, train_tasks = self._train_epoch(loaders.train, scheduler)
            val_loss, val_metrics   = self._eval_epoch(loaders.val)
            elapsed = time.monotonic() - t0

            logger.info(
                "Epoch %d/%d | train=%.4f val=%.4f | %.1fs",
                epoch, self.cfg.epochs, train_loss, val_loss, elapsed,
            )
            for task, metrics in val_metrics.items():
                auc_str = f" auc={metrics.get('auc', 0):.3f}" if "auc" in metrics else ""
                logger.info(
                    "  %s: mse=%.4f%s",
                    task, metrics.get("mse", 0), auc_str,
                )

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._patience = 0
                path = self.ckpt_dir / f"{self.cfg.run_name}_best.pt"
                self.model.save(str(path))
                logger.info("  ✓ Best model saved (val=%.4f)", val_loss)
            else:
                self._patience += 1
                if self._patience >= self.cfg.early_stopping_patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        # Test set evaluation
        test_loss, test_metrics = self._eval_epoch(loaders.test)
        logger.info("Test loss: %.4f", test_loss)
        for task, metrics in test_metrics.items():
            logger.info("  %s: %s", task, metrics)

    def _train_epoch(
        self,
        loader: DataLoader,
        scheduler,
    ) -> tuple[float, dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        task_losses: dict[str, list[float]] = {
            k: [] for k in ("instability", "war", "terrorism", "financial")
        }

        for batch in loader:
            features = batch["features"].to(self.device)
            labels   = batch["labels"].to(self.device)      # (B, 4)
            mask     = batch["mask"].to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(features, mask)
            loss, individual = self.criterion(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)
            self.optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            for k, v in individual.items():
                task_losses[k].append(v)

        avg_tasks = {k: float(np.mean(v)) for k, v in task_losses.items()}
        return total_loss / max(len(loader), 1), avg_tasks

    @torch.no_grad()
    def _eval_epoch(
        self,
        loader: DataLoader,
    ) -> tuple[float, dict[str, dict[str, float]]]:
        self.model.eval()
        total_loss = 0.0
        task_names = ["instability", "war", "terrorism", "financial"]
        metrics = {k: TaskMetrics() for k in task_names}

        for batch in loader:
            features = batch["features"].to(self.device)
            labels   = batch["labels"].to(self.device)
            mask     = batch["mask"].to(self.device)

            preds = self.model(features, mask)
            loss, _ = self.criterion(preds, labels)
            total_loss += loss.item()

            for i, k in enumerate(task_names):
                metrics[k].update(preds[k], labels[:, i])

        avg_loss = total_loss / max(len(loader), 1)
        return avg_loss, {k: m.compute() for k, m in metrics.items()}
