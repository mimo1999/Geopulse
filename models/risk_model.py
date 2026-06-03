"""
Hybrid Temporal Transformer for geopolitical risk forecasting.

Architecture:
    Input features (B, T, F)
           ↓
    Linear projection to d_model
           ↓
    Positional encoding
           ↓
    Transformer encoder (multi-head self-attention × N layers)
           ↓
    Attention pooling over time
           ↓
    Multi-task output heads

Output heads (Phase 1: instability + composite risk):
    - risk_score     (0–1)   composite weighted score
    - instability    (0–1)
    - war_prob       (0–1)
    - terror_risk    (0–1)
    - financial_str  (0–1)
    - confidence     (0–1)   via MC Dropout variance

Confidence estimation: Monte Carlo Dropout
    Run N forward passes with dropout active at inference time.
    Confidence = 1 - Var(predictions across passes)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 365, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Attention Pooling
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """
    Weighted sum over time using a learned attention score.
    Supports an external mask to ignore padded timesteps.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:    (B, T, d_model)
            mask: (B, T) float, 1=valid 0=ignore
        Returns:
            (B, d_model)
        """
        scores = self.attn(x).squeeze(-1)         # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)       # (B, T)
        return (weights.unsqueeze(-1) * x).sum(dim=1)  # (B, d_model)


# ---------------------------------------------------------------------------
# Multi-task Output Head
# ---------------------------------------------------------------------------

class RiskHead(nn.Module):
    """Single output head: linear → sigmoid → [0, 1]."""

    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)           # (B,)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class HybridRiskTransformer(nn.Module):
    """
    Hybrid Temporal Transformer for multi-task geopolitical risk scoring.

    Args:
        num_features:  Number of input feature dimensions (default 7).
        d_model:       Transformer hidden dimension (default 128).
        num_heads:     Attention heads (default 4).
        num_layers:    Transformer encoder layers (default 2).
        dropout:       Dropout probability (default 0.1).
        seq_len:       Maximum sequence length (default 90).

    Risk weight constants follow the specification:
        Risk = 0.4·I + 0.3·W + 0.2·T + 0.1·F
    """

    RISK_WEIGHTS = {
        "instability": 0.40,
        "war":         0.30,
        "terrorism":   0.20,
        "financial":   0.10,
    }

    def __init__(
        self,
        num_features: int = 7,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        seq_len: int = 90,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,          # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.pool = AttentionPooling(d_model)

        # Multi-task heads
        self.head_instability = RiskHead(d_model)
        self.head_war         = RiskHead(d_model)
        self.head_terrorism   = RiskHead(d_model)
        self.head_financial   = RiskHead(d_model)

        # MC Dropout for confidence estimation
        self.mc_dropout = nn.Dropout(p=0.15)

        self._d_model = d_model
        self._init_weights()
        # Default to eval so deterministic inference works out of the box;
        # call .train() explicitly when training.
        self.eval()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        features: Tensor,
        mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            features: (B, T, num_features)  float32
            mask:     (B, T)                float32, 1=valid

        Returns:
            Dict with keys:
                instability, war, terrorism, financial, risk_score
        """
        # Project inputs
        x = self.input_proj(features)             # (B, T, d_model)
        x = self.pos_enc(x)

        # Build padding mask for transformer (True = ignore)
        src_key_padding_mask: Optional[Tensor] = None
        if mask is not None:
            src_key_padding_mask = (mask == 0)    # (B, T) bool

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.mc_dropout(x)                    # active during inference for MC

        pooled = self.pool(x, mask)               # (B, d_model)

        instability = self.head_instability(pooled)
        war         = self.head_war(pooled)
        terrorism   = self.head_terrorism(pooled)
        financial   = self.head_financial(pooled)

        risk_score = (
            self.RISK_WEIGHTS["instability"] * instability
            + self.RISK_WEIGHTS["war"]       * war
            + self.RISK_WEIGHTS["terrorism"] * terrorism
            + self.RISK_WEIGHTS["financial"] * financial
        ).clamp(0.0, 1.0)

        return {
            "instability": instability,
            "war":         war,
            "terrorism":   terrorism,
            "financial":   financial,
            "risk_score":  risk_score,
        }

    @torch.no_grad()
    def predict_with_confidence(
        self,
        features: Tensor,
        mask: Optional[Tensor] = None,
        n_passes: int = 50,
    ) -> dict[str, Tensor]:
        """
        Monte Carlo Dropout inference.

        Runs `n_passes` stochastic forward passes with dropout active,
        then returns mean predictions and confidence.

        Args:
            features:  (B, T, F)
            mask:      (B, T) optional
            n_passes:  MC samples (default 50)

        Returns:
            Dict with keys: instability, war, terrorism, financial,
                            risk_score, confidence, variance
        """
        # Enable dropout during inference
        self.train()           # activates Dropout layers
        torch.set_grad_enabled(False)

        all_risk: list[Tensor] = []
        all_outputs: list[dict[str, Tensor]] = []

        for _ in range(n_passes):
            out = self.forward(features, mask)
            all_risk.append(out["risk_score"])
            all_outputs.append(out)

        self.eval()
        torch.set_grad_enabled(True)

        # Stack → (n_passes, B)
        risk_stack = torch.stack(all_risk, dim=0)
        mean_risk  = risk_stack.mean(dim=0)
        var_risk   = risk_stack.var(dim=0)
        confidence = (1.0 - var_risk).clamp(0.0, 1.0)

        # Average each head
        result: dict[str, Tensor] = {}
        for key in ("instability", "war", "terrorism", "financial"):
            stacked = torch.stack([o[key] for o in all_outputs], dim=0)
            result[key] = stacked.mean(dim=0)

        result["risk_score"] = mean_risk
        result["confidence"] = confidence
        result["variance"]   = var_risk

        return result

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "num_features": self.input_proj[0].in_features,
                "d_model":      self._d_model,
            },
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "HybridRiskTransformer":
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
