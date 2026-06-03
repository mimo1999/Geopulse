"""
Phase 3: EscalationForecaster — multi-step ahead risk trajectory.

Architecture
────────────
    Input: (B, T, F)  — T-day history, F=7 features per day
           ↓
    Linear projection → d_model
           ↓
    Sinusoidal Positional Encoding
           ↓
    TransformerEncoder (Pre-LN, same spec as HybridRiskTransformer)
           ↓  memory (B, T, d_model)
    Horizon embeddings (H learnable query vectors)
           ↓
    TransformerDecoder (cross-attention into encoder memory)
           ↓  (B, H, d_model)
    Multi-task ForecastHead per horizon step
           ↓
    Output dict: instability/war/terrorism/financial/risk_score each (B, H)

Risk score formula (same weights as Phase 1/2 model):
    risk = 0.4·I + 0.3·W + 0.2·T + 0.1·F  clamped to [0, 1]

Confidence intervals:
    MC Dropout (n_passes forward passes) → mean ± std per horizon step.
    lower_bound = μ - 1.28σ  (≈ 10th pct)
    upper_bound = μ + 1.28σ  (≈ 90th pct)

Encoder weight transfer:
    load_encoder_weights(path)  copies matching keys from a
    HybridRiskTransformer checkpoint, enabling warm-start.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Positional Encoding (identical to risk_model.py — no import to keep
# models self-contained for checkpoint portability)
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 400, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Forecast Head
# ---------------------------------------------------------------------------

class ForecastHead(nn.Module):
    """Shared MLP head: (d_model,) → (5,) risk task scores, all in [0,1]."""

    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 5),   # [instability, war, terrorism, financial, risk_score]
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)   # (B, 5) or (N, 5)


# ---------------------------------------------------------------------------
# EscalationForecaster
# ---------------------------------------------------------------------------

@dataclass
class ForecasterConfig:
    num_features: int = 7
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    seq_len: int = 90
    horizon: int = 4          # bi-weekly steps ahead (H)
    mc_dropout_p: float = 0.15


class EscalationForecaster(nn.Module):
    """
    Transformer encoder-decoder for multi-step ahead risk forecasting.

    Args:
        num_features:  Input feature dimensions (default 7).
        d_model:       Hidden dimension (default 128).
        num_heads:     Attention heads (default 4).
        num_layers:    Encoder+decoder layers each (default 2).
        dropout:       Dropout probability (default 0.1).
        seq_len:       Max input sequence length (default 90).
        horizon:       Forecast horizon steps (default 4 bi-weekly periods).
    """

    RISK_WEIGHTS = {"instability": 0.40, "war": 0.30, "terrorism": 0.20, "financial": 0.10}
    TASK_IDX     = {"instability": 0, "war": 1, "terrorism": 2, "financial": 3, "risk_score": 4}

    def __init__(
        self,
        num_features: int = 7,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        seq_len: int = 90,
        horizon: int = 4,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self._cfg = ForecasterConfig(
            num_features=num_features,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            seq_len=seq_len,
            horizon=horizon,
        )

        # ----- Encoder -----
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = _PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # ----- Decoder -----
        # H learnable horizon query embeddings
        self.horizon_queries = nn.Parameter(torch.randn(1, horizon, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # ----- Output head (shared across all horizon steps) -----
        self.head = ForecastHead(d_model, hidden=64)

        # MC Dropout
        self.mc_dropout = nn.Dropout(p=0.15)

        self._horizon = horizon
        self._init_weights()
        self.eval()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.horizon_queries, std=0.02)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        features: Tensor,                    # (B, T, F)
        mask: Optional[Tensor] = None,       # (B, T) float 1=valid
    ) -> dict[str, Tensor]:
        """
        Args:
            features: (B, T, num_features)  float32
            mask:     (B, T)                float32  1=valid 0=pad

        Returns:
            Dict with keys instability, war, terrorism, financial, risk_score
            each of shape (B, H).
        """
        B = features.size(0)

        # Encode input sequence
        x = self.input_proj(features)            # (B, T, d_model)
        x = self.pos_enc(x)

        src_key_padding_mask: Optional[Tensor] = None
        if mask is not None:
            src_key_padding_mask = (mask == 0)   # (B, T) bool

        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, d_model)
        memory = self.mc_dropout(memory)

        # Decode with horizon queries
        queries = self.horizon_queries.expand(B, -1, -1)   # (B, H, d_model)
        decoded = self.decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=src_key_padding_mask,
        )   # (B, H, d_model)

        # Apply forecast head to each horizon step
        # Reshape to (B*H, d_model) → (B*H, 5) → reshape back
        BH = B * self._horizon
        flat = decoded.reshape(BH, -1)           # (B*H, d_model)
        preds = self.head(flat)                  # (B*H, 5)
        preds = preds.reshape(B, self._horizon, 5)  # (B, H, 5)

        instability = preds[:, :, 0]
        war         = preds[:, :, 1]
        terrorism   = preds[:, :, 2]
        financial   = preds[:, :, 3]

        risk_score = (
            self.RISK_WEIGHTS["instability"] * instability
            + self.RISK_WEIGHTS["war"]       * war
            + self.RISK_WEIGHTS["terrorism"] * terrorism
            + self.RISK_WEIGHTS["financial"] * financial
        ).clamp(0.0, 1.0)

        return {
            "instability": instability,    # (B, H)
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
        n_passes: int = 30,
    ) -> dict[str, Tensor]:
        """
        MC Dropout inference over H horizon steps.

        Returns:
            Dict with same keys as forward() plus:
                confidence  (B, H)   = 1 - Var(risk across passes)
                variance    (B, H)
                lower_bound (B, H)   ≈ 10th percentile
                upper_bound (B, H)   ≈ 90th percentile
        """
        self.train()
        torch.set_grad_enabled(False)

        all_risk: list[Tensor] = []
        all_out: list[dict[str, Tensor]] = []

        for _ in range(n_passes):
            out = self.forward(features, mask)
            all_risk.append(out["risk_score"])
            all_out.append(out)

        self.eval()
        torch.set_grad_enabled(True)

        risk_stack = torch.stack(all_risk, dim=0)    # (passes, B, H)
        mean_risk  = risk_stack.mean(dim=0)
        var_risk   = risk_stack.var(dim=0)
        std_risk   = risk_stack.std(dim=0)

        confidence   = (1.0 - var_risk).clamp(0.0, 1.0)
        lower_bound  = (mean_risk - 1.28 * std_risk).clamp(0.0, 1.0)
        upper_bound  = (mean_risk + 1.28 * std_risk).clamp(0.0, 1.0)

        result: dict[str, Tensor] = {}
        for key in ("instability", "war", "terrorism", "financial"):
            stacked = torch.stack([o[key] for o in all_out], dim=0)
            result[key] = stacked.mean(dim=0)

        result["risk_score"]  = mean_risk
        result["confidence"]  = confidence
        result["variance"]    = var_risk
        result["lower_bound"] = lower_bound
        result["upper_bound"] = upper_bound

        return result

    # ------------------------------------------------------------------
    # Encoder weight transfer from HybridRiskTransformer checkpoint
    # ------------------------------------------------------------------

    def load_encoder_weights(self, checkpoint_path: str, strict: bool = False) -> int:
        """
        Copy matching encoder weights from a HybridRiskTransformer checkpoint.
        Compatible keys: input_proj.*, pos_enc.*, transformer.* → encoder.*

        Args:
            checkpoint_path: Path to .pt file saved by HybridRiskTransformer.save()
            strict:          If True, raise on any missing key.

        Returns:
            Number of keys successfully loaded.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        src_state = ckpt.get("state_dict", ckpt)

        # Key mapping: HybridRiskTransformer → EscalationForecaster
        key_map = {
            "input_proj.0.weight":  "input_proj.0.weight",
            "input_proj.0.bias":    "input_proj.0.bias",
            "input_proj.1.weight":  "input_proj.1.weight",
            "input_proj.1.bias":    "input_proj.1.bias",
        }
        # Encoder layers: transformer.layers.* → encoder.layers.*
        for k, v in list(src_state.items()):
            if k.startswith("transformer."):
                new_k = k.replace("transformer.", "encoder.", 1)
                key_map[k] = new_k
            elif k.startswith("pos_enc."):
                key_map[k] = k

        own_state = self.state_dict()
        loaded = 0
        for src_k, dst_k in key_map.items():
            if src_k in src_state and dst_k in own_state:
                src_v = src_state[src_k]
                if own_state[dst_k].shape == src_v.shape:
                    own_state[dst_k].copy_(src_v)
                    loaded += 1

        self.load_state_dict(own_state, strict=False)
        return loaded

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "num_features": self._cfg.num_features,
                "d_model":      self._cfg.d_model,
                "num_heads":    self._cfg.num_heads,
                "num_layers":   self._cfg.num_layers,
                "dropout":      self._cfg.dropout,
                "seq_len":      self._cfg.seq_len,
                "horizon":      self._cfg.horizon,
            },
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "EscalationForecaster":
        ckpt = torch.load(path, map_location=device)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
