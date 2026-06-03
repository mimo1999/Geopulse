"""
Tests for models/forecaster.py — EscalationForecaster.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from models.forecaster import EscalationForecaster


B = 2    # batch size
T = 90   # sequence length
F = 7    # features
H = 4    # horizon steps


@pytest.fixture
def model():
    return EscalationForecaster(
        num_features=F, d_model=64, num_heads=4,
        num_layers=2, dropout=0.0, seq_len=T, horizon=H,
    )


# ---------------------------------------------------------------------------

def test_forward_output_keys(model):
    x = torch.randn(B, T, F)
    out = model(x)
    for key in ("instability", "war", "terrorism", "financial", "risk_score"):
        assert key in out, f"Missing key: {key}"


def test_forward_shapes(model):
    x   = torch.randn(B, T, F)
    out = model(x)
    for key in ("instability", "war", "terrorism", "financial", "risk_score"):
        assert out[key].shape == (B, H), f"{key}: expected ({B},{H}), got {out[key].shape}"


def test_outputs_in_range(model):
    model.eval()
    x   = torch.randn(B, T, F)
    out = model(x)
    for key in ("instability", "war", "terrorism", "financial", "risk_score"):
        assert out[key].min().item() >= 0.0, f"{key} below 0"
        assert out[key].max().item() <= 1.0, f"{key} above 1"


def test_with_mask(model):
    x    = torch.randn(B, T, F)
    mask = torch.ones(B, T)
    mask[:, -10:] = 0.0    # last 10 steps masked
    out  = model(x, mask)
    assert out["risk_score"].shape == (B, H)


def test_mc_dropout_confidence(model):
    model.eval()
    x   = torch.randn(B, T, F)
    out = model.predict_with_confidence(x, n_passes=5)
    for key in ("risk_score", "confidence", "variance", "lower_bound", "upper_bound"):
        assert key in out
        assert out[key].shape == (B, H)


def test_confidence_in_range(model):
    x   = torch.randn(B, T, F)
    out = model.predict_with_confidence(x, n_passes=5)
    assert out["confidence"].min().item() >= 0.0
    assert out["confidence"].max().item() <= 1.0


def test_lower_upper_bounds(model):
    x   = torch.randn(B, T, F)
    out = model.predict_with_confidence(x, n_passes=5)
    # lower ≤ mean ≤ upper
    assert (out["lower_bound"] <= out["risk_score"] + 1e-4).all()
    assert (out["upper_bound"] >= out["risk_score"] - 1e-4).all()


def test_risk_score_consistency(model):
    """risk_score should equal weighted sum of individual heads."""
    x   = torch.randn(B, T, F)
    out = model(x)
    expected = (
        0.40 * out["instability"]
        + 0.30 * out["war"]
        + 0.20 * out["terrorism"]
        + 0.10 * out["financial"]
    ).clamp(0.0, 1.0)
    assert torch.allclose(out["risk_score"], expected, atol=1e-5)


def test_save_load(model):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "forecaster.pt")
        model.save(path)
        loaded = EscalationForecaster.load(path)

    x     = torch.randn(B, T, F)
    model.eval()
    loaded.eval()
    out1  = model(x)
    out2  = loaded(x)
    assert torch.allclose(out1["risk_score"], out2["risk_score"], atol=1e-5)


def test_encoder_weight_load_runs(model):
    """load_encoder_weights should run without error even with incompatible checkpoint."""
    import tempfile
    # Create a minimal HybridRiskTransformer checkpoint
    dummy_state = {
        "input_proj.0.weight": torch.randn(64, 7),
        "input_proj.0.bias":   torch.zeros(64),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "dummy.pt")
        torch.save({"state_dict": dummy_state}, path)
        loaded = model.load_encoder_weights(path)
    # At minimum input_proj.0.weight should be loaded (if dims match)
    # d_model=64, num_features=7 → input_proj[0] is Linear(7, 64) → weight (64,7) matches
    assert loaded >= 2


def test_horizon_steps_configurable():
    for H_val in [1, 2, 6]:
        m   = EscalationForecaster(d_model=32, num_heads=4, horizon=H_val)
        x   = torch.randn(1, 90, 7)
        out = m(x)
        assert out["risk_score"].shape == (1, H_val)
