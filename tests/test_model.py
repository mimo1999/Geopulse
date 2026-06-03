"""Tests for HybridRiskTransformer model."""

import pytest
import torch
from models.risk_model import HybridRiskTransformer, PositionalEncoding, AttentionPooling


@pytest.fixture
def model():
    return HybridRiskTransformer(
        num_features=7,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        seq_len=90,
    )


def test_model_forward_shape(model):
    B, T, F = 4, 90, 7
    features = torch.randn(B, T, F)
    mask = torch.ones(B, T)
    out = model(features, mask)
    for key in ("instability", "war", "terrorism", "financial", "risk_score"):
        assert key in out
        assert out[key].shape == (B,), f"{key} has wrong shape"


def test_risk_score_range(model):
    B, T, F = 8, 90, 7
    features = torch.randn(B, T, F) * 10   # large values
    out = model(features)
    assert (out["risk_score"] >= 0.0).all()
    assert (out["risk_score"] <= 1.0).all()


def test_model_with_mask(model):
    """Masked timesteps should not affect output shape."""
    B, T, F = 2, 90, 7
    features = torch.randn(B, T, F)
    mask = torch.zeros(B, T)
    mask[:, -30:] = 1.0   # only last 30 days valid
    out = model(features, mask)
    assert out["risk_score"].shape == (B,)


def test_risk_weights_sum(model):
    """Risk score should equal weighted sum of components."""
    B, T, F = 1, 90, 7
    features = torch.zeros(B, T, F)
    with torch.no_grad():
        out = model(features)
    expected = (
        0.40 * out["instability"]
        + 0.30 * out["war"]
        + 0.20 * out["terrorism"]
        + 0.10 * out["financial"]
    )
    assert torch.allclose(out["risk_score"], expected.clamp(0, 1), atol=1e-5)


def test_mc_dropout_confidence(model):
    B, T, F = 2, 90, 7
    features = torch.randn(B, T, F)
    out = model.predict_with_confidence(features, n_passes=10)
    assert "confidence" in out
    assert "variance" in out
    assert (out["confidence"] >= 0.0).all()
    assert (out["confidence"] <= 1.0).all()


def test_parameter_count(model):
    n = model.parameter_count()
    assert n > 0
    # With d_model=64 it should be in a reasonable range
    assert 10_000 < n < 5_000_000


def test_save_load(model, tmp_path):
    path = str(tmp_path / "test_model.pt")
    model.save(path)
    loaded = HybridRiskTransformer.load(path)
    assert loaded.parameter_count() == model.parameter_count()

    # Outputs should match
    features = torch.randn(2, 90, 7)
    with torch.no_grad():
        out1 = model(features)
        out2 = loaded(features)
    assert torch.allclose(out1["risk_score"], out2["risk_score"], atol=1e-5)


def test_positional_encoding():
    pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
    x = torch.zeros(2, 50, 64)
    out = pe(x)
    assert out.shape == (2, 50, 64)
    # PE should add non-zero values
    assert not torch.allclose(out, x)


def test_attention_pooling():
    pool = AttentionPooling(d_model=64)
    x = torch.randn(4, 90, 64)
    out = pool(x)
    assert out.shape == (4, 64)


def test_attention_pooling_with_mask():
    pool = AttentionPooling(d_model=64)
    x = torch.randn(4, 90, 64)
    mask = torch.zeros(4, 90)
    mask[:, -10:] = 1.0
    out = pool(x, mask)
    assert out.shape == (4, 64)
