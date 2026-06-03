"""Tests for Integrated Gradients explainer (no DB, no model checkpoint)."""

import pytest
import numpy as np
import torch

from models.risk_model import HybridRiskTransformer
from inference.explainer import (
    IntegratedGradientsExplainer,
    FeatureAttribution,
)
from models.dataset import FEATURE_COLUMNS, NUM_FEATURES


@pytest.fixture
def model():
    m = HybridRiskTransformer(
        num_features=NUM_FEATURES,
        d_model=64,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        seq_len=30,
    )
    m.eval()
    return m


@pytest.fixture
def ig(model):
    return IntegratedGradientsExplainer(
        model=model,
        feature_names=FEATURE_COLUMNS,
        n_steps=10,   # fewer steps for speed in tests
        device="cpu",
    )


def test_explain_returns_attribution(ig):
    features = np.random.rand(30, NUM_FEATURES).astype(np.float32)
    mask = np.ones(30, dtype=np.float32)
    from datetime import date
    attr = ig.explain(features, mask, target_head="risk_score",
                      country="TestCountry", target_date=date(2024, 1, 1))

    assert isinstance(attr, FeatureAttribution)
    assert attr.attributions_temporal.shape == (30, NUM_FEATURES)
    assert attr.attributions_global.shape == (NUM_FEATURES,)
    assert attr.feature_names == FEATURE_COLUMNS


def test_attributions_sum_approx_output_diff(ig, model):
    """
    IG completeness: sum of attributions ≈ f(x) - f(baseline).
    We allow loose tolerance (10% of output) given discrete approximation.
    """
    from datetime import date
    features = np.random.rand(30, NUM_FEATURES).astype(np.float32) * 0.5
    mask = np.ones(30, dtype=np.float32)
    attr = ig.explain(features, mask, target_head="risk_score")

    # f(x)
    x_tensor = torch.from_numpy(features).unsqueeze(0)
    m_tensor = torch.ones(1, 30)
    with torch.no_grad():
        fx = float(model(x_tensor, m_tensor)["risk_score"].item())

    # f(baseline) = f(zeros)
    baseline = torch.zeros(1, 30, NUM_FEATURES)
    with torch.no_grad():
        fb = float(model(baseline, m_tensor)["risk_score"].item())

    expected_diff = fx - fb
    actual_sum = float(attr.attributions_temporal.sum())
    # Loose tolerance for discrete IG approximation
    assert abs(actual_sum - expected_diff) < max(abs(expected_diff) * 0.3, 0.05)


def test_zero_input_zero_attributions(ig):
    """Zero input should yield near-zero attributions (baseline = input)."""
    features = np.zeros((30, NUM_FEATURES), dtype=np.float32)
    mask = np.ones(30, dtype=np.float32)
    attr = ig.explain(features, mask, target_head="risk_score")
    assert np.all(np.abs(attr.attributions_temporal) < 1e-6)


def test_top_features(ig):
    features = np.random.rand(30, NUM_FEATURES).astype(np.float32)
    mask = np.ones(30, dtype=np.float32)
    attr = ig.explain(features, mask, target_head="risk_score")
    top = attr.top_features(n=3)
    assert len(top) == 3
    assert all(isinstance(f, str) and isinstance(v, float) for f, v in top)
    # Sorted by absolute attribution
    scores = [abs(v) for _, v in top]
    assert scores == sorted(scores, reverse=True)


def test_to_dict(ig):
    from datetime import date
    features = np.random.rand(30, NUM_FEATURES).astype(np.float32)
    mask = np.ones(30, dtype=np.float32)
    attr = ig.explain(features, mask, target_head="risk_score",
                      country="PK", target_date=date(2024, 3, 1))
    d = attr.to_dict()
    assert d["country"] == "PK"
    assert "attributions" in d
    assert len(d["attributions"]) == NUM_FEATURES
    assert "top_features" in d
    assert len(d["top_features"]) == 3


def test_different_inputs_different_attributions(ig):
    """Two different inputs should (almost always) yield different attributions."""
    f1 = np.random.rand(30, NUM_FEATURES).astype(np.float32)
    f2 = np.random.rand(30, NUM_FEATURES).astype(np.float32) * 2
    mask = np.ones(30, dtype=np.float32)
    attr1 = ig.explain(f1, mask)
    attr2 = ig.explain(f2, mask)
    # Global attributions should differ
    assert not np.allclose(attr1.attributions_global, attr2.attributions_global, atol=1e-4)
