"""
Tests for models/gnn.py — RiskGNN and GraphAttentionLayer.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from models.gnn import GraphAttentionLayer, RiskGNN, build_adjacency, NODE_FEATURE_DIM

N = 20    # number of countries (nodes)
F = NODE_FEATURE_DIM   # 12 features


@pytest.fixture
def adj():
    """Random sparse adjacency matrix."""
    raw = torch.rand(N, N)
    mask = (raw > 0.7).float()
    raw  = raw * mask
    # Row-normalise
    rs = raw.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return raw / rs


@pytest.fixture
def x():
    return torch.rand(N, F)


@pytest.fixture
def gnn():
    return RiskGNN(node_features=F, hidden=32, out_features=8, num_layers=2)


# ---------------------------------------------------------------------------

def test_gat_layer_output_shape(adj, x):
    layer = GraphAttentionLayer(in_features=F, out_features=32)
    out   = layer(x, adj)
    assert out.shape == (N, 32)


def test_gat_layer_no_nan(adj, x):
    layer = GraphAttentionLayer(in_features=F, out_features=32)
    out   = layer(x, adj)
    assert not torch.isnan(out).any(), "GAT layer output contains NaN"


def test_gnn_forward_keys(gnn, x, adj):
    out = gnn(x, adj)
    for key in ("node_embeddings", "contagion_score", "risk_amplification"):
        assert key in out, f"Missing key: {key}"


def test_gnn_forward_shapes(gnn, x, adj):
    out = gnn(x, adj)
    assert out["node_embeddings"].shape   == (N, 8)
    assert out["contagion_score"].shape   == (N,)
    assert out["risk_amplification"].shape == (N,)


def test_gnn_contagion_range(gnn, x, adj):
    gnn.eval()
    out = gnn(x, adj)
    assert out["contagion_score"].min().item() >= 0.0
    assert out["contagion_score"].max().item() <= 1.0


def test_gnn_amplification_range(gnn, x, adj):
    """risk_amplification uses Tanh → [-1, 1]."""
    gnn.eval()
    out = gnn(x, adj)
    assert out["risk_amplification"].min().item() >= -1.0
    assert out["risk_amplification"].max().item() <= 1.0


def test_gnn_sparse_graph(gnn, x):
    """Zero-edge graph should produce valid output (no nan/inf)."""
    adj_zero = torch.eye(N)   # only self-loops
    out = gnn(x, adj_zero)
    assert not torch.isnan(out["node_embeddings"]).any()
    assert not torch.isinf(out["contagion_score"]).any()


def test_gnn_fully_connected(gnn, x):
    """Fully connected (uniform) graph should still be stable."""
    adj_full = torch.ones(N, N) / N
    out = gnn(x, adj_full)
    assert not torch.isnan(out["node_embeddings"]).any()


def test_gnn_no_nan(gnn, x, adj):
    out = gnn(x, adj)
    for key, val in out.items():
        assert not torch.isnan(val).any(), f"NaN in {key}"
        assert not torch.isinf(val).any(), f"Inf in {key}"


def test_gnn_single_node():
    """Single-node graph: should not crash."""
    gnn = RiskGNN(node_features=F, hidden=16, out_features=4, num_layers=2)
    x   = torch.rand(1, F)
    adj = torch.ones(1, 1)
    out = gnn(x, adj)
    assert out["node_embeddings"].shape == (1, 4)


def test_build_adjacency_basic():
    idx = {"US": 0, "RU": 1, "CN": 2}
    rows = [
        {"country_a": "RU", "country_b": "US", "spillover_weight": 0.7},
        {"country_a": "CN", "country_b": "RU", "spillover_weight": 0.5},
    ]
    adj = build_adjacency(idx, rows, min_weight=0.2, add_self_loops=True)
    assert adj.shape == (3, 3)
    # Undirected: both (RU, US) and (US, RU) should be > 0
    assert adj[0, 1].item() > 0
    assert adj[1, 0].item() > 0


def test_build_adjacency_min_weight_filter():
    idx  = {"A": 0, "B": 1, "C": 2}
    rows = [
        {"country_a": "A", "country_b": "B", "spillover_weight": 0.05},
        {"country_a": "B", "country_b": "C", "spillover_weight": 0.80},
    ]
    adj = build_adjacency(idx, rows, min_weight=0.20, add_self_loops=False)
    # Low-weight edge A-B should be 0
    assert adj[0, 1].item() == pytest.approx(0.0)
    # High-weight edge B-C should be > 0
    assert adj[1, 2].item() > 0


def test_build_adjacency_unknown_country():
    """Countries not in index should be silently skipped."""
    idx  = {"US": 0}
    rows = [{"country_a": "UNKNOWN", "country_b": "US", "spillover_weight": 0.9}]
    adj  = build_adjacency(idx, rows, add_self_loops=False)
    assert adj.shape == (1, 1)
    assert adj[0, 0].item() == pytest.approx(0.0)


def test_build_adjacency_row_normalised():
    idx  = {"A": 0, "B": 1, "C": 2}
    rows = [
        {"country_a": "A", "country_b": "B", "spillover_weight": 0.6},
        {"country_a": "A", "country_b": "C", "spillover_weight": 0.4},
    ]
    adj = build_adjacency(idx, rows, min_weight=0.1, add_self_loops=False)
    for i in range(3):
        row_sum = adj[i].sum().item()
        assert row_sum == pytest.approx(1.0, abs=0.01) or row_sum == pytest.approx(0.0, abs=0.01), \
            f"Row {i} not normalised: {row_sum}"
