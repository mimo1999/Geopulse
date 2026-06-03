"""
Phase 3: RiskGNN — Graph Attention Network for country risk enrichment.

Graph definition:
    Nodes  = countries  (N)
    Node features  = [risk_score, instability, war, terrorism, financial,
                       confidence, protest_score, violence_score,
                       diplomatic_stress, economic_stress, terrorism_score,
                       avg_goldstein]  → 12-dimensional
    Edges  = spillover pairs from country_spillover table
    Weights = spillover_weight (0–1)

GAT layer (Veličković et al., 2018 — simplified single-head variant):
    e_ij    = LeakyReLU(a^T [W·h_i || W·h_j])   attention logit
    α_ij    = softmax_j(e_ij)                     normalized weights
    h'_i    = σ(Σ_j α_ij · W · h_j)              aggregated message

Two-layer stack:
    Layer 1: (node_features=12 → hidden=64)  with ELU + dropout
    Layer 2: (64 → out_features=8)           with ELU
    Output heads:
        contagion_score:     (N,)  risk imported from the network
        risk_amplification:  (N,)  net change in risk due to neighbors

Pure PyTorch — no torch_geometric or similar required.
Adjacency matrix is always row-normalized to prevent activation explosion.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("models.gnn")

NODE_FEATURE_DIM = 12   # fixed: see module docstring


# ---------------------------------------------------------------------------
# Graph Attention Layer
# ---------------------------------------------------------------------------

class GraphAttentionLayer(nn.Module):
    """
    Single-head Graph Attention Layer.

    Args:
        in_features:  Input node feature dimension.
        out_features: Output node feature dimension.
        dropout:      Dropout on attention weights.
        alpha:        LeakyReLU negative slope.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.in_f  = in_features
        self.out_f = out_features
        self.dropout = dropout

        self.W = nn.Linear(in_features, out_features, bias=False)
        # Attention vector: a^T [W·h_i || W·h_j] = a1^T W·h_i + a2^T W·h_j
        self.a1 = nn.Linear(out_features, 1, bias=False)
        self.a2 = nn.Linear(out_features, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)
        self.attn_drop  = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a1.weight)
        nn.init.xavier_uniform_(self.a2.weight)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Args:
            x:   (N, in_features)   node features
            adj: (N, N)             adjacency matrix (row-normalised, float)
                                    adj[i, j] > 0 means edge i→j exists.

        Returns:
            h:  (N, out_features)   updated node embeddings
        """
        N = x.size(0)

        h = self.W(x)   # (N, out_features)

        # Compute attention coefficients
        # e_ij = LeakyReLU(a1(h_i) + a2(h_j))
        e_i = self.a1(h)   # (N, 1)
        e_j = self.a2(h)   # (N, 1)
        # Broadcast: (N, 1) + (1, N) → (N, N)
        e = self.leaky_relu(e_i + e_j.T)     # (N, N)

        # Mask: only attend to existing edges
        # Set non-edges to -1e9 so softmax gives ~0
        mask = (adj > 0).float()
        e = e * mask + (-1e9) * (1.0 - mask)

        alpha = F.softmax(e, dim=1)          # (N, N) row-softmax
        alpha = self.attn_drop(alpha)

        # Also apply edge weights from adjacency
        alpha = alpha * adj                  # scale by spillover strength

        # Re-normalise after weighting (avoid zero rows)
        row_sum = alpha.sum(dim=1, keepdim=True).clamp(min=1e-8)
        alpha   = alpha / row_sum

        h_new = torch.mm(alpha, h)           # (N, out_features)
        return F.elu(h_new)


# ---------------------------------------------------------------------------
# RiskGNN
# ---------------------------------------------------------------------------

class RiskGNN(nn.Module):
    """
    Two-layer GAT for country risk enrichment.

    Args:
        node_features: Input node feature size (default 12).
        hidden:        Hidden layer size (default 64).
        out_features:  Output embedding size (default 8).
        num_layers:    Number of GAT layers (default 2).
        dropout:       Dropout probability (default 0.1).
    """

    def __init__(
        self,
        node_features: int = NODE_FEATURE_DIM,
        hidden: int = 64,
        out_features: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert num_layers >= 1

        layers = []
        in_dim = node_features
        for i in range(num_layers - 1):
            layers.append(GraphAttentionLayer(in_dim, hidden, dropout=dropout))
            in_dim = hidden
        layers.append(GraphAttentionLayer(in_dim, out_features, dropout=dropout))

        self.gat_layers = nn.ModuleList(layers)
        self.dropout    = nn.Dropout(dropout)

        # Output heads
        self.contagion_head   = nn.Sequential(
            nn.Linear(out_features, 1), nn.Sigmoid()
        )
        self.amplif_head      = nn.Sequential(
            nn.Linear(out_features, 1), nn.Tanh()   # signed delta
        )

        self._init_weights()
        self.eval()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, adj: Tensor) -> dict[str, Tensor]:
        """
        Args:
            x:   (N, node_features)    node feature matrix
            adj: (N, N)                adjacency (row-normalised)

        Returns:
            node_embeddings:   (N, out_features)
            contagion_score:   (N,)   imported risk [0, 1]
            risk_amplification:(N,)   network risk delta [-1, 1]
        """
        h = x
        for i, layer in enumerate(self.gat_layers):
            h = layer(h, adj)
            if i < len(self.gat_layers) - 1:
                h = self.dropout(h)

        contagion   = self.contagion_head(h).squeeze(-1)    # (N,)
        amplif      = self.amplif_head(h).squeeze(-1)       # (N,)

        return {
            "node_embeddings":    h,
            "contagion_score":    contagion,
            "risk_amplification": amplif,
        }

    def save(self, path: str) -> None:
        # Infer config from first and last layer
        first_layer = self.gat_layers[0]
        last_layer  = self.gat_layers[-1]
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "node_features": first_layer.in_f,
                "hidden":        first_layer.out_f,
                "out_features":  last_layer.out_f,
                "num_layers":    len(self.gat_layers),
            },
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "RiskGNN":
        ckpt   = torch.load(path, map_location=device)
        model  = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Adjacency builder
# ---------------------------------------------------------------------------

def build_adjacency(
    country_index: dict[str, int],
    spillover_rows: list[dict],
    min_weight: float = 0.15,
    add_self_loops: bool = True,
) -> Tensor:
    """
    Build a normalised adjacency matrix from spillover rows.

    Args:
        country_index:  {country_code: node_index}
        spillover_rows: list of dicts with keys country_a, country_b,
                        spillover_weight
        min_weight:     Edges below this weight are pruned.
        add_self_loops: Add weight-1 self-connections for every node.

    Returns:
        adj: (N, N) float32 tensor, row-normalised.
    """
    N   = len(country_index)
    adj = torch.zeros(N, N)

    if add_self_loops:
        for i in range(N):
            adj[i, i] = 1.0

    for row in spillover_rows:
        a = row.get("country_a") or row.get("country")
        b = row.get("country_b") or row.get("neighbor")
        w = float(row.get("spillover_weight", 0.0))

        if w < min_weight:
            continue
        if a not in country_index or b not in country_index:
            continue

        i, j = country_index[a], country_index[b]
        adj[i, j] = max(adj[i, j].item(), w)
        adj[j, i] = max(adj[j, i].item(), w)   # undirected

    # Row-normalise
    row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
    adj = adj / row_sum

    return adj
