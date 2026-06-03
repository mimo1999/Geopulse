"""
Phase 3: GNN-based country risk enrichment engine.

Loads the spillover adjacency graph from country_spillover table,
fetches latest risk scores + features per country, runs RiskGNN,
and persists contagion scores + embeddings to gnn_node_embeddings.

Also provides network-adjusted risk scores that account for
second-order spillover effects (A→B→C) missed by direct correlation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import psycopg2
import psycopg2.extras
import torch

from models.gnn import RiskGNN, build_adjacency, NODE_FEATURE_DIM

logger = logging.getLogger("inference.gnn_spillover")


# ---------------------------------------------------------------------------
# GNN Enrichment Engine
# ---------------------------------------------------------------------------

class GNNSpilloverEngine:
    """
    Computes GNN-based risk enrichment for all tracked countries.

    Usage::

        engine = GNNSpilloverEngine(dsn=dsn)
        results = engine.enrich(as_of=date.today())
    """

    _FETCH_RISK_SQL = """
        SELECT
            l.country,
            COALESCE(l.risk_score, 0.0)        AS risk_score,
            COALESCE(p.instability_score, 0.0)  AS instability,
            COALESCE(p.war_probability, 0.0)    AS war,
            COALESCE(p.terrorism_risk, 0.0)     AS terrorism,
            COALESCE(p.financial_stress, 0.0)   AS financial,
            COALESCE(p.confidence, 0.5)         AS confidence,
            COALESCE(l.protest_score, 0.0)      AS protest_score,
            COALESCE(l.violence_score, 0.0)     AS violence_score,
            COALESCE(l.diplomatic_stress, 0.0)  AS diplomatic_stress,
            COALESCE(l.economic_stress, 0.0)    AS economic_stress,
            COALESCE(l.terrorism_score, 0.0)    AS terrorism_score,
            COALESCE(l.avg_sentiment, 0.0)      AS avg_sentiment
        FROM latest_country_risk l
        LEFT JOIN LATERAL (
            SELECT instability_score, war_probability, terrorism_risk,
                   financial_stress, confidence
            FROM country_risk_predictions
            WHERE country = l.country
            ORDER BY prediction_time DESC
            LIMIT 1
        ) p ON TRUE
        WHERE l.risk_score IS NOT NULL
    """

    _FETCH_SPILLOVER_SQL = """
        SELECT country_a, country_b, spillover_weight
        FROM country_spillover
        WHERE computed_date = (SELECT MAX(computed_date) FROM country_spillover)
          AND spillover_weight >= %s
    """

    _UPSERT_EMBEDDING_SQL = """
        INSERT INTO gnn_node_embeddings (
            country, computed_date, embedding,
            contagion_score, risk_amplification, network_adjusted_risk
        ) VALUES (
            %(country)s, %(computed_date)s, %(embedding)s,
            %(contagion_score)s, %(risk_amplification)s, %(network_adjusted_risk)s
        )
        ON CONFLICT (country, computed_date)
        DO UPDATE SET
            embedding             = EXCLUDED.embedding,
            contagion_score       = EXCLUDED.contagion_score,
            risk_amplification    = EXCLUDED.risk_amplification,
            network_adjusted_risk = EXCLUDED.network_adjusted_risk,
            computed_at           = NOW()
    """

    _FETCH_EMBEDDING_SQL = """
        SELECT contagion_score, risk_amplification, network_adjusted_risk, embedding
        FROM gnn_node_embeddings
        WHERE country = %s AND computed_date = %s
    """

    def __init__(
        self,
        dsn: str,
        model_path: Optional[str] = None,
        min_edge_weight: float = 0.15,
        device: str = "cpu",
    ):
        self._dsn             = dsn
        self._min_edge_weight = min_edge_weight
        self._device          = device

        # Load pre-trained GNN or use default-init weights
        if model_path and __import__("pathlib").Path(model_path).exists():
            self._model = RiskGNN.load(model_path, device)
            logger.info("GNN loaded from %s (%d params)", model_path, self._model.parameter_count())
        else:
            self._model = RiskGNN(
                node_features=NODE_FEATURE_DIM,
                hidden=64,
                out_features=8,
                num_layers=2,
            ).to(device)
            logger.info("GNN using default weights (no checkpoint found)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(
        self,
        as_of: Optional[date] = None,
        persist: bool = True,
    ) -> dict[str, dict]:
        """
        Run GNN enrichment for all countries and return per-country results.

        Args:
            as_of:   Date for persisted results (default today).
            persist: Write embeddings to gnn_node_embeddings table.

        Returns:
            {country: {contagion_score, risk_amplification,
                        network_adjusted_risk, embedding: list[float]}}
        """
        if as_of is None:
            as_of = date.today()

        # 1. Load node features from DB
        node_data = self._fetch_node_features()
        if not node_data:
            logger.warning("No node features available for GNN enrichment")
            return {}

        countries = list(node_data.keys())
        country_idx = {c: i for i, c in enumerate(countries)}
        N = len(countries)

        # Build feature matrix (N, 12)
        x = np.zeros((N, NODE_FEATURE_DIM), dtype=np.float32)
        for c, feats in node_data.items():
            x[country_idx[c]] = feats

        # 2. Load spillover adjacency
        spillover_rows = self._fetch_spillover()
        adj = build_adjacency(country_idx, spillover_rows, self._min_edge_weight)

        # 3. Run GNN
        x_tensor   = torch.from_numpy(x).to(self._device)
        adj_tensor = adj.to(self._device)

        with torch.no_grad():
            gnn_out = self._model(x_tensor, adj_tensor)

        contagion = gnn_out["contagion_score"].cpu().numpy()     # (N,)
        amplif    = gnn_out["risk_amplification"].cpu().numpy()  # (N,)
        embeds    = gnn_out["node_embeddings"].cpu().numpy()     # (N, 8)

        # 4. Compute network-adjusted risk
        base_risks = x[:, 0]   # first feature is risk_score
        # Scale amplification to ±0.15 delta range
        adjusted_risks = np.clip(
            base_risks + amplif * 0.15,
            0.0, 1.0,
        )

        # 5. Package results
        results = {}
        for c in countries:
            i = country_idx[c]
            results[c] = {
                "contagion_score":      float(contagion[i]),
                "risk_amplification":   float(amplif[i]),
                "network_adjusted_risk": float(adjusted_risks[i]),
                "embedding":            embeds[i].tolist(),
            }

        # 6. Persist if requested
        if persist:
            self._persist(results, as_of)

        return results

    def fetch_country_enrichment(
        self,
        country: str,
        as_of: Optional[date] = None,
    ) -> Optional[dict]:
        """Fetch stored GNN enrichment from DB."""
        if as_of is None:
            as_of = date.today()
        try:
            conn = psycopg2.connect(self._dsn)
            with conn, conn.cursor() as cur:
                cur.execute(self._FETCH_EMBEDDING_SQL, (country, as_of))
                row = cur.fetchone()
            conn.close()
            if not row:
                return None
            contagion, amplif, adj_risk, embedding = row
            return {
                "contagion_score":       float(contagion or 0),
                "risk_amplification":    float(amplif or 0),
                "network_adjusted_risk": float(adj_risk or 0),
                "embedding":             list(embedding or []),
            }
        except Exception as exc:
            logger.warning("Fetch GNN embedding error for %s: %s", country, exc)
            return None

    def get_top_influencers(
        self,
        country: str,
        top_n: int = 5,
        as_of: Optional[date] = None,
    ) -> list[dict]:
        """Return the top-N countries influencing this country via the GNN graph."""
        try:
            conn = psycopg2.connect(self._dsn)
            with conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT country_a AS influencer, spillover_weight
                    FROM country_spillover
                    WHERE country_b = %s
                      AND computed_date = (SELECT MAX(computed_date) FROM country_spillover)
                    UNION
                    SELECT country_b AS influencer, spillover_weight
                    FROM country_spillover
                    WHERE country_a = %s
                      AND computed_date = (SELECT MAX(computed_date) FROM country_spillover)
                    ORDER BY spillover_weight DESC
                    LIMIT %s
                """, (country, country, top_n))
                rows = cur.fetchall()
            conn.close()
            return [{"country": r[0], "spillover_weight": float(r[1])} for r in rows]
        except Exception as exc:
            logger.warning("Fetch influencers error for %s: %s", country, exc)
            return []

    # ------------------------------------------------------------------
    # Internal data loaders
    # ------------------------------------------------------------------

    def _fetch_node_features(self) -> dict[str, np.ndarray]:
        """Fetch 12-dim feature vector per country from DB."""
        try:
            conn = psycopg2.connect(self._dsn)
            with conn, conn.cursor() as cur:
                cur.execute(self._FETCH_RISK_SQL)
                rows = cur.fetchall()
            conn.close()
        except Exception as exc:
            logger.warning("GNN fetch node features failed: %s", exc)
            return {}

        result: dict[str, np.ndarray] = {}
        for row in rows:
            country = row[0]
            feats   = np.array([float(v or 0) for v in row[1:]], dtype=np.float32)
            if feats.shape[0] == NODE_FEATURE_DIM:
                result[country] = feats
        return result

    def _fetch_spillover(self) -> list[dict]:
        """Fetch latest spillover pairs from DB."""
        try:
            conn = psycopg2.connect(self._dsn)
            with conn, conn.cursor() as cur:
                cur.execute(self._FETCH_SPILLOVER_SQL, (self._min_edge_weight,))
                rows = cur.fetchall()
            conn.close()
            return [
                {"country_a": r[0], "country_b": r[1], "spillover_weight": r[2]}
                for r in rows
            ]
        except Exception as exc:
            logger.warning("GNN fetch spillover failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self, results: dict[str, dict], computed_date: date) -> None:
        rows = []
        for country, data in results.items():
            rows.append({
                "country":               country,
                "computed_date":         computed_date,
                "embedding":             data["embedding"],
                "contagion_score":       data["contagion_score"],
                "risk_amplification":    data["risk_amplification"],
                "network_adjusted_risk": data["network_adjusted_risk"],
            })

        try:
            conn = psycopg2.connect(self._dsn)
            with conn, conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, self._UPSERT_EMBEDDING_SQL, rows)
            conn.close()
            logger.info("GNN: persisted %d node embeddings for %s", len(rows), computed_date)
        except Exception as exc:
            logger.warning("GNN persist failed: %s", exc)
