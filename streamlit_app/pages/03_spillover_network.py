"""
Phase 2 — Global Spillover Network Page.

Visualizes the risk contagion graph:
  - Nodes = countries, sized by risk score
  - Edges = spillover weight (risk correlation + bilateral events)
  - Adjacency = geographic border pairs highlighted
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Spillover Network — GeoPulse",
    page_icon="🕸️",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #0d0d0d; color: #e0e0e0; }
    h1, h2, h3 { font-family: 'Courier New', monospace; color: #cc2222; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=600)
def get_spillover_pairs(min_weight: float = 0.3) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch heatmap data and name mapping."""
    try:
        resp = requests.get(f"{BACKEND_URL}/global/heatmap", timeout=10)
        countries_data = resp.json().get("countries", [])
        df = pd.DataFrame(countries_data)
        name_map = {}
        if not df.empty and "country" in df.columns:
            name_col = "name" if "name" in df.columns else "country"
            name_map = dict(zip(df["country"], df[name_col]))
        return df, name_map
    except Exception:
        return pd.DataFrame(), {}


@st.cache_data(ttl=600)
def get_neighbors(country: str) -> list[dict]:
    try:
        resp = requests.get(
            f"{BACKEND_URL}/country/{country}/spillover",
            params={"top_n": 10}, timeout=10,
        )
        return resp.json().get("neighbors", [])
    except Exception:
        return []


def main():
    st.markdown("<h1>🕸️ Risk Spillover Network</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#555; font-family:Courier New; font-size:12px;'>"
        "Risk contagion analysis — countries with correlated risk trajectories "
        "and high bilateral event co-occurrence.</p>",
        unsafe_allow_html=True,
    )

    # --- Controls ---
    col1, col2 = st.columns([2, 1])
    with col1:
        focus_country = st.text_input(
            "Focus Country (leave blank for global view)",
            value="", placeholder="e.g. PK",
        )
    with col2:
        min_weight = st.slider("Min Spillover Weight", 0.0, 1.0, 0.3, 0.05)

    df_heatmap, name_map = get_spillover_pairs()

    if df_heatmap.empty:
        st.error("No data available. Ensure backend is running and data is loaded.")
        return

    # --- Focus mode: single country network ---
    if focus_country.strip():
        country = focus_country.strip().upper()
        neighbors = get_neighbors(country)

        if not neighbors:
            st.warning(f"No spillover data for {country}. Run POST /analyze/spillover.")
        else:
            filtered = [n for n in neighbors if n.get("spillover_weight", 0) >= min_weight]
            display_name = name_map.get(country, country)
            fig = _build_ego_network(country, filtered, df_heatmap, name_map)
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": True})

            # Table
            st.markdown(f"**Top neighbors for {display_name}**")
            df_n = pd.DataFrame(filtered)
            if not df_n.empty:
                # Add human-readable name for neighbor
                if "neighbor" in df_n.columns:
                    df_n.insert(0, "Neighbor Name", df_n["neighbor"].map(lambda c: name_map.get(c, c)))
                for col in ["spillover_weight", "risk_correlation", "cooccurrence_score"]:
                    if col in df_n:
                        df_n[col] = df_n[col].round(4)
                df_n.columns = [c.replace("_", " ").title() for c in df_n.columns]
                st.dataframe(df_n, use_container_width=True, hide_index=True)

    else:
        # --- Global view: top-risk countries with risk scores ---
        st.markdown("### Global Risk Distribution")
        if "risk_score" in df_heatmap.columns:
            top_countries = (
                df_heatmap.dropna(subset=["risk_score"])
                .sort_values("risk_score", ascending=False)
                .head(30)
            )
            x_labels = top_countries["country"].map(lambda c: name_map.get(c, c))

            fig_bar = go.Figure(go.Bar(
                x=x_labels,
                y=top_countries["risk_score"],
                marker_color=[
                    _risk_color(r) for r in top_countries["risk_score"]
                ],
                text=top_countries["risk_score"].round(3),
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="Top 30 Countries by Risk Score",
                paper_bgcolor="#0d0d0d",
                plot_bgcolor="#0d0d0d",
                xaxis=dict(tickfont=dict(color="#ccc"), tickangle=-45),
                yaxis=dict(showgrid=True, gridcolor="#1a1a1a",
                           tickfont=dict(color="#888"), range=[0, 1]),
                font=dict(color="#888"),
                height=400,
                margin=dict(l=40, r=20, t=40, b=100),
            )
            st.plotly_chart(fig_bar, use_container_width=True,
                            config={"displayModeBar": False})

        st.info(
            "🔎 Enter a country code above to see its spillover network. "
            "Run POST /analyze/spillover first to populate network data."
        )


def _risk_color(score: float) -> str:
    if score >= 0.80: return "#000000"
    if score >= 0.65: return "#3b0000"
    if score >= 0.50: return "#6b0000"
    if score >= 0.35: return "#8b1a1a"
    return "#2d4a2d"


def _build_ego_network(
    center: str,
    neighbors: list[dict],
    heatmap_df: pd.DataFrame,
    name_map: dict[str, str] | None = None,
) -> go.Figure:
    """Build a simple ego-network graph for a focal country."""
    nm = name_map or {}
    risk_lookup = {}
    if not heatmap_df.empty and "country" in heatmap_df.columns:
        risk_lookup = dict(zip(
            heatmap_df["country"],
            heatmap_df.get("risk_score", [0.0] * len(heatmap_df)),
        ))

    # Positions: center at (0,0), neighbors on circle
    import math
    n = len(neighbors)
    positions = {center: (0, 0)}
    for i, nb in enumerate(neighbors):
        angle = 2 * math.pi * i / max(n, 1)
        r = 1.5
        positions[nb["neighbor"]] = (r * math.cos(angle), r * math.sin(angle))

    # Edges
    edge_x, edge_y = [], []
    for nb in neighbors:
        x0, y0 = positions[center]
        x1, y1 = positions[nb["neighbor"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    # Nodes
    all_nodes = [center] + [nb["neighbor"] for nb in neighbors]
    node_x = [positions[c][0] for c in all_nodes]
    node_y = [positions[c][1] for c in all_nodes]
    node_sizes = [
        30 + risk_lookup.get(c, 0.3) * 40 for c in all_nodes
    ]
    node_colors = [_risk_color(risk_lookup.get(c, 0.3)) for c in all_nodes]
    node_labels = [nm.get(c, c) for c in all_nodes]
    node_text = [
        f"{nm.get(c, c)}<br>Risk: {risk_lookup.get(c, 0):.3f}"
        for c in all_nodes
    ]

    fig = go.Figure()

    # Edge traces
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#3a3a3a", width=1),
        hoverinfo="none",
        showlegend=False,
    ))

    # Node traces
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(color="#cc2222", width=1),
        ),
        text=node_labels,
        textposition="top center",
        textfont=dict(color="#ccc", size=11),
        hovertext=node_text,
        hoverinfo="text",
        showlegend=False,
    ))

    center_name = nm.get(center, center)
    fig.update_layout(
        title=f"Spillover Network — {center_name}",
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#0d0d0d",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(color="#888"),
    )
    return fig


main()
