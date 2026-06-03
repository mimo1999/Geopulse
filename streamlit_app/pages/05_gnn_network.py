"""
Phase 3 — GNN Spillover Network Page.

Visualises the Graph Neural Network country influence graph.
Nodes = countries, coloured by contagion score (imported risk).
Edge thickness = spillover weight.
Sidebar = country inspector showing GNN enrichment detail.
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="GNN Network — GeoPulse",
    page_icon="🕸️",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #0d0d0d; color: #e0e0e0; }
    h1, h2, h3 { font-family: 'Courier New', monospace; color: #cc2222; }
    .node-card {
        background: #141414; border: 1px solid #2a2a2a;
        border-radius: 6px; padding: 14px 18px; margin: 6px 0;
    }
    .contagion-bar {
        height: 8px; border-radius: 4px; background: #1a3a6a;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def get_gnn_network(min_weight: float = 0.20) -> dict:
    try:
        r = requests.get(
            f"{BACKEND_URL}/global/gnn_network",
            params={"min_weight": min_weight}, timeout=20,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300)
def get_gnn_influence(country: str) -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}/country/{country}/gnn_influence", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300)
def get_countries() -> tuple[list[str], dict[str, str]]:
    try:
        r = requests.get(f"{BACKEND_URL}/countries", timeout=10)
        entries = r.json().get("countries", [])
        codes   = [x["country"] for x in entries]
        names   = {x["country"]: x.get("name", x["country"]) for x in entries}
        return codes, names
    except Exception:
        return [], {}


# ---------------------------------------------------------------------------
# GNN network chart
# ---------------------------------------------------------------------------

def build_gnn_graph(network_data: dict, name_map: dict[str, str] | None = None) -> go.Figure:
    nodes = network_data.get("nodes", [])
    edges = network_data.get("edges", [])

    if not nodes:
        fig = go.Figure()
        fig.add_annotation(
            text="No GNN network data. Run POST /analyze/gnn to compute.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#666", size=14),
        )
        fig.update_layout(
            paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            height=500,
        )
        return fig

    # Use circular layout for positioning (no spring-layout without networkx)
    import math
    nm = name_map or {}
    n = len(nodes)
    angles = [2 * math.pi * i / n for i in range(n)]
    country_pos = {}
    for i, node in enumerate(nodes):
        country_pos[node["country"]] = (
            math.cos(angles[i]),
            math.sin(angles[i]),
        )

    fig = go.Figure()

    # Draw edges
    for edge in edges:
        src = edge.get("source", "")
        dst = edge.get("target", "")
        w   = float(edge.get("weight", 0))
        if src not in country_pos or dst not in country_pos:
            continue
        x0, y0 = country_pos[src]
        x1, y1 = country_pos[dst]
        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(color=f"rgba(100,50,50,{min(w, 0.9):.2f})", width=max(1, w * 4)),
            hoverinfo="none",
            showlegend=False,
        ))

    # Draw nodes
    country_codes = [node["country"] for node in nodes]
    country_names = [nm.get(c, c) for c in country_codes]
    x_nodes    = [country_pos[c][0] for c in country_codes]
    y_nodes    = [country_pos[c][1] for c in country_codes]
    risk_scores = [float(node.get("risk_score", 0)) for node in nodes]
    contagion   = [float(node.get("contagion_score", 0)) for node in nodes]
    adj_risk    = [float(node.get("network_adjusted_risk", node.get("risk_score", 0))) for node in nodes]

    # Node color = contagion score (blue = imported risk, red = high base risk)
    node_colors = [
        f"rgb({int(200*r)}, {int(30*(1-c))}, {int(200*c)})"
        for r, c in zip(risk_scores, contagion)
    ]
    node_sizes = [max(8, 8 + 20 * r) for r in adj_risk]

    hover_texts = [
        f"<b>{name}</b><br>"
        f"Risk: {r:.3f}<br>"
        f"Contagion: {ct:.3f}<br>"
        f"Net-adj risk: {ar:.3f}"
        for name, r, ct, ar in zip(country_names, risk_scores, contagion, adj_risk)
    ]

    fig.add_trace(go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(color="#333", width=0.5),
        ),
        text=country_names,
        textposition="top center",
        textfont=dict(color="#888", size=9),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
        showlegend=False,
        name="Countries",
    ))

    fig.update_layout(
        title="GNN Country Risk Network — Node colour: risk (red) + contagion (blue)",
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#0d0d0d",
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-1.4, 1.4]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-1.4, 1.4]),
        font=dict(color="#888"),
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main():
    st.markdown(
        "<h1>🕸️ GNN Spillover Network</h1>"
        "<p style='color:#555; font-family:Courier New; font-size:13px;'>"
        "Graph Attention Network — second-order risk contagion between countries</p>",
        unsafe_allow_html=True,
    )

    col_net, col_inspect = st.columns([3, 1])

    with col_inspect:
        st.markdown("#### Country Inspector")
        countries, name_map = get_countries()
        country   = st.selectbox(
            "Select Country", countries or ["No data"],
            format_func=lambda x: name_map.get(x, x),
            key="gnn_country",
        )

        min_weight = st.slider("Min Edge Weight", 0.10, 0.60, 0.20, 0.05)

        if st.button("🔄 Recompute GNN", use_container_width=True):
            try:
                r = requests.post(f"{BACKEND_URL}/analyze/gnn", json={}, timeout=10)
                if r.ok:
                    st.success("GNN computation triggered (background)")
                else:
                    st.error(f"Error: {r.status_code}")
            except Exception as e:
                st.error(str(e))
            st.cache_data.clear()

        st.divider()

        if country and country != "No data":
            display_name = name_map.get(country, country)
            gnn = get_gnn_influence(country)
            if "error" in gnn:
                st.info(
                    "No GNN data available. Run POST /analyze/gnn or "
                    "click Recompute GNN above."
                )
            else:
                contagion = gnn.get("contagion_score", 0)
                amplif    = gnn.get("risk_amplification", 0)
                adj_risk  = gnn.get("network_adjusted_risk", 0)
                base_risk = gnn.get("base_risk_score", adj_risk)

                st.markdown(
                    f"""
                    <div class="node-card">
                        <div style="font-family:Courier New; font-size:11px; color:#555;">
                            GNN ENRICHMENT — {display_name}
                        </div>
                        <br/>
                        <b style="color:#8888ff">Contagion Score</b><br/>
                        <div style="font-size:22px; color:#8888ff;">{contagion:.3f}</div>
                        <div style="color:#555; font-size:11px;">Risk imported from neighbours</div>
                        <br/>
                        <b style="color:#cc6622">Risk Amplification</b><br/>
                        <div style="font-size:22px; color:#cc6622;">{amplif:+.3f}</div>
                        <div style="color:#555; font-size:11px;">Network delta vs base model</div>
                        <br/>
                        <b style="color:#cc2222">Network-Adjusted Risk</b><br/>
                        <div style="font-size:22px; color:#cc2222;">{adj_risk:.3f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Top influencers
                influencers = gnn.get("top_influencers", [])
                if influencers:
                    st.markdown("**Top Influencers**")
                    for inf in influencers[:5]:
                        inf_c = inf.get("country", "")
                        inf_name = name_map.get(inf_c, inf_c)
                        inf_w = inf.get("spillover_weight", 0)
                        st.markdown(
                            f"<div style='font-family:Courier New; font-size:12px; "
                            f"color:#888; margin:3px 0;'>"
                            f"→ <b style='color:#ccc'>{inf_name}</b>  "
                            f"<span style='color:#555'>w={inf_w:.3f}</span></div>",
                            unsafe_allow_html=True,
                        )

    with col_net:
        network_data = get_gnn_network(min_weight)
        if "error" in network_data:
            st.info(
                "No GNN network data available. "
                "Run POST /analyze/gnn to compute the graph, "
                "or click 'Recompute GNN' in the sidebar."
            )
            # Show raw spillover as fallback
            st.markdown("#### Fallback: Spillover Edge Table")
            try:
                r = requests.get(f"{BACKEND_URL}/country/{countries[0] if countries else 'US'}/spillover",
                                 params={"top_n": 10}, timeout=10)
                if r.ok:
                    data = r.json()
                    neighbors = data.get("neighbors", [])
                    if neighbors:
                        df = pd.DataFrame(neighbors)
                        st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception:
                pass
        else:
            fig = build_gnn_graph(network_data, name_map)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

            # Network statistics
            nodes = network_data.get("nodes", [])
            edges = network_data.get("edges", [])
            if nodes:
                st.markdown("---")
                nc1, nc2, nc3 = st.columns(3)
                nc1.metric("Countries (Nodes)", len(nodes))
                nc2.metric("Connections (Edges)", len(edges))
                avg_contagion = sum(n.get("contagion_score", 0) for n in nodes) / max(len(nodes), 1)
                nc3.metric("Avg Contagion Score", f"{avg_contagion:.3f}")

                # Top contagion countries
                st.markdown("**Highest Contagion Scores** (most risk imported)")
                df_nodes = pd.DataFrame(nodes)
                if "contagion_score" in df_nodes.columns:
                    top_c = df_nodes.sort_values("contagion_score", ascending=False).head(10).copy()
                    top_c.insert(0, "Country", top_c["country"].map(lambda c: name_map.get(c, c)))
                    display_cols = [c for c in ["Country", "contagion_score",
                                                "risk_amplification", "network_adjusted_risk",
                                                "risk_score"] if c in top_c.columns]
                    st.dataframe(
                        top_c[display_cols],
                        use_container_width=True,
                        hide_index=True,
                    )


main()
