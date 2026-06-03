"""
Phase 2 — Event Explorer Page.

Browse and filter recent high-impact GDELT events:
  - Filter by country, category, date range, min mentions
  - Sort by intensity (Goldstein scale) or mentions
  - Actor pair network visualization
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, date

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Event Explorer — GeoPulse",
    page_icon="📰",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #0d0d0d; color: #e0e0e0; }
    h1, h2, h3 { font-family: 'Courier New', monospace; color: #cc2222; }
    .event-row {
        background: #141414;
        border-left: 3px solid #6b0000;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 0 4px 4px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def get_countries() -> tuple[list[str], dict[str, str]]:
    try:
        resp = requests.get(f"{BACKEND_URL}/countries", timeout=10)
        entries = resp.json().get("countries", [])
        codes   = [r["country"] for r in entries]
        names   = {r["country"]: r.get("name", r["country"]) for r in entries}
        return codes, names
    except Exception:
        return [], {}


@st.cache_data(ttl=120)
def get_events(country: str, days: int, category: str | None) -> list[dict]:
    try:
        resp = requests.get(
            f"{BACKEND_URL}/country/{country}/events",
            params={"days": days}, timeout=10,
        )
        events = resp.json().get("events", [])
        if category and category != "All":
            events = [e for e in events if e.get("category") == category.lower()]
        return events
    except Exception:
        return []


CATEGORIES = ["All", "Military", "Terrorism", "Protest", "Sanctions", "Diplomatic"]
CATEGORY_COLORS = {
    "protest":    "#b8860b",
    "military":   "#8b0000",
    "terrorism":  "#4b0000",
    "sanctions":  "#4a4a8a",
    "diplomatic": "#2a5a2a",
}


def main():
    st.markdown("<h1>📰 Event Explorer</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#555; font-family:Courier New; font-size:12px;'>"
        "Browse aggregated GDELT event clusters by country, category, and time period.</p>",
        unsafe_allow_html=True,
    )

    # --- Filters ---
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        countries, name_map = get_countries()
        country = st.selectbox(
            "Country", countries if countries else ["—"],
            format_func=lambda x: name_map.get(x, x),
            key="explorer_country",
        )
    with col2:
        category = st.selectbox("Category", CATEGORIES, key="explorer_category")
    with col3:
        days = st.select_slider("Window", [7, 14, 30, 60, 90, 180], value=30)

    if not country or country == "—":
        return

    events = get_events(country, days, category if category != "All" else None)

    if not events:
        st.warning("No events found for the selected filters.")
        return

    df = pd.DataFrame(events)
    df["cluster_date"] = pd.to_datetime(df["cluster_date"])
    df = df.sort_values(["cluster_date", "total_mentions"], ascending=[False, False])

    # --- Summary metrics ---
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Event Groups", len(df))
    c2.metric("Total Mentions",     f"{df['total_mentions'].sum():,}")
    c3.metric("Avg Goldstein",      f"{df['avg_goldstein'].mean():.2f}" if "avg_goldstein" in df else "N/A")
    c4.metric("Days Covered",       df["cluster_date"].nunique())

    # --- Category breakdown ---
    st.divider()
    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown("**By Category**")
        cat_summary = (
            df.groupby("category")["total_mentions"].sum()
            .reset_index()
            .sort_values("total_mentions", ascending=True)
        )
        fig_pie = go.Figure(go.Bar(
            x=cat_summary["total_mentions"],
            y=cat_summary["category"],
            orientation="h",
            marker_color=[CATEGORY_COLORS.get(c, "#555") for c in cat_summary["category"]],
        ))
        fig_pie.update_layout(
            paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
            xaxis=dict(tickfont=dict(color="#888"), showgrid=True, gridcolor="#1a1a1a"),
            yaxis=dict(tickfont=dict(color="#ccc")),
            height=220, margin=dict(l=100, r=20, t=10, b=20),
            font=dict(color="#888"),
        )
        st.plotly_chart(fig_pie, use_container_width=True,
                        config={"displayModeBar": False})

    with col_r:
        st.markdown("**Intensity Over Time**")
        fig_intensity = go.Figure()
        for cat, color in CATEGORY_COLORS.items():
            sub = df[df["category"] == cat]
            if sub.empty:
                continue
            fig_intensity.add_trace(go.Scatter(
                x=sub["cluster_date"],
                y=sub["max_intensity"].abs() if "max_intensity" in sub else sub["avg_goldstein"].abs(),
                name=cat.title(),
                mode="markers+lines",
                line=dict(color=color, width=1),
                marker=dict(color=color, size=6),
            ))
        fig_intensity.update_layout(
            paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
            xaxis=dict(tickfont=dict(color="#888"), showgrid=True, gridcolor="#1a1a1a"),
            yaxis=dict(tickfont=dict(color="#888"), showgrid=True, gridcolor="#1a1a1a",
                       title="|Intensity|"),
            legend=dict(bgcolor="#141414", bordercolor="#2a2a2a",
                        font=dict(color="#888", size=10)),
            height=220, margin=dict(l=60, r=20, t=10, b=40),
            font=dict(color="#888"),
        )
        st.plotly_chart(fig_intensity, use_container_width=True,
                        config={"displayModeBar": False})

    # --- Event table ---
    st.divider()
    st.markdown("**Event Log**")

    display_cols = ["cluster_date", "category", "event_count",
                    "total_mentions", "avg_goldstein", "avg_tone", "max_intensity"]
    display_cols = [c for c in display_cols if c in df.columns]
    display_df = df[display_cols].copy()
    display_df["cluster_date"] = display_df["cluster_date"].dt.date
    for col in ["avg_goldstein", "avg_tone", "max_intensity"]:
        if col in display_df:
            display_df[col] = display_df[col].round(3)

    display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


main()
