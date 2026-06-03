"""
Phase 3 — Escalation Forecast Page.

Shows 4-step bi-weekly risk trajectory with confidence intervals.
Includes global escalation alerts for countries predicted to spike.
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Escalation Forecast — GeoPulse",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #0d0d0d; color: #e0e0e0; }
    h1, h2, h3 { font-family: 'Courier New', monospace; color: #cc2222; }
    .forecast-card {
        background: #141414;
        border: 1px solid #2a2a2a;
        border-left: 3px solid #cc2222;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 6px 0;
    }
    .step-label {
        font-size: 11px; letter-spacing: 1px; color: #555; font-family: Courier New;
    }
    .step-score {
        font-size: 28px; font-weight: bold; color: #cc2222; margin: 4px 0;
    }
    .alert-row {
        background: #1a0000; border: 1px solid #3b0000;
        border-radius: 4px; padding: 8px 12px; margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

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


@st.cache_data(ttl=300)
def get_forecast(country: str) -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}/country/{country}/forecast", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300)
def get_escalation_alerts(min_risk: float = 0.60) -> list[dict]:
    try:
        r = requests.get(
            f"{BACKEND_URL}/global/escalation_alerts",
            params={"min_risk": min_risk}, timeout=10,
        )
        r.raise_for_status()
        return r.json().get("alerts", [])
    except Exception:
        return []


@st.cache_data(ttl=120)
def get_current_risk(country: str) -> dict:
    try:
        r = requests.post(f"{BACKEND_URL}/riskscore", json={"country": country}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def build_forecast_chart(country: str, forecast: dict, current_risk: dict) -> go.Figure:
    steps = forecast.get("forecasts", [])
    if not steps:
        return go.Figure()

    dates      = [s["target_date"] for s in steps]
    risk_mean  = [s["risk_score"] for s in steps]
    lower      = [s["lower_bound"] for s in steps]
    upper      = [s["upper_bound"] for s in steps]
    conf       = [s["confidence"] for s in steps]

    # Prepend current point
    current_date  = forecast.get("forecast_date", str(datetime.today().date()))
    current_score = current_risk.get("risk_score", None)

    fig = go.Figure()

    # Confidence interval ribbon
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(204, 34, 34, 0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% Confidence Interval",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Forecast line
    all_x = ([current_date] + dates) if current_score is not None else dates
    all_y = ([current_score] + risk_mean) if current_score is not None else risk_mean

    fig.add_trace(go.Scatter(
        x=all_x,
        y=all_y,
        mode="lines+markers",
        name="Forecast Risk",
        line=dict(color="#cc2222", width=2.5),
        marker=dict(
            size=[10] + [8] * len(dates),
            color=["#ff8800"] + ["#cc2222"] * len(dates),
            symbol=["diamond"] + ["circle"] * len(dates),
        ),
    ))

    # Threshold lines
    for level, y, color in [
        ("CRITICAL", 0.80, "#550000"),
        ("HIGH",     0.65, "#3b0000"),
        ("ELEVATED", 0.50, "#2a2a00"),
    ]:
        fig.add_hline(
            y=y, line_dash="dot", line_color=color, line_width=1,
            annotation_text=level, annotation_position="right",
            annotation_font_color=color,
        )

    fig.update_layout(
        title=f"{country} — Escalation Forecast (4-step, bi-weekly)",
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#0d0d0d",
        xaxis=dict(
            showgrid=True, gridcolor="#1a1a1a",
            tickfont=dict(color="#888"),
            title="Target Date",
        ),
        yaxis=dict(
            range=[0, 1.05],
            showgrid=True, gridcolor="#1a1a1a",
            tickfont=dict(color="#888"),
            title="Predicted Risk Score",
        ),
        legend=dict(bgcolor="#141414", bordercolor="#2a2a2a", font=dict(color="#888")),
        font=dict(color="#888"),
        height=380,
        margin=dict(l=60, r=80, t=50, b=50),
    )
    return fig


def build_task_ribbon_chart(country: str, steps: list[dict]) -> go.Figure:
    """Show per-task forecasts as area chart."""
    dates = [s["target_date"] for s in steps]
    tasks = [
        ("instability",     "Instability",  "#cc4422"),
        ("war_probability", "War Risk",     "#8b0000"),
        ("terrorism_risk",  "Terrorism",    "#6b2200"),
        ("financial_stress","Financial",    "#3a5a7a"),
    ]
    fig = go.Figure()
    for key, name, color in tasks:
        y = [s[key] for s in steps]
        fig.add_trace(go.Scatter(
            x=dates, y=y, name=name,
            mode="lines+markers",
            line=dict(color=color, width=1.5, dash="dot"),
            marker=dict(size=6),
        ))
    fig.update_layout(
        title=f"{country} — Per-Task Forecast Breakdown",
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        xaxis=dict(showgrid=True, gridcolor="#1a1a1a", tickfont=dict(color="#888")),
        yaxis=dict(range=[0, 1], showgrid=True, gridcolor="#1a1a1a",
                   tickfont=dict(color="#888")),
        legend=dict(bgcolor="#141414", bordercolor="#2a2a2a", font=dict(color="#888", size=10)),
        font=dict(color="#888"), height=260,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main():
    st.markdown(
        "<h1>📈 Escalation Forecast</h1>"
        "<p style='color:#555; font-family:Courier New; font-size:13px;'>"
        "Multi-step ahead risk trajectory · 4 bi-weekly periods (≈ 8 weeks)</p>",
        unsafe_allow_html=True,
    )

    countries, name_map = get_countries()
    if not countries:
        st.error("Backend unavailable or no countries loaded.")
        return

    col_sel, col_thresh = st.columns([3, 1])
    with col_sel:
        country = st.selectbox(
            "Select Country", countries,
            format_func=lambda x: name_map.get(x, x),
            key="forecast_country",
        )
    with col_thresh:
        alert_thresh = st.slider("Alert Threshold", 0.40, 0.90, 0.60, 0.05)

    if not country:
        return

    display_name = name_map.get(country, country)

    # Fetch data
    forecast     = get_forecast(country)
    current_risk = get_current_risk(country)

    if "error" in forecast or "forecasts" not in forecast:
        st.warning(
            "No forecast available for this country. "
            "Train and deploy the forecaster model first, or the country may have "
            "insufficient feature data. The system is using trend extrapolation as fallback."
        )
        # Try to show trend extrapolation from riskscore endpoint
        if current_risk:
            st.info(f"Current risk score: **{current_risk.get('risk_score', 'N/A'):.3f}**  |  "
                    f"Trend: {current_risk.get('trend', 'N/A')}")
        return

    steps = forecast.get("forecasts", [])

    # ---- Header metrics ----
    if steps and current_risk:
        current_score = current_risk.get("risk_score", 0)
        final_score   = steps[-1]["risk_score"]
        delta         = final_score - current_score

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Current Risk", f"{current_score:.3f}")
        c2.metric("14-day Forecast", f"{steps[0]['risk_score']:.3f}",
                  delta=f"{steps[0]['risk_score']-current_score:+.3f}")
        c3.metric("28-day Forecast", f"{steps[1]['risk_score']:.3f}" if len(steps)>1 else "N/A")
        c4.metric("56-day Forecast", f"{steps[-1]['risk_score']:.3f}",
                  delta=f"{delta:+.3f}")
        c5.metric("Model Version", forecast.get("model_version", "N/A"))

    st.divider()

    # ---- Main forecast chart ----
    fig = build_forecast_chart(display_name, forecast, current_risk)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ---- Step detail cards ----
    st.markdown("### Forecast Steps")
    cols = st.columns(len(steps))
    for i, (step, col) in enumerate(zip(steps, cols)):
        score = step["risk_score"]
        conf  = step["confidence"]
        lo    = step["lower_bound"]
        hi    = step["upper_bound"]
        # Level color
        border_col = "#cc2222" if score >= 0.65 else "#8b4444" if score >= 0.50 else "#2a5a2a"
        with col:
            st.markdown(
                f"""
                <div class="forecast-card" style="border-left-color:{border_col};">
                    <div class="step-label">STEP {step['step']} — {step['target_date']}</div>
                    <div class="step-score">{score:.3f}</div>
                    <div style="color:#666; font-size:11px;">
                        CI: [{lo:.3f}, {hi:.3f}]<br/>
                        Conf: {conf:.0%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ---- Per-task breakdown ----
    st.markdown("### Per-Task Breakdown")
    fig2 = build_task_ribbon_chart(display_name, steps)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ---- Global escalation alerts ----
    st.markdown("### 🚨 Global Escalation Alerts")
    st.caption(f"Countries predicted to exceed risk {alert_thresh:.2f} in the next period")

    alerts = get_escalation_alerts(alert_thresh)
    if alerts:
        df_alerts = pd.DataFrame(alerts)
        # Add human-readable name column
        if "country" in df_alerts.columns:
            df_alerts.insert(
                0, "Country",
                df_alerts["country"].map(lambda c: name_map.get(c, c)),
            )
        cols_show = [c for c in ["Country", "predicted_risk", "current_risk",
                                  "delta", "confidence", "target_date"] if c in df_alerts.columns]
        if cols_show:
            st.dataframe(
                df_alerts[cols_show].sort_values(
                    "predicted_risk" if "predicted_risk" in df_alerts.columns else cols_show[0],
                    ascending=False,
                ).head(20),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No escalation alerts. Forecaster may not be trained yet or no countries exceed threshold.")


main()
