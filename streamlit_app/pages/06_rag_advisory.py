"""
Phase 3 — RAG Advisory Page.

Retrieval-Augmented advisory generation for any country.
Shows enriched advisory text plus the retrieved analogues that informed it.
Optional side-by-side comparison with the rule-based advisory.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Advisory — GeoPulse",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #0d0d0d; color: #e0e0e0; }
    h1, h2, h3 { font-family: 'Courier New', monospace; color: #cc2222; }
    .advisory-box {
        background: #0f0f0f;
        border: 1px solid #2a2a2a;
        border-left: 3px solid #cc2222;
        border-radius: 6px;
        padding: 18px 22px;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.8;
        color: #ddd;
        white-space: pre-wrap;
        margin: 8px 0;
    }
    .rule-box {
        background: #0f0f0f;
        border: 1px solid #1a1a1a;
        border-left: 3px solid #444;
        border-radius: 6px;
        padding: 18px 22px;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.8;
        color: #888;
        white-space: pre-wrap;
        margin: 8px 0;
    }
    .context-card {
        background: #141414;
        border: 1px solid #222;
        border-radius: 5px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 12px;
    }
    .similarity-badge {
        display: inline-block; padding: 2px 8px;
        background: #1a1a3a; border-radius: 3px;
        color: #6666cc; font-size: 11px; font-family: Courier New;
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
def get_rag_advisory(country: str, include_retrieved: bool = True) -> dict:
    try:
        r = requests.get(
            f"{BACKEND_URL}/country/{country}/rag_advisory",
            params={"include_retrieved": include_retrieved},
            timeout=20,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300)
def get_rule_advisory(country: str) -> dict:
    try:
        r = requests.post(f"{BACKEND_URL}/riskscore", json={"country": country}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=600)
def get_corpus_stats() -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}/advisory/corpus/stats", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main():
    st.markdown(
        "<h1>🔬 RAG Advisory Intelligence</h1>"
        "<p style='color:#555; font-family:Courier New; font-size:13px;'>"
        "Retrieval-Augmented advisory generation with historical analogue context</p>",
        unsafe_allow_html=True,
    )

    # ---- Sidebar controls ----
    with st.sidebar:
        st.markdown("#### Controls")
        countries, name_map = get_countries()
        country   = st.selectbox(
            "Country", countries or ["No data"],
            format_func=lambda x: name_map.get(x, x),
            key="rag_country",
        )

        show_comparison = st.toggle("Compare with Rule-Based Advisory", value=True)
        include_forecast = st.toggle("Include Forecast Context", value=True)

        st.divider()
        st.markdown("#### Corpus Management")
        stats = get_corpus_stats()
        if stats:
            st.metric("Seed Entries", stats.get("seed_entries", "N/A"))
            st.metric("DB Entries",   stats.get("db_entries",   "N/A"))

        if st.button("🔄 Rebuild Corpus", use_container_width=True):
            try:
                r = requests.post(
                    f"{BACKEND_URL}/advisory/corpus/rebuild",
                    json={}, timeout=15,
                )
                if r.ok:
                    st.success(f"Corpus rebuilt: {r.json().get('corpus_size', '?')} entries")
                else:
                    st.error(f"Error {r.status_code}")
            except Exception as e:
                st.error(str(e))
            st.cache_data.clear()

        st.divider()
        st.markdown(
            "<p style='font-size:11px; color:#444; font-family:Courier New;'>"
            "RAG uses TF-IDF cosine retrieval over ~30 seed situations + "
            "auto-generated event cluster entries. Ollama optional for "
            "narrative polishing."
            "</p>",
            unsafe_allow_html=True,
        )

    if not country or country == "No data":
        return

    display_name = name_map.get(country, country)
    st.markdown(f"<h3 style='color:#888; font-family:Courier New;'>{display_name}</h3>", unsafe_allow_html=True)

    # ---- Fetch advisories ----
    rag_data  = get_rag_advisory(country, include_retrieved=True)
    rule_data = get_rule_advisory(country) if show_comparison else {}

    # ---- Risk header ----
    if rule_data and "risk_score" in rule_data:
        score = rule_data.get("risk_score", 0)
        level = rule_data.get("level", "UNKNOWN")
        conf  = rule_data.get("confidence", 0)
        trend = rule_data.get("trend", "stable")
        trend_arrow = {"increasing": "↑", "stable": "→", "decreasing": "↓"}.get(trend, "")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk Score",  f"{score:.3f}")
        c2.metric("Level",       level)
        c3.metric("Confidence",  f"{conf:.0%}")
        c4.metric("Trend",       f"{trend_arrow} {trend}")

    st.divider()

    # ---- Advisory panels ----
    if show_comparison:
        col_rag, col_rule = st.columns(2)

        with col_rag:
            st.markdown("#### RAG-Enhanced Advisory")
            if "error" in rag_data:
                st.warning(f"RAG unavailable: {rag_data['error']}")
                st.info("The RAG engine requires the backend to be running with Phase 3 routes.")
            else:
                advisory_text = rag_data.get("advisory", "No advisory generated.")
                rag_conf      = rag_data.get("rag_confidence", 0)
                st.markdown(
                    f'<div class="advisory-box">{advisory_text}</div>',
                    unsafe_allow_html=True,
                )
                if rag_conf:
                    st.caption(f"RAG confidence: {rag_conf:.2f}")

        with col_rule:
            st.markdown("#### Rule-Based Advisory (baseline)")
            if rule_data and "advisory" in rule_data:
                st.markdown(
                    f'<div class="rule-box">{rule_data["advisory"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("Rule-based advisory unavailable.")
    else:
        st.markdown("#### Advisory")
        if "error" in rag_data:
            st.warning(f"RAG unavailable: {rag_data['error']}")
        else:
            advisory_text = rag_data.get("advisory", "No advisory generated.")
            st.markdown(
                f'<div class="advisory-box">{advisory_text}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ---- Retrieved contexts ----
    retrieved = rag_data.get("retrieved_contexts", []) if "error" not in rag_data else []
    if retrieved:
        st.markdown("#### Retrieved Analogues")
        st.caption(f"Top-{len(retrieved)} situations retrieved by TF-IDF cosine similarity")

        for ctx in retrieved:
            sim      = ctx.get("similarity", 0)
            sit_type = ctx.get("situation_type", "").replace("_", " ").title()
            rl       = ctx.get("risk_level", "")
            text     = ctx.get("text", "")
            tags     = ctx.get("tags", [])

            badge_color = {
                "CRITICAL": "#cc0000", "HIGH": "#8b0000",
                "ELEVATED": "#6b3300", "MODERATE": "#4a4a00", "LOW": "#1a3a1a",
            }.get(rl, "#333")

            st.markdown(
                f"""
                <div class="context-card">
                    <span class="similarity-badge">sim={sim:.3f}</span>
                    &nbsp;
                    <span style="color:{badge_color}; font-weight:bold; font-size:11px;">{rl}</span>
                    &nbsp;
                    <span style="color:#666; font-size:11px;">{sit_type}</span>
                    <br/><br/>
                    <span style="color:#bbb;">{text}</span>
                    <br/>
                    <span style="color:#444; font-size:10px;">{' · '.join(tags[:6])}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    elif "error" not in rag_data:
        st.info(
            "No analogues retrieved. The TF-IDF corpus may need rebuilding — "
            "click 'Rebuild Corpus' in the sidebar."
        )

    # ---- Major drivers ----
    if rule_data and "major_drivers" in rule_data:
        with st.expander("Key Risk Drivers"):
            for d in rule_data["major_drivers"]:
                st.markdown(f"• {d}")


main()
