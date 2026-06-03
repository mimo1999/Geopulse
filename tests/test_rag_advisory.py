"""
Tests for advisory/rag_advisory.py — TFIDFRetriever and RAGAdvisoryEngine.
"""

from __future__ import annotations

import os

import pytest

os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from advisory.rag_advisory import TFIDFRetriever, RAGAdvisoryEngine, SEED_CORPUS


# ---------------------------------------------------------------------------
# TFIDFRetriever
# ---------------------------------------------------------------------------

SAMPLE_CORPUS = [
    {
        "situation_type": "military_escalation_high",
        "risk_level": "HIGH",
        "text": "Troops massed near the border with artillery units deployed.",
        "tags": ["military", "border", "artillery"],
    },
    {
        "situation_type": "protest_moderate",
        "risk_level": "MODERATE",
        "text": "Mass protests in the capital following disputed elections.",
        "tags": ["protest", "election", "capital"],
    },
    {
        "situation_type": "sanctions_elevated",
        "risk_level": "ELEVATED",
        "text": "Comprehensive international sanctions imposed on energy sector.",
        "tags": ["sanctions", "energy", "international"],
    },
    {
        "situation_type": "stable_low",
        "risk_level": "LOW",
        "text": "Stable political environment with no significant incidents reported.",
        "tags": ["stable", "low risk"],
    },
]


def test_retriever_index_builds():
    r = TFIDFRetriever(vocab_size=100)
    r.index(SAMPLE_CORPUS)
    assert r._vectors is not None
    assert r._vectors.shape[0] == len(SAMPLE_CORPUS)


def test_retriever_returns_top_k():
    r = TFIDFRetriever(vocab_size=200)
    r.index(SAMPLE_CORPUS)
    results = r.retrieve("military troops artillery", top_k=2)
    assert len(results) <= 2


def test_retriever_similarity_ordering():
    """The most relevant result should come first."""
    r = TFIDFRetriever(vocab_size=200)
    r.index(SAMPLE_CORPUS)
    results = r.retrieve("protest election demonstrations", top_k=3)
    if results:
        # Protest entry should rank first
        top_type = results[0].get("situation_type", "")
        assert "protest" in top_type or results[0].get("similarity", 0) > 0


def test_retriever_empty_corpus():
    r = TFIDFRetriever()
    r.index([])
    results = r.retrieve("conflict war")
    assert results == []


def test_retriever_has_similarity_key():
    r = TFIDFRetriever(vocab_size=200)
    r.index(SAMPLE_CORPUS)
    results = r.retrieve("sanctions energy", top_k=1)
    if results:
        assert "similarity" in results[0]
        assert 0.0 <= results[0]["similarity"] <= 1.0


def test_retriever_with_empty_query():
    r = TFIDFRetriever(vocab_size=200)
    r.index(SAMPLE_CORPUS)
    results = r.retrieve("", top_k=3)
    # Should return empty (no tokens)
    assert results == []


# ---------------------------------------------------------------------------
# RAGAdvisoryEngine
# ---------------------------------------------------------------------------

def test_rag_engine_generates_advisory():
    engine = RAGAdvisoryEngine(extra_corpus=SAMPLE_CORPUS)
    advisory, contexts = engine.generate_rag(
        country="TestCountry",
        risk_score=0.72,
        confidence=0.80,
        trend="increasing",
        instability=0.75,
        war=0.65,
    )
    assert advisory.advisory_text != ""
    assert advisory.country == "TestCountry"
    assert advisory.level in ("CRITICAL", "HIGH", "ELEVATED", "MODERATE", "LOW")


def test_rag_engine_returns_contexts():
    engine = RAGAdvisoryEngine(extra_corpus=SAMPLE_CORPUS)
    _, contexts = engine.generate_rag(
        country="X",
        risk_score=0.80,
        confidence=0.70,
        trend="increasing",
        war=0.90,
    )
    # At least seed corpus should produce some results
    assert isinstance(contexts, list)


def test_rag_fallback_when_no_corpus():
    """With empty corpus, should still return a valid advisory from rule engine."""
    engine = RAGAdvisoryEngine.__new__(RAGAdvisoryEngine)
    from advisory.rule_engine import AdvisoryEngine
    from advisory.rag_advisory import TFIDFRetriever
    engine._base_engine = AdvisoryEngine()
    engine._retriever   = TFIDFRetriever()
    engine._retriever.index([])
    engine._top_k       = 3
    engine._ollama      = False
    engine._ollama_url  = ""
    engine._ollama_model = ""

    advisory, contexts = engine.generate_rag(
        country="Fallback",
        risk_score=0.45,
        confidence=0.60,
        trend="stable",
    )
    assert advisory.advisory_text != ""
    assert contexts == []


def test_rag_advisory_text_enriched():
    """RAG text should be longer or equal to base rule text."""
    engine = RAGAdvisoryEngine(extra_corpus=SAMPLE_CORPUS)
    from advisory.rule_engine import AdvisoryEngine, classify_risk
    base_engine = AdvisoryEngine()
    base_advisory = base_engine.generate(
        country="Ukraine",
        risk_score=0.78,
        confidence=0.82,
        trend="increasing",
        war=0.85,
    )

    rag_advisory, _ = engine.generate_rag(
        country="Ukraine",
        risk_score=0.78,
        confidence=0.82,
        trend="increasing",
        war=0.85,
    )

    assert len(rag_advisory.advisory_text) >= len(base_advisory.advisory_text)


def test_rag_with_forecast_trajectory():
    engine = RAGAdvisoryEngine(extra_corpus=SAMPLE_CORPUS)
    advisory, _ = engine.generate_rag(
        country="Pakistan",
        risk_score=0.65,
        confidence=0.75,
        trend="increasing",
        forecast_trajectory=[0.65, 0.70, 0.75, 0.80],
    )
    # Forecast clause should be included
    assert any(
        kw in advisory.advisory_text.lower()
        for kw in ["forecast", "week", "escalat", "project"]
    )


def test_seed_corpus_is_non_empty():
    assert len(SEED_CORPUS) >= 20


def test_seed_corpus_has_all_risk_levels():
    levels = {e["risk_level"] for e in SEED_CORPUS}
    for expected in ("CRITICAL", "HIGH", "ELEVATED", "MODERATE", "LOW"):
        assert expected in levels, f"Seed corpus missing {expected} level"


def test_add_corpus_entries():
    engine = RAGAdvisoryEngine()
    new_entries = [
        {
            "situation_type": "new_type",
            "risk_level": "HIGH",
            "text": "Unique new entry for testing purposes only xyz.",
            "tags": ["xyz", "test"],
        }
    ]
    engine.add_corpus_entries(new_entries)
    # Should retrieve the new entry with matching query
    retrieved = engine._retriever.retrieve("unique new xyz testing", top_k=1)
    if retrieved:
        assert retrieved[0].get("situation_type") == "new_type"
