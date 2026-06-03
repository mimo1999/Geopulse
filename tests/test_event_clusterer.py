"""Tests for event clusterer classification logic."""

import pytest
import json
from datetime import date
from preprocessing.event_clusterer import EventClusterer


@pytest.fixture
def clusterer():
    return EventClusterer(dsn="postgresql://test:test@localhost/test")


def _event(root, base=None, quad=1, goldstein=0.0, tone=0.0, mentions=5,
           a1name="USA", a2name="RUS", a1c="US", a2c="RU"):
    return (root, base or root * 10, quad, goldstein, tone, mentions,
            a1name, a2name, a1c, a2c)


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

def test_terrorism_classification(clusterer):
    # Root 18 (assault), quad 4 → terrorism
    assert clusterer._classify_event(18, 180, 4) == "terrorism"
    assert clusterer._classify_event(20, 200, 4) == "terrorism"


def test_military_classification(clusterer):
    # Root 19 (Fight), quad 4 — not in base_codes terrorism, so military
    assert clusterer._classify_event(19, 190, 3) == "military"
    assert clusterer._classify_event(15, 150, 4) == "military"


def test_protest_classification(clusterer):
    assert clusterer._classify_event(14, 141, 3) == "protest"
    assert clusterer._classify_event(14, 145, 4) == "protest"


def test_sanctions_classification(clusterer):
    # Base code 163
    assert clusterer._classify_event(16, 163, 3) == "sanctions"
    assert clusterer._classify_event(17, 168, 4) == "sanctions"


def test_diplomatic_default(clusterer):
    assert clusterer._classify_event(1, 10, 1) == "diplomatic"
    assert clusterer._classify_event(2, 20, 2) == "diplomatic"
    assert clusterer._classify_event(None, None, None) == "diplomatic"


# ---------------------------------------------------------------------------
# Cluster building tests
# ---------------------------------------------------------------------------

def test_build_clusters_basic(clusterer):
    events = [
        _event(14, quad=3, goldstein=-2.0, tone=-15.0, mentions=20),
        _event(14, quad=4, goldstein=-5.0, tone=-25.0, mentions=50),
        _event(1, quad=1, goldstein=3.0,  tone=5.0,   mentions=5),
    ]
    clusters = clusterer._build_clusters("US", date(2024, 1, 1), events)
    cats = {c["category"] for c in clusters}
    assert "protest" in cats
    assert "diplomatic" in cats


def test_cluster_aggregation(clusterer):
    events = [
        _event(14, goldstein=-3.0, mentions=30),
        _event(14, goldstein=-5.0, mentions=20),
    ]
    clusters = clusterer._build_clusters("US", date(2024, 1, 1), events)
    protest_cluster = next((c for c in clusters if c["category"] == "protest"), None)
    assert protest_cluster is not None
    assert protest_cluster["event_count"] == 2
    assert protest_cluster["total_mentions"] == 50
    assert protest_cluster["avg_goldstein"] == pytest.approx(-4.0, abs=0.01)


def test_max_intensity(clusterer):
    events = [
        _event(19, quad=4, goldstein=-2.0),
        _event(19, quad=4, goldstein=-9.0),
        _event(19, quad=4, goldstein=-4.0),
    ]
    clusters = clusterer._build_clusters("SY", date(2024, 1, 1), events)
    mil_cluster = next((c for c in clusters if c["category"] == "military"), None)
    assert mil_cluster is not None
    assert mil_cluster["max_intensity"] == pytest.approx(9.0, abs=0.01)


def test_top_actor_pairs(clusterer):
    events = [
        _event(1, a1name="USA", a2name="RUS"),
        _event(1, a1name="USA", a2name="RUS"),
        _event(1, a1name="USA", a2name="CHN"),
    ]
    pairs = clusterer._top_actor_pairs(events, top_n=5)
    assert len(pairs) <= 5
    assert any(p["actor1"] == "USA" and p["actor2"] == "RUS" for p in pairs)
    # USA-RUS should have count=2
    usa_rus = next(p for p in pairs if p["actor1"] == "USA" and p["actor2"] == "RUS")
    assert usa_rus["count"] == 2


def test_top_actor_pairs_serializable(clusterer):
    events = [_event(1)]
    pairs = clusterer._top_actor_pairs(events)
    # Should be JSON-serializable
    json.dumps(pairs)


def test_empty_events(clusterer):
    clusters = clusterer._build_clusters("XX", date(2024, 1, 1), [])
    assert clusters == []
