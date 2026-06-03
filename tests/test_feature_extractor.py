"""Tests for feature extractor logic (unit, no DB required)."""

import pytest
from datetime import date
from preprocessing.feature_extractor import FeatureExtractor


@pytest.fixture
def extractor(tmp_path):
    # Use a dummy DSN — unit tests only call _compute_features directly
    return FeatureExtractor(dsn="postgresql://test:test@localhost:5432/test")


def _make_events(rows: list[tuple]) -> list[tuple]:
    """rows: (event_root_code, quad_class, goldstein, avg_tone, mentions, sources)"""
    return rows


def test_all_cooperation_events(extractor):
    events = [
        (1, 2, 5.0, 10.0, 5, 2),
        (2, 1, 4.0, 8.0, 3, 1),
        (3, 2, 3.0, 7.0, 2, 1),
    ]
    feats = extractor._compute_features("US", date(2024, 1, 1), events)
    assert feats["violence_score"] == pytest.approx(0.0)
    assert feats["avg_goldstein"] > 0.5   # positive goldstein → above midpoint


def test_all_conflict_events(extractor):
    events = [
        (14, 4, -5.0, -20.0, 10, 3),
        (19, 4, -8.0, -30.0, 15, 5),
        (20, 4, -9.0, -40.0, 20, 8),
    ]
    feats = extractor._compute_features("SY", date(2024, 1, 1), events)
    assert feats["violence_score"] == pytest.approx(1.0)
    assert feats["protest_score"] == pytest.approx(1/3, abs=0.01)
    assert feats["diplomatic_stress"] > 0.5


def test_terrorism_detection(extractor):
    events = [
        (18, 4, -8.0, -50.0, 30, 10),  # assault + quad 4
        (19, 4, -7.0, -40.0, 20, 8),
        (1,  1, 3.0,  5.0, 2, 1),
    ]
    feats = extractor._compute_features("PK", date(2024, 1, 1), events)
    assert feats["terrorism_score"] > 0.0


def test_goldstein_normalization(extractor):
    # Pure goldstein = 0.0 → should be at midpoint (0.5)
    events = [(1, 1, 0.0, 0.0, 1, 1)]
    feats = extractor._compute_features("X", date(2024, 1, 1), events)
    assert feats["avg_goldstein"] == pytest.approx(0.5, abs=0.01)


def test_sentiment_normalization(extractor):
    # avg_tone = 0 → ~0.5 when normalized to -100..+100
    events = [(1, 1, 0.0, 0.0, 1, 1)]
    feats = extractor._compute_features("X", date(2024, 1, 1), events)
    assert feats["avg_sentiment"] == pytest.approx(0.5, abs=0.01)


def test_feature_keys(extractor):
    events = [(1, 1, 0.0, 0.0, 1, 1)]
    feats = extractor._compute_features("US", date(2024, 1, 1), events)
    expected_keys = {
        "country", "feature_date", "total_events",
        "conflict_events", "cooperation_events",
        "protest_score", "violence_score", "diplomatic_stress",
        "economic_stress", "terrorism_score",
        "avg_sentiment", "avg_goldstein",
    }
    assert expected_keys.issubset(set(feats.keys()))


def test_scores_clamped_to_unit_interval(extractor):
    events = [(20, 4, -10.0, -100.0, 100, 50)] * 100
    feats = extractor._compute_features("X", date(2024, 1, 1), events)
    for key in ("protest_score", "violence_score", "diplomatic_stress",
                "economic_stress", "terrorism_score", "avg_sentiment", "avg_goldstein"):
        assert 0.0 <= feats[key] <= 1.0, f"{key} out of range: {feats[key]}"
