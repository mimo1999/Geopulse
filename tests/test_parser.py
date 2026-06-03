"""Tests for GDELT event parser."""

import pytest
from datetime import date
from ingestion.gdelt_parser import GDELTParser, GDELTEvent


@pytest.fixture
def parser():
    return GDELTParser()


def _make_row(**overrides) -> dict:
    """Build a minimal valid GDELT row dict."""
    base = {
        "global_event_id": "123456",
        "sqldate": "20240315",
        "actor1_code": "US",
        "actor1_name": "UNITED STATES",
        "actor1_country_code": "US",
        "actor1_type1_code": "GOV",
        "actor2_code": "RU",
        "actor2_name": "RUSSIA",
        "actor2_country_code": "RU",
        "actor2_type1_code": "GOV",
        "event_code": "14",
        "event_base_code": "14",
        "event_root_code": "14",
        "quad_class": "3",
        "goldstein_scale": "-5.0",
        "num_mentions": "10",
        "num_sources": "3",
        "num_articles": "8",
        "avg_tone": "-4.5",
        "action_geo_country_code": "US",
        "action_geo_lat": "38.89",
        "action_geo_long": "-77.03",
        "source_url": "http://example.com/article",
    }
    base.update(overrides)
    return base


def test_parse_valid_row(parser):
    row = _make_row()
    events, skipped = parser.parse_chunk([row])
    assert len(events) == 1
    assert skipped == 0
    e = events[0]
    assert e.global_event_id == 123456
    assert e.event_date == date(2024, 3, 15)
    assert e.actor1_country == "US"
    assert e.goldstein == pytest.approx(-5.0)
    assert e.quad_class == 3


def test_parse_missing_date(parser):
    row = _make_row(sqldate="")
    events, skipped = parser.parse_chunk([row])
    assert len(events) == 0
    assert skipped == 1


def test_parse_invalid_event_id(parser):
    row = _make_row(global_event_id="0")
    events, skipped = parser.parse_chunk([row])
    assert len(events) == 0
    assert skipped == 1


def test_goldstein_clamping(parser):
    # Goldstein out-of-range should be clamped
    row = _make_row(goldstein_scale="15.0")
    events, _ = parser.parse_chunk([row])
    assert events[0].goldstein == pytest.approx(10.0)

    row = _make_row(goldstein_scale="-15.0")
    events, _ = parser.parse_chunk([row])
    assert events[0].goldstein == pytest.approx(-10.0)


def test_empty_optional_fields(parser):
    row = _make_row(actor2_code="", avg_tone="", source_url="")
    events, skipped = parser.parse_chunk([row])
    assert len(events) == 1
    e = events[0]
    assert e.actor2_code is None
    assert e.avg_tone is None
    assert e.source_url is None


def test_to_db_dict(parser):
    row = _make_row()
    events, _ = parser.parse_chunk([row])
    d = parser.to_db_dict(events[0])
    assert "global_event_id" in d
    assert "event_date" in d
    assert isinstance(d["event_date"], date)


def test_parse_large_chunk(parser):
    rows = [_make_row(global_event_id=str(i)) for i in range(1, 1001)]
    events, skipped = parser.parse_chunk(rows)
    assert len(events) == 1000
    assert skipped == 0
