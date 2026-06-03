"""Tests for event deduplication and cleaning."""

import pytest
from datetime import date
from ingestion.event_cleaner import EventCleaner, BloomFilter


@pytest.fixture
def cleaner():
    return EventCleaner(min_date_str="2020-01-01")


def _make_event(event_id: int = 1, **overrides) -> dict:
    base = {
        "global_event_id": event_id,
        "event_date": date(2024, 3, 15),
        "action_geo_country": "US",
        "actor1_country": "US",
        "actor2_country": "RU",
        "latitude": 38.89,
        "longitude": -77.03,
        "source_url": "http://example.com",
    }
    base.update(overrides)
    return base


# BloomFilter tests

def test_bloom_add_contains():
    bf = BloomFilter(expected_items=1000)
    bf.add(42)
    assert 42 in bf
    assert 43 not in bf


def test_bloom_no_false_negatives():
    bf = BloomFilter(expected_items=10000)
    ids = list(range(0, 5000))
    for i in ids:
        bf.add(i)
    for i in ids:
        assert i in bf, f"{i} should be in bloom filter"


# Cleaner tests

def test_dedup_same_batch(cleaner):
    rows = [_make_event(1), _make_event(1)]   # duplicate
    clean, stats = cleaner.clean_batch(rows)
    assert len(clean) == 1
    assert stats.duplicates_removed == 1


def test_dedup_across_batches(cleaner):
    rows1 = [_make_event(1)]
    rows2 = [_make_event(1)]   # same ID in second batch
    clean1, _ = cleaner.clean_batch(rows1)
    clean2, stats2 = cleaner.clean_batch(rows2)
    assert len(clean1) == 1
    assert len(clean2) == 0
    assert stats2.duplicates_removed == 1


def test_filter_old_date(cleaner):
    old_row = _make_event(99, event_date=date(2019, 1, 1))
    clean, stats = cleaner.clean_batch([old_row])
    assert len(clean) == 0
    assert stats.invalid_removed == 1


def test_filter_no_country(cleaner):
    row = _make_event(10, action_geo_country=None)
    clean, stats = cleaner.clean_batch([row])
    assert len(clean) == 0
    assert stats.invalid_removed == 1


def test_country_normalization(cleaner):
    # FIPS RS → ISO RU
    row = _make_event(5, actor1_country="RS", action_geo_country="RS")
    clean, _ = cleaner.clean_batch([row])
    assert clean[0]["actor1_country"] == "RU"
    assert clean[0]["action_geo_country"] == "RU"


def test_coordinate_clamping(cleaner):
    row = _make_event(7, latitude=200.0, longitude=-200.0)
    clean, _ = cleaner.clean_batch([row])
    assert clean[0]["latitude"] is None
    assert clean[0]["longitude"] is None


def test_url_truncation(cleaner):
    long_url = "http://example.com/" + "x" * 3000
    row = _make_event(8, source_url=long_url)
    clean, _ = cleaner.clean_batch([row])
    assert len(clean[0]["source_url"]) == 2048


def test_cumulative_stats(cleaner):
    batch1 = [_make_event(i) for i in range(10)]
    batch2 = [_make_event(i) for i in range(5, 15)]   # 5 overlap
    cleaner.clean_batch(batch1)
    cleaner.clean_batch(batch2)
    stats = cleaner.cumulative_stats
    assert stats.total_in == 20
    assert stats.duplicates_removed == 5
