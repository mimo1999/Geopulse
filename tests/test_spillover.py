"""Tests for spillover analyzer (unit tests, no DB)."""

import pytest
from datetime import date
from inference.spillover import SpilloverAnalyzer, ADJACENT_PAIRS


@pytest.fixture
def analyzer():
    return SpilloverAnalyzer(
        dsn="postgresql://test:test@localhost/test",
        window_days=30,
        min_overlap_days=5,
    )


# ---------------------------------------------------------------------------
# Spillover row builder (no DB)
# ---------------------------------------------------------------------------

def test_build_spillover_rows_basic(analyzer):
    corr = {("IN", "PK"): 0.85, ("US", "RU"): -0.70}
    cooc = {("IN", "PK"): 100,  ("US", "RU"): 200}
    rows = analyzer._build_spillover_rows(corr, cooc, date(2024, 1, 1))
    assert len(rows) == 2
    # Sorted by weight desc
    assert rows[0]["spillover_weight"] >= rows[1]["spillover_weight"]


def test_build_spillover_adjacency_bonus(analyzer):
    # IN-PK is adjacent
    corr = {("IN", "PK"): 0.5}
    cooc = {("IN", "PK"): 50}
    rows_adj = analyzer._build_spillover_rows(corr, cooc, date(2024, 1, 1))

    # Compare with non-adjacent pair with same scores
    corr2 = {("US", "ZW"): 0.5}
    cooc2 = {("US", "ZW"): 50}
    rows_non = analyzer._build_spillover_rows(corr2, cooc2, date(2024, 1, 1))

    # Adjacent pair should have higher spillover
    assert rows_adj[0]["spillover_weight"] > rows_non[0]["spillover_weight"]
    assert rows_adj[0]["is_adjacent"] is True
    assert rows_non[0]["is_adjacent"] is False


def test_build_spillover_country_ordering(analyzer):
    """country_a must always be lexicographically before country_b."""
    corr = {("ZZ", "AA"): 0.5}
    cooc = {}
    rows = analyzer._build_spillover_rows(corr, cooc, date(2024, 1, 1))
    assert len(rows) == 1
    # Expect this pair was passed with a > b — check stored canonically
    # (the _build function should handle this from the corr/cooc dicts)
    row = rows[0]
    assert row["country_a"] <= row["country_b"]


def test_build_spillover_weight_range(analyzer):
    corr = {("CN", "US"): -0.9, ("RU", "UA"): 0.95, ("FR", "DE"): 0.1}
    cooc = {("CN", "US"): 500, ("RU", "UA"): 1000, ("FR", "DE"): 10}
    rows = analyzer._build_spillover_rows(corr, cooc, date(2024, 1, 1))
    for row in rows:
        assert 0.0 <= row["spillover_weight"] <= 1.0, \
            f"Weight out of range: {row['spillover_weight']}"


def test_empty_inputs(analyzer):
    rows = analyzer._build_spillover_rows({}, {}, date(2024, 1, 1))
    assert rows == []


def test_top_n_limit(analyzer):
    """Only top_n_pairs should be returned."""
    analyzer._top_n = 3
    corr = {(f"A{i}", f"B{i}"): 0.5 - i * 0.05 for i in range(10)}
    rows = analyzer._build_spillover_rows(corr, {}, date(2024, 1, 1))
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# Adjacent pairs sanity check
# ---------------------------------------------------------------------------

def test_adjacent_pairs_symmetric():
    """All pairs in ADJACENT_PAIRS use frozenset so order doesn't matter."""
    for pair in ADJACENT_PAIRS:
        assert len(pair) == 2, f"Pair should have 2 elements: {pair}"


def test_india_pakistan_adjacent():
    assert frozenset({"IN", "PK"}) in ADJACENT_PAIRS


def test_russia_ukraine_adjacent():
    assert frozenset({"RU", "UA"}) in ADJACENT_PAIRS
