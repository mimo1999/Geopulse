"""Tests for Phase 2 proxy label generator."""

import pytest
from datetime import date
from preprocessing.label_generator import LabelGenerator


@pytest.fixture
def gen():
    return LabelGenerator(dsn="postgresql://test:test@localhost/test")


# We test only the internal signal methods (no DB)

def _make_events(*rows) -> list[tuple]:
    """rows: (root_code, base_code, quad, goldstein, tone, mentions, a1c, a2c, date)"""
    return list(rows)


def _e(root, base=None, quad=1, goldstein=0.0, tone=0.0, mentions=5, a1c="US", a2c="US", d=None):
    return (root, base or root * 10, quad, goldstein, tone, mentions, a1c, a2c, d or date(2024,1,1))


# --------------- instability ---------------

def test_instability_all_cooperation(gen):
    events = [_e(1, quad=1), _e(2, quad=2), _e(3, quad=1)]
    score = gen._instability_signal(events, len(events))
    assert score < 0.5, f"Cooperation events should yield low instability, got {score}"


def test_instability_all_conflict(gen):
    events = [_e(14, quad=4, goldstein=-8.0), _e(19, quad=4), _e(20, quad=4)]
    score = gen._instability_signal(events, len(events))
    assert score > 0.5, f"Conflict events should yield high instability, got {score}"


def test_instability_range(gen):
    events = [_e(i, quad=3, goldstein=-3.0) for i in range(1, 21)]
    score = gen._instability_signal(events, len(events))
    assert 0.0 <= score <= 1.0


# --------------- war ---------------

def test_war_peaceful(gen):
    events = [_e(1, quad=1), _e(2, quad=2)]
    score = gen._war_signal(events, [], len(events))
    assert score < 0.3


def test_war_combat_events(gen):
    # Root code 19 (Fight) and 20 (Mass violence), quad 4, cross-border
    events = [_e(19, quad=4, a1c="RU", a2c="UA") for _ in range(10)]
    events += [_e(20, quad=4, a1c="RU", a2c="UA") for _ in range(5)]
    score = gen._war_signal(events, [], len(events))
    assert score > 0.5, f"Fight events should yield high war signal, got {score}"


def test_war_escalation_context(gen):
    events = [_e(19, quad=4) for _ in range(5)]
    context = [_e(19, quad=4) for _ in range(10)]   # prior context has fights too
    score_with    = gen._war_signal(events, context, len(events))
    score_without = gen._war_signal(events, [], len(events))
    assert score_with >= score_without, "Escalation context should amplify war signal"


# --------------- terrorism ---------------

def test_terrorism_no_events(gen):
    events = [_e(1, quad=1)]
    score = gen._terrorism_signal(events, len(events))
    assert score < 0.2


def test_terrorism_bombing_events(gen):
    # base code 183 = conduct suicide/car bombing
    events = [_e(18, base=183, quad=4, mentions=50) for _ in range(5)]
    score = gen._terrorism_signal(events, len(events))
    assert score > 0.5


# --------------- financial ---------------

def test_financial_no_sanctions(gen):
    events = [_e(1, quad=1)]
    score = gen._financial_signal(events, len(events))
    assert score < 0.2


def test_financial_sanctions(gen):
    # base code 163 = impose sanctions
    events = [_e(16, base=163, quad=3) for _ in range(8)]
    score = gen._financial_signal(events, len(events))
    assert score > 0.3


# --------------- sigmoid ---------------

def test_sigmoid_zero():
    assert LabelGenerator._sigmoid(0.0) == pytest.approx(0.5, abs=0.01)


def test_sigmoid_large_positive():
    assert LabelGenerator._sigmoid(10.0) > 0.99


def test_sigmoid_large_negative():
    assert LabelGenerator._sigmoid(-10.0) < 0.01
