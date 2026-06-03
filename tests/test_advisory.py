"""Tests for rule-based advisory engine."""

import pytest
from advisory.rule_engine import AdvisoryEngine, RiskAdvisory, classify_risk


def test_classify_risk_levels():
    assert classify_risk(0.90) == "CRITICAL"
    assert classify_risk(0.80) == "CRITICAL"
    assert classify_risk(0.70) == "HIGH"
    assert classify_risk(0.65) == "HIGH"
    assert classify_risk(0.55) == "ELEVATED"
    assert classify_risk(0.40) == "MODERATE"
    assert classify_risk(0.10) == "LOW"
    assert classify_risk(0.00) == "LOW"


@pytest.fixture
def engine():
    return AdvisoryEngine(ollama_enabled=False)


def test_generate_returns_advisory(engine):
    advisory = engine.generate(
        country="Pakistan",
        risk_score=0.78,
        confidence=0.84,
        trend="increasing",
        instability=0.80,
        war=0.75,
        terrorism=0.60,
        financial=0.40,
    )
    assert isinstance(advisory, RiskAdvisory)
    assert advisory.country == "Pakistan"
    assert advisory.risk_score == pytest.approx(0.78)
    assert advisory.level == "HIGH"
    assert advisory.trend == "increasing"
    assert len(advisory.major_drivers) > 0
    assert len(advisory.advisory_text) > 20


def test_critical_advisory_text(engine):
    advisory = engine.generate(
        country="TestCountry",
        risk_score=0.92,
        confidence=0.70,
        trend="increasing",
        instability=0.95,
        war=0.90,
    )
    assert advisory.level == "CRITICAL"
    assert "CRITICAL" in advisory.advisory_text.upper()


def test_low_risk_advisory(engine):
    advisory = engine.generate(
        country="Iceland",
        risk_score=0.08,
        confidence=0.90,
        trend="stable",
    )
    assert advisory.level == "LOW"
    assert "LOW" in advisory.advisory_text.upper()


def test_increasing_trend_appended(engine):
    advisory = engine.generate(
        country="Ukraine",
        risk_score=0.70,
        confidence=0.75,
        trend="increasing",
        war=0.80,
    )
    assert "increas" in advisory.advisory_text.lower()


def test_to_dict(engine):
    advisory = engine.generate(
        country="Germany",
        risk_score=0.25,
        confidence=0.85,
        trend="stable",
    )
    d = advisory.to_dict()
    assert "country" in d
    assert "risk_score" in d
    assert "major_drivers" in d
    assert isinstance(d["major_drivers"], list)


def test_war_sub_rule(engine):
    advisory = engine.generate(
        country="X",
        risk_score=0.72,
        confidence=0.70,
        trend="stable",
        war=0.75,
    )
    # War sub-rule at 0.70 should trigger
    assert any("military" in d.lower() or "escalation" in advisory.advisory_text.lower()
               for d in advisory.major_drivers + [advisory.advisory_text])


def test_drivers_max_three(engine):
    advisory = engine.generate(
        country="Y",
        risk_score=0.70,
        confidence=0.60,
        trend="stable",
        instability=0.80,
        war=0.75,
        terrorism=0.70,
        financial=0.65,
        feature_scores={
            "protest_score": 0.50,
            "violence_score": 0.55,
        },
    )
    assert len(advisory.major_drivers) <= 3
