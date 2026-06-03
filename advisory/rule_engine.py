"""
Rule-based advisory generation engine.

Produces structured risk advisories from model scores.
No LLM required — template-driven, fast, deterministic.

Optionally falls back to Ollama for narrative generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("advisory.engine")


# ---------------------------------------------------------------------------
# Risk level classification
# ---------------------------------------------------------------------------

RISK_LEVELS = {
    "CRITICAL":  (0.80, 1.00),
    "HIGH":      (0.65, 0.80),
    "ELEVATED":  (0.50, 0.65),
    "MODERATE":  (0.35, 0.50),
    "LOW":       (0.00, 0.35),
}


def classify_risk(score: float) -> str:
    for level, (lo, hi) in RISK_LEVELS.items():
        if lo <= score < hi:
            return level
    return "CRITICAL" if score >= 1.0 else "LOW"


# ---------------------------------------------------------------------------
# Advisory output
# ---------------------------------------------------------------------------

@dataclass
class RiskAdvisory:
    country: str
    risk_score: float
    confidence: float
    level: str
    trend: str                         # "increasing" | "stable" | "decreasing"
    major_drivers: list[str] = field(default_factory=list)
    advisory_text: str = ""
    short_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "country":       self.country,
            "risk_score":    round(self.risk_score, 4),
            "confidence":    round(self.confidence, 4),
            "level":         self.level,
            "trend":         self.trend,
            "major_drivers": self.major_drivers,
            "advisory":      self.advisory_text,
            "summary":       self.short_summary,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AdvisoryEngine:
    """
    Generates structured risk advisories from model output.

    Phase 1: entirely rule-based.
    Phase 2: optional Ollama narrative generation.
    """

    # --- Advisory templates per level ---
    _TEMPLATES: dict[str, str] = {
        "CRITICAL": (
            "CRITICAL RISK — {country} presents maximum instability indicators "
            "over the monitoring period. Significant deterioration is assessed "
            "with high confidence. Immediate elevated monitoring warranted."
        ),
        "HIGH": (
            "HIGH RISK — {country} shows strong escalation indicators across "
            "multiple dimensions. Conditions may deteriorate further within "
            "the 30-day forecast horizon."
        ),
        "ELEVATED": (
            "ELEVATED RISK — {country} exhibits above-threshold instability "
            "signals. Monitor closely for further deterioration."
        ),
        "MODERATE": (
            "MODERATE RISK — {country} shows some instability indicators but "
            "remains below high-risk thresholds. Routine monitoring recommended."
        ),
        "LOW": (
            "LOW RISK — {country} currently presents minimal instability "
            "indicators. Continue standard monitoring cadence."
        ),
    }

    # --- Sub-advisories triggered by specific scores ---
    _SUB_RULES: list[tuple[str, float, str]] = [
        # (driver_key, threshold, message)
        ("war",         0.70, "Elevated military escalation risk detected. "
                              "Cross-border incidents or mobilization events observed."),
        ("terrorism",   0.65, "Elevated terrorism and asymmetric conflict indicators."),
        ("instability", 0.75, "High internal political instability. "
                              "Civil unrest or governance disruption likely."),
        ("financial",   0.60, "Economic stress indicators elevated. "
                              "Sanctions, capital flight, or fiscal strain detected."),
        ("war",         0.50, "Low-level military activity noted."),
        ("terrorism",   0.45, "Security incidents trending above baseline."),
    ]

    # --- Driver labels (what the model heads map to) ---
    _DRIVER_LABELS = {
        "instability": "Political instability",
        "war":         "Military / armed conflict risk",
        "terrorism":   "Terrorism / asymmetric threat",
        "financial":   "Economic / financial stress",
    }

    # --- Feature-to-driver mapping ---
    _FEATURE_DRIVERS: dict[str, tuple[float, str]] = {
        "protest_score":     (0.30, "Civil protests and demonstrations"),
        "violence_score":    (0.30, "Material conflict events"),
        "diplomatic_stress": (0.40, "Diplomatic tensions"),
        "economic_stress":   (0.30, "Economic sanctions or shocks"),
        "terrorism_score":   (0.25, "Terrorism-related incidents"),
    }

    def __init__(self, ollama_enabled: bool = False, ollama_url: str = "http://localhost:11434"):
        self._ollama_enabled = ollama_enabled
        self._ollama_url = ollama_url

    def generate(
        self,
        country: str,
        risk_score: float,
        confidence: float,
        trend: str,
        instability: float = 0.0,
        war: float = 0.0,
        terrorism: float = 0.0,
        financial: float = 0.0,
        feature_scores: Optional[dict[str, float]] = None,
    ) -> RiskAdvisory:
        """
        Generate a complete risk advisory.

        Args:
            country:        ISO country name or code.
            risk_score:     Composite 0–1 score.
            confidence:     Model confidence 0–1.
            trend:          "increasing" | "stable" | "decreasing"
            instability:    Instability head output.
            war:            War probability head output.
            terrorism:      Terrorism risk head output.
            financial:      Financial stress head output.
            feature_scores: Raw feature scores for driver identification.

        Returns:
            RiskAdvisory with text and structured fields.
        """
        level = classify_risk(risk_score)
        major_drivers = self._identify_drivers(
            instability=instability,
            war=war,
            terrorism=terrorism,
            financial=financial,
            feature_scores=feature_scores or {},
        )

        advisory_text = self._build_advisory(
            country=country,
            level=level,
            trend=trend,
            instability=instability,
            war=war,
            terrorism=terrorism,
            financial=financial,
        )

        short_summary = self._short_summary(country, level, trend, risk_score)

        return RiskAdvisory(
            country=country,
            risk_score=risk_score,
            confidence=confidence,
            level=level,
            trend=trend,
            major_drivers=major_drivers,
            advisory_text=advisory_text,
            short_summary=short_summary,
        )

    def _identify_drivers(
        self,
        instability: float,
        war: float,
        terrorism: float,
        financial: float,
        feature_scores: dict[str, float],
    ) -> list[str]:
        """Return top-3 risk drivers from model heads + features."""
        candidates: list[tuple[float, str]] = []

        for head_key, score in [
            ("instability", instability),
            ("war", war),
            ("terrorism", terrorism),
            ("financial", financial),
        ]:
            if score >= 0.40:
                candidates.append((score, self._DRIVER_LABELS[head_key]))

        for feat, (threshold, label) in self._FEATURE_DRIVERS.items():
            val = feature_scores.get(feat, 0.0)
            if val >= threshold:
                candidates.append((val, label))

        # Sort descending by score, deduplicate labels, return top 3
        seen = set()
        drivers = []
        for score, label in sorted(candidates, reverse=True):
            if label not in seen:
                seen.add(label)
                drivers.append(label)
            if len(drivers) >= 3:
                break

        return drivers or ["General instability indicators"]

    def _build_advisory(
        self,
        country: str,
        level: str,
        trend: str,
        instability: float,
        war: float,
        terrorism: float,
        financial: float,
    ) -> str:
        lines = [self._TEMPLATES[level].format(country=country)]

        # Trend modifier
        if trend == "increasing":
            lines.append("Risk trajectory is increasing — conditions may worsen "
                         "within the 30-day forecast window.")
        elif trend == "decreasing":
            lines.append("Risk trajectory is improving — monitor for consolidation.")

        # Sub-rule modifiers (apply first matching rule per category)
        triggered: set[str] = set()
        for driver_key, threshold, message in self._SUB_RULES:
            scores = {"war": war, "terrorism": terrorism,
                      "instability": instability, "financial": financial}
            if driver_key not in triggered and scores[driver_key] >= threshold:
                lines.append(message)
                triggered.add(driver_key)

        return " ".join(lines)

    def _short_summary(
        self,
        country: str,
        level: str,
        trend: str,
        risk_score: float,
    ) -> str:
        trend_word = {"increasing": "↑", "stable": "→", "decreasing": "↓"}.get(trend, "")
        return f"{country}: {level} ({risk_score:.2f}) {trend_word}"

    # ------------------------------------------------------------------
    # Optional Ollama narrative (Phase 2)
    # ------------------------------------------------------------------

    def generate_narrative(
        self,
        advisory: RiskAdvisory,
        recent_events: list[str] | None = None,
    ) -> str:
        """
        Generate an analyst-style narrative using Ollama.
        Only called when ollama_enabled=True.
        """
        if not self._ollama_enabled:
            return advisory.advisory_text

        try:
            import requests
            prompt = self._build_ollama_prompt(advisory, recent_events or [])
            resp = requests.post(
                f"{self._ollama_url}/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("response", advisory.advisory_text).strip()
        except Exception as exc:
            logger.warning("Ollama narrative generation failed: %s", exc)
            return advisory.advisory_text

    def _build_ollama_prompt(
        self,
        advisory: RiskAdvisory,
        events: list[str],
    ) -> str:
        event_text = "\n".join(f"- {e}" for e in events[:5]) or "No recent events available."
        return (
            f"You are a geopolitical risk analyst. "
            f"Write a concise 2-3 sentence risk summary for {advisory.country}.\n\n"
            f"Risk level: {advisory.level} ({advisory.risk_score:.2f})\n"
            f"Trend: {advisory.trend}\n"
            f"Key drivers: {', '.join(advisory.major_drivers)}\n\n"
            f"Recent events:\n{event_text}\n\n"
            f"Advisory (factual, professional, no speculation):"
        )
