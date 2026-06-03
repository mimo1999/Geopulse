"""
Phase 3: RAG Advisory Engine.

Retrieval-Augmented Generation for geopolitical risk advisories.

Pipeline:
    1. Query  : build a feature vector from current country scores
    2. Retrieve: cosine similarity over in-memory TF-IDF corpus
    3. Augment : inject retrieved context into advisory template
    4. Generate: produce enriched advisory text (optionally via Ollama)

Operates without any external LLM:
    - TF-IDF vectoriser built from fixed vocabulary at init time
    - Retrieval via numpy cosine similarity (no faiss required)
    - Output: richer template text that cites retrieved analogues

With Ollama:
    - Retrieved snippets + current scores → LLM prompt
    - Falls back to template if Ollama unavailable

Corpus entries have the shape:
    { situation_type, risk_level, text, tags: list[str] }

The corpus is seeded with ~80 handcrafted entries and augmented
at runtime from the advisory_corpus DB table.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from advisory.rule_engine import AdvisoryEngine, RiskAdvisory

logger = logging.getLogger("advisory.rag")

# ---------------------------------------------------------------------------
# Seed corpus — curated situation descriptions
# ---------------------------------------------------------------------------

SEED_CORPUS: list[dict] = [
    # CRITICAL — military
    {"situation_type": "military_escalation_critical", "risk_level": "CRITICAL",
     "text": "Full-scale military hostilities have been reported with cross-border artillery exchanges and mobilisation of armoured units. International observers have recorded multiple ceasefire violations.",
     "tags": ["military", "war", "escalation", "armed conflict", "artillery"]},
    {"situation_type": "state_collapse_critical", "risk_level": "CRITICAL",
     "text": "Central government authority has collapsed in key provinces. Armed non-state actors are contesting territorial control and critical infrastructure is under threat.",
     "tags": ["state failure", "instability", "armed groups", "governance"]},
    # CRITICAL — terrorism
    {"situation_type": "terrorism_critical", "risk_level": "CRITICAL",
     "text": "Mass casualty terrorist attacks have been carried out against civilian and government targets. Security forces are engaged in active counter-terrorism operations.",
     "tags": ["terrorism", "mass casualty", "attacks", "security operations"]},
    # CRITICAL — economic
    {"situation_type": "financial_crisis_critical", "risk_level": "CRITICAL",
     "text": "The national currency has collapsed and bank runs are reported across major cities. International credit ratings have been downgraded to junk status and capital controls imposed.",
     "tags": ["financial crisis", "currency collapse", "sanctions", "capital flight"]},
    # HIGH — military
    {"situation_type": "military_buildup_high", "risk_level": "HIGH",
     "text": "Large-scale troop concentrations have been observed near contested borders. Military exercises are underway with live ammunition and strategic bomber deployments have been noted.",
     "tags": ["military", "troop buildup", "border tension", "exercises"]},
    {"situation_type": "civil_war_high", "risk_level": "HIGH",
     "text": "Ongoing internal armed conflict between government forces and organised rebel factions. Urban fighting reported in multiple provincial capitals with significant civilian displacement.",
     "tags": ["civil war", "rebel", "armed conflict", "displacement", "urban fighting"]},
    # HIGH — protests
    {"situation_type": "mass_protest_high", "risk_level": "HIGH",
     "text": "Mass protests have paralysed major urban centres following disputed election results. Security forces have deployed riot control measures and dozens of arrests have been recorded.",
     "tags": ["protest", "unrest", "election", "riot", "civil disorder"]},
    # HIGH — terrorism
    {"situation_type": "terrorism_campaign_high", "risk_level": "HIGH",
     "text": "A coordinated terrorism campaign is targeting government institutions and transportation hubs. Bomb-making materials and suspect networks have been interdicted by security services.",
     "tags": ["terrorism", "bombing", "campaign", "security", "interdiction"]},
    # HIGH — sanctions
    {"situation_type": "sanctions_pressure_high", "risk_level": "HIGH",
     "text": "Comprehensive multilateral sanctions have been imposed targeting the financial sector, energy exports, and dual-use technology imports. The central bank's reserve position is deteriorating.",
     "tags": ["sanctions", "economic pressure", "financial", "energy", "reserves"]},
    # HIGH — diplomatic
    {"situation_type": "diplomatic_crisis_high", "risk_level": "HIGH",
     "text": "Senior diplomats have been expelled and embassies downgraded following a severe bilateral incident. Back-channel communications have been suspended and multilateral mediation is being sought.",
     "tags": ["diplomatic crisis", "expulsion", "bilateral", "mediation", "escalation"]},
    # ELEVATED — general
    {"situation_type": "political_instability_elevated", "risk_level": "ELEVATED",
     "text": "Governance indicators have weakened following cabinet resignations and parliamentary dysfunction. Public confidence in institutions is at a multi-year low.",
     "tags": ["political instability", "governance", "institutions", "cabinet"]},
    {"situation_type": "protest_elevated", "risk_level": "ELEVATED",
     "text": "Regular protest activity has increased significantly with demonstrations occurring in multiple cities weekly. Some incidents of property damage and clashes with police have been reported.",
     "tags": ["protest", "demonstrations", "police", "civil unrest"]},
    {"situation_type": "border_tension_elevated", "risk_level": "ELEVATED",
     "text": "Cross-border incidents including incursions, shot-firing, and arrests of nationals from the neighbouring state have increased sharply over the past month.",
     "tags": ["border", "tension", "incursion", "neighbouring", "incident"]},
    {"situation_type": "economic_stress_elevated", "risk_level": "ELEVATED",
     "text": "Inflation has exceeded 20% annually and unemployment is rising. IMF negotiations are underway and debt restructuring talks have begun with key creditors.",
     "tags": ["economic stress", "inflation", "unemployment", "IMF", "debt"]},
    {"situation_type": "terrorism_sporadic_elevated", "risk_level": "ELEVATED",
     "text": "Sporadic terrorist incidents continue in outlying provinces. Security forces have conducted targeted raids and several cells have been disrupted.",
     "tags": ["terrorism", "sporadic", "security forces", "raids", "cells"]},
    # MODERATE — general
    {"situation_type": "political_tension_moderate", "risk_level": "MODERATE",
     "text": "Political tensions between ruling and opposition parties have intensified ahead of scheduled elections. Media restrictions and arrests of opposition figures have been reported.",
     "tags": ["political tension", "elections", "opposition", "media"]},
    {"situation_type": "trade_dispute_moderate", "risk_level": "MODERATE",
     "text": "Bilateral trade disputes have escalated with tariff increases on key commodity sectors. Negotiations are ongoing but no resolution is imminent.",
     "tags": ["trade", "tariff", "dispute", "bilateral", "economic"]},
    {"situation_type": "military_low_level_moderate", "risk_level": "MODERATE",
     "text": "Low-level skirmishes have been recorded in disputed territory with casualties on both sides. No large-scale offensive has been launched.",
     "tags": ["skirmish", "disputed territory", "low-level conflict", "casualties"]},
    # LOW — general
    {"situation_type": "stable_low", "risk_level": "LOW",
     "text": "The security environment remains stable with no significant incidents reported. Diplomatic relations are functioning normally.",
     "tags": ["stable", "no incidents", "diplomatic"]},
    {"situation_type": "economic_recovery_low", "risk_level": "LOW",
     "text": "Economic indicators are improving with GDP growth resuming and foreign investment increasing. Confidence in the institutional framework is recovering.",
     "tags": ["economic recovery", "growth", "investment", "stability"]},
    # Contagion / spillover situations
    {"situation_type": "regional_contagion_high", "risk_level": "HIGH",
     "text": "Regional instability originating in a neighbouring state is exerting pressure through refugee flows, cross-border armed groups, and disrupted trade corridors.",
     "tags": ["contagion", "regional", "refugee", "spillover", "neighbouring", "cross-border"]},
    {"situation_type": "sanctions_secondary_elevated", "risk_level": "ELEVATED",
     "text": "Secondary sanctions risk is affecting the economy due to close financial and trade ties with a sanctioned state. Correspondent banking relationships are being disrupted.",
     "tags": ["secondary sanctions", "spillover", "banking", "financial contagion"]},
    # Trending deterioration
    {"situation_type": "deteriorating_trajectory", "risk_level": "ELEVATED",
     "text": "Multiple risk indicators are trending upward simultaneously suggesting a systemic deterioration rather than isolated incidents. Historical analogues suggest a 30-60 day escalation window.",
     "tags": ["deteriorating", "trend", "escalation", "systemic", "trajectory"]},
    {"situation_type": "improving_trajectory", "risk_level": "MODERATE",
     "text": "Risk indicators are showing a sustained downward trend following a period of elevated activity. Peace talks are progressing and confidence-building measures are in place.",
     "tags": ["improving", "trend", "peace talks", "de-escalation", "stabilisation"]},
    # Conflict types
    {"situation_type": "ethnic_conflict_high", "risk_level": "HIGH",
     "text": "Ethnic or communal violence has erupted following an incitement incident. Inter-communal clashes have been reported in several districts and security forces are struggling to contain them.",
     "tags": ["ethnic", "communal", "violence", "inter-communal", "clashes"]},
    {"situation_type": "coup_attempt_critical", "risk_level": "CRITICAL",
     "text": "A military or political coup attempt is underway or has recently occurred. Constitutional order is suspended and a transitional authority or military junta has assumed power.",
     "tags": ["coup", "military", "junta", "constitutional", "transitional"]},
    {"situation_type": "cyberattack_elevated", "risk_level": "ELEVATED",
     "text": "State-attributed cyberattacks have disrupted critical infrastructure including power grids and banking systems. Attribution points to a state actor with geopolitical motivations.",
     "tags": ["cyberattack", "critical infrastructure", "state actor", "power grid", "banking"]},
    {"situation_type": "humanitarian_crisis_high", "risk_level": "HIGH",
     "text": "A humanitarian crisis is deepening with food insecurity affecting large population segments. International aid organisations have raised emergency alerts.",
     "tags": ["humanitarian", "food insecurity", "aid", "crisis", "population"]},
    # Specific event patterns
    {"situation_type": "election_violence_high", "risk_level": "HIGH",
     "text": "Election-related violence has been reported including intimidation of candidates, attacks on polling stations, and post-result unrest. International election observers have flagged irregularities.",
     "tags": ["election", "violence", "polling", "post-election", "fraud"]},
    {"situation_type": "assassination_critical", "risk_level": "CRITICAL",
     "text": "A senior political or military figure has been assassinated or is subject to a credible assassination attempt. The incident has significantly destabilised the political environment.",
     "tags": ["assassination", "political figure", "destabilisation", "security"]},
]


# ---------------------------------------------------------------------------
# TF-IDF Retriever (pure numpy, no external deps)
# ---------------------------------------------------------------------------

class TFIDFRetriever:
    """
    Lightweight TF-IDF retriever for the advisory corpus.

    Builds a vocabulary from corpus text at init time.
    Retrieval via cosine similarity (numpy).
    """

    def __init__(self, vocab_size: int = 800):
        self._vocab_size = vocab_size
        self._vocab:   dict[str, int] = {}
        self._vectors: Optional[np.ndarray] = None   # (C, V) float32
        self._entries: list[dict] = []

    # ------------------------------------------------------------------
    # Corpus management
    # ------------------------------------------------------------------

    def index(self, entries: list[dict]) -> None:
        """
        Build TF-IDF index from corpus entries.

        Args:
            entries: list of dicts with at least a 'text' key.
                     Also uses 'tags', 'risk_level', 'situation_type'.
        """
        if not entries:
            self._entries = []
            self._vectors = np.zeros((0, 1), dtype=np.float32)
            return

        self._entries = entries

        # Build document strings (text + tags + situation_type)
        docs = [self._entry_to_doc(e) for e in entries]

        # Build vocabulary from top-N most common tokens
        token_freq: dict[str, int] = {}
        tokenised = [self._tokenise(d) for d in docs]
        for tokens in tokenised:
            for t in tokens:
                token_freq[t] = token_freq.get(t, 0) + 1

        # Sort by frequency, keep top vocab_size
        top_tokens = sorted(token_freq.items(), key=lambda x: -x[1])
        self._vocab = {t: i for i, (t, _) in enumerate(top_tokens[: self._vocab_size])}

        # Build TF-IDF matrix
        n_docs = len(docs)
        V = len(self._vocab)
        tf  = np.zeros((n_docs, V), dtype=np.float32)
        for di, tokens in enumerate(tokenised):
            for t in tokens:
                if t in self._vocab:
                    tf[di, self._vocab[t]] += 1.0
            row_sum = tf[di].sum()
            if row_sum > 0:
                tf[di] /= row_sum

        # IDF: log((1+N) / (1 + df)) + 1
        df = (tf > 0).sum(axis=0).astype(np.float32)
        idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

        tfidf = tf * idf

        # L2-normalise
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True) + 1e-8
        self._vectors = (tfidf / norms).astype(np.float32)

        logger.debug("TF-IDF index built: %d docs, %d vocab", n_docs, V)

    def retrieve(self, query_text: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve top-K corpus entries by cosine similarity.

        Args:
            query_text: Free-text query (advisory context string).
            top_k:      Number of results to return.

        Returns:
            list of dicts: each entry plus 'similarity' score.
        """
        if self._vectors is None or len(self._entries) == 0:
            return []

        q_vec = self._vectorise(query_text)
        if q_vec is None:
            return []

        sims = (self._vectors @ q_vec).flatten()   # (C,)

        if len(sims) == 0:
            return []

        top_idxs = np.argsort(-sims)[:top_k]
        results  = []
        for idx in top_idxs:
            if sims[idx] < 0.05:   # below minimum relevance threshold
                continue
            entry = dict(self._entries[idx])
            entry["similarity"] = round(float(sims[idx]), 4)
            results.append(entry)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _entry_to_doc(self, entry: dict) -> str:
        parts = [entry.get("text", "")]
        parts.extend(entry.get("tags", []))
        parts.append(entry.get("situation_type", "").replace("_", " "))
        parts.append(entry.get("risk_level", ""))
        return " ".join(parts)

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r"\b[a-z]{2,}\b", text)
        # Remove very common stop words
        _stop = {
            "the", "a", "an", "is", "are", "in", "of", "to", "and",
            "or", "for", "on", "at", "by", "have", "has", "been", "be",
            "with", "from", "this", "that", "it", "its", "were", "was",
            "as", "not", "but", "all", "also", "been", "their",
        }
        return [t for t in tokens if t not in _stop]

    def _vectorise(self, text: str) -> Optional[np.ndarray]:
        tokens = self._tokenise(text)
        if not tokens:
            return None
        V = len(self._vocab)
        if V == 0:
            return None
        vec = np.zeros(V, dtype=np.float32)
        for t in tokens:
            if t in self._vocab:
                vec[self._vocab[t]] += 1.0
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm


# ---------------------------------------------------------------------------
# RAG Advisory Engine
# ---------------------------------------------------------------------------

class RAGAdvisoryEngine:
    """
    Enriches rule-based advisories with retrieved context snippets.

    Usage::

        engine = RAGAdvisoryEngine()
        advisory = engine.generate_rag(
            country="Ukraine",
            risk_score=0.82,
            confidence=0.78,
            trend="increasing",
            instability=0.85,
            war=0.91,
            terrorism=0.44,
            financial=0.33,
        )
    """

    def __init__(
        self,
        extra_corpus: Optional[list[dict]] = None,
        top_k: int = 3,
        ollama_enabled: bool = False,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "mistral",
    ):
        self._base_engine = AdvisoryEngine(
            ollama_enabled=ollama_enabled, ollama_url=ollama_url
        )
        self._retriever  = TFIDFRetriever(vocab_size=800)
        self._top_k      = top_k
        self._ollama     = ollama_enabled
        self._ollama_url = ollama_url
        self._ollama_model = ollama_model

        # Seed + optional extra corpus
        corpus = list(SEED_CORPUS)
        if extra_corpus:
            corpus.extend(extra_corpus)
        self._retriever.index(corpus)

    def add_corpus_entries(self, entries: list[dict]) -> None:
        """Add new entries to the corpus and rebuild index."""
        corpus = list(SEED_CORPUS) + entries
        self._retriever.index(corpus)

    def generate_rag(
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
        forecast_trajectory: Optional[list[float]] = None,
    ) -> tuple[RiskAdvisory, list[dict]]:
        """
        Generate a RAG-enriched advisory.

        Returns:
            (advisory, retrieved_contexts)
            advisory: RiskAdvisory with enriched advisory_text
            retrieved_contexts: top-K retrieved corpus entries
        """
        # Step 1: Generate base advisory from rule engine
        base = self._base_engine.generate(
            country=country,
            risk_score=risk_score,
            confidence=confidence,
            trend=trend,
            instability=instability,
            war=war,
            terrorism=terrorism,
            financial=financial,
            feature_scores=feature_scores or {},
        )

        # Step 2: Build retrieval query
        query = self._build_query(base, forecast_trajectory)

        # Step 3: Retrieve similar situations
        retrieved = self._retriever.retrieve(query, top_k=self._top_k)

        # Step 4: Augment advisory text with retrieved context
        enriched_text = self._augment(base, retrieved, forecast_trajectory)

        # Step 5: Optional Ollama polish
        if self._ollama and retrieved:
            enriched_text = self._ollama_polish(
                country=country,
                base_text=enriched_text,
                retrieved=retrieved,
                base_advisory=base,
            )

        base.advisory_text = enriched_text
        return base, retrieved

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_query(
        self,
        advisory: RiskAdvisory,
        forecast: Optional[list[float]],
    ) -> str:
        parts = [advisory.level, advisory.trend]
        parts.extend(advisory.major_drivers)

        # Encode trajectory direction
        if forecast and len(forecast) >= 2:
            if forecast[-1] > forecast[0] + 0.05:
                parts.append("deteriorating trajectory escalation")
            elif forecast[-1] < forecast[0] - 0.05:
                parts.append("improving trajectory de-escalation")

        # Add risk-head keywords
        if advisory.risk_score >= 0.80:
            parts.append("critical instability conflict crisis")
        elif advisory.risk_score >= 0.65:
            parts.append("high risk armed conflict protests")

        return " ".join(parts)

    def _augment(
        self,
        base: RiskAdvisory,
        retrieved: list[dict],
        forecast: Optional[list[float]],
    ) -> str:
        text = base.advisory_text

        if not retrieved:
            if forecast:
                text += " " + self._forecast_clause(forecast)
            return text

        # Add retrieved context as "Historical analogues" paragraph
        analogue_lines = []
        for r in retrieved[:2]:   # at most 2 to avoid bloat
            sim = r.get("similarity", 0)
            if sim < 0.05:
                continue
            snippet = r.get("text", "")
            if snippet:
                analogue_lines.append(f"[{r.get('risk_level','')}/{r.get('situation_type','').replace('_',' ')}] {snippet}")

        if analogue_lines:
            text += (
                " Historical analogues of similar risk patterns: "
                + " — ".join(analogue_lines)
            )

        if forecast:
            text += " " + self._forecast_clause(forecast)

        return text

    @staticmethod
    def _forecast_clause(forecast: list[float]) -> str:
        if not forecast:
            return ""
        current = forecast[0]
        final   = forecast[-1]
        horizon_weeks = len(forecast) * 2

        if final > current + 0.08:
            return (
                f"Model forecast projects risk escalating to {final:.2f} "
                f"within {horizon_weeks} weeks — heightened monitoring advised."
            )
        elif final < current - 0.08:
            return (
                f"Model forecast projects risk moderating to {final:.2f} "
                f"within {horizon_weeks} weeks — conditions may stabilise."
            )
        else:
            return (
                f"Model forecast is stable ({final:.2f}) over the next "
                f"{horizon_weeks} weeks."
            )

    def _ollama_polish(
        self,
        country: str,
        base_text: str,
        retrieved: list[dict],
        base_advisory: RiskAdvisory,
    ) -> str:
        """Polish the advisory via Ollama with retrieved context injected."""
        try:
            import requests as req
            context_snippets = "\n".join(
                f"- {r['text']}" for r in retrieved[:2]
            )
            prompt = (
                f"You are a senior geopolitical risk analyst. "
                f"Rewrite the following advisory in a concise professional style "
                f"(3-4 sentences). Incorporate the relevant analogues below where "
                f"appropriate. Do not add speculation or new facts.\n\n"
                f"Country: {country}\n"
                f"Risk: {base_advisory.level} ({base_advisory.risk_score:.2f})\n"
                f"Trend: {base_advisory.trend}\n\n"
                f"Draft advisory:\n{base_text}\n\n"
                f"Relevant analogues:\n{context_snippets}\n\n"
                f"Rewritten advisory (professional, factual):"
            )
            resp = req.post(
                f"{self._ollama_url}/api/generate",
                json={"model": self._ollama_model, "prompt": prompt, "stream": False},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json().get("response", "").strip()
            return result if result else base_text
        except Exception as exc:
            logger.debug("Ollama polish failed: %s", exc)
            return base_text
