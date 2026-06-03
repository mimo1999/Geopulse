"""
Phase 3: RAG Engine — runtime wrapper for RAGAdvisoryEngine.

Loads advisory corpus from DB (advisory_corpus table) at startup,
supplements seed corpus, and provides country-level enriched advisories.
Also handles corpus building from event_clusters data.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import psycopg2
import psycopg2.extras

from advisory.rag_advisory import RAGAdvisoryEngine, SEED_CORPUS
from advisory.rule_engine import RiskAdvisory

logger = logging.getLogger("inference.rag_engine")


class RAGEngine:
    """
    High-level RAG advisory engine with DB-backed corpus.

    Loads advisory_corpus table at init and keeps an in-memory
    TF-IDF index. Re-indexing is triggered by rebuild_corpus().

    Usage::

        engine = RAGEngine(dsn=dsn)
        advisory, contexts = engine.generate(country="UA", ...)
    """

    def __init__(
        self,
        dsn: str,
        top_k: int = 3,
        ollama_enabled: bool = False,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "mistral",
    ):
        self._dsn  = dsn
        self._top_k = top_k

        # Load extra corpus entries from DB
        db_entries = self._load_corpus_from_db()
        logger.info("RAG corpus: %d seed + %d DB entries", len(SEED_CORPUS), len(db_entries))

        self._engine = RAGAdvisoryEngine(
            extra_corpus=db_entries,
            top_k=top_k,
            ollama_enabled=ollama_enabled,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        forecast_trajectory: Optional[list[float]] = None,
    ) -> tuple[RiskAdvisory, list[dict]]:
        """
        Generate a RAG-enriched advisory for a country.

        Returns:
            (advisory, retrieved_contexts)
        """
        return self._engine.generate_rag(
            country=country,
            risk_score=risk_score,
            confidence=confidence,
            trend=trend,
            instability=instability,
            war=war,
            terrorism=terrorism,
            financial=financial,
            feature_scores=feature_scores,
            forecast_trajectory=forecast_trajectory,
        )

    def rebuild_corpus(self) -> int:
        """
        Rebuild the TF-IDF index from DB + seed corpus.
        Returns total corpus size.
        """
        db_entries = self._load_corpus_from_db()
        self._engine.add_corpus_entries(db_entries)
        total = len(SEED_CORPUS) + len(db_entries)
        logger.info("RAG corpus rebuilt: %d entries", total)
        return total

    def build_corpus_from_clusters(
        self,
        country: Optional[str] = None,
        days: int = 90,
        persist: bool = True,
    ) -> int:
        """
        Auto-generate corpus entries from recent event_clusters data.
        Converts cluster summaries into situation descriptions.

        Args:
            country: Specific country (None = all).
            days:    How many days back to scan.
            persist: Save generated entries to advisory_corpus table.

        Returns:
            Number of entries generated.
        """
        since = (date.today().replace(year=date.today().year) -
                 __import__("datetime").timedelta(days=days))
        try:
            conn = psycopg2.connect(self._dsn)
            with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if country:
                    cur.execute("""
                        SELECT country, cluster_date, category, event_count,
                               total_mentions, avg_goldstein, avg_tone
                        FROM event_clusters
                        WHERE country = %s AND cluster_date >= %s
                        ORDER BY total_mentions DESC
                        LIMIT 200
                    """, (country, since))
                else:
                    cur.execute("""
                        SELECT country, cluster_date, category, event_count,
                               total_mentions, avg_goldstein, avg_tone
                        FROM event_clusters
                        WHERE cluster_date >= %s
                        ORDER BY total_mentions DESC
                        LIMIT 500
                    """, (since,))
                rows = [dict(r) for r in cur.fetchall()]
            conn.close()
        except Exception as exc:
            logger.warning("Failed to load event clusters for corpus building: %s", exc)
            return 0

        entries = [self._cluster_to_corpus_entry(r) for r in rows]
        entries = [e for e in entries if e]

        if persist and entries:
            self._persist_corpus_entries(entries)

        self._engine.add_corpus_entries(entries)
        logger.info("Built %d corpus entries from event clusters", len(entries))
        return len(entries)

    def get_corpus_stats(self) -> dict:
        """Return summary of current corpus."""
        try:
            conn = psycopg2.connect(self._dsn)
            with conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT risk_level, COUNT(*) AS cnt
                    FROM advisory_corpus
                    GROUP BY risk_level
                    ORDER BY cnt DESC
                """)
                rows = cur.fetchall()
            conn.close()
            db_counts = {r[0]: r[1] for r in rows}
        except Exception:
            db_counts = {}

        return {
            "seed_entries":  len(SEED_CORPUS),
            "db_entries":    sum(db_counts.values()),
            "by_level":      db_counts,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_corpus_from_db(self) -> list[dict]:
        try:
            conn = psycopg2.connect(self._dsn)
            with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT situation_type, risk_level, text, tags
                    FROM advisory_corpus
                    ORDER BY created_at DESC
                    LIMIT 500
                """)
                rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            return rows
        except Exception as exc:
            logger.debug("Failed to load advisory corpus from DB: %s", exc)
            return []

    @staticmethod
    def _cluster_to_corpus_entry(row: dict) -> Optional[dict]:
        """Convert an event_cluster row to an advisory corpus entry."""
        category  = row.get("category", "")
        mentions  = int(row.get("total_mentions") or 0)
        goldstein = float(row.get("avg_goldstein") or 0.0)
        tone      = float(row.get("avg_tone") or 0.0)
        count     = int(row.get("event_count") or 0)
        country   = row.get("country", "")

        if count < 3 or mentions < 10:
            return None

        # Determine risk level from goldstein + mentions
        if goldstein < -6 or mentions > 5000:
            risk_level = "CRITICAL"
        elif goldstein < -4 or mentions > 2000:
            risk_level = "HIGH"
        elif goldstein < -2 or mentions > 500:
            risk_level = "ELEVATED"
        elif goldstein < 0:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        # Category-specific text template
        category_descs = {
            "protest":    f"significant protest activity ({count} events, {mentions} mentions)",
            "military":   f"military and armed conflict events ({count} events, {mentions} mentions)",
            "terrorism":  f"terrorism and asymmetric violence ({count} events, {mentions} mentions)",
            "sanctions":  f"sanctions and economic coercion measures ({count} events, {mentions} mentions)",
            "diplomatic": f"diplomatic tensions and negotiations ({count} events, {mentions} mentions)",
        }
        desc = category_descs.get(category, f"{category} events ({count} events)")

        goldstein_desc = ""
        if goldstein < -5:
            goldstein_desc = "with extreme negative Goldstein scale scores indicating severe destabilisation"
        elif goldstein < -3:
            goldstein_desc = "with strongly negative conflict indicators"
        elif goldstein < 0:
            goldstein_desc = "with moderately negative conflict indicators"

        tone_desc = ""
        if tone < -10:
            tone_desc = "Media coverage is strongly hostile"
        elif tone < -5:
            tone_desc = "Media tone is negative"

        text = (
            f"Recorded {desc} {goldstein_desc}. "
            f"{tone_desc}. "
            f"Average Goldstein scale: {goldstein:.1f}."
        ).strip()

        return {
            "situation_type": f"{category}_{risk_level.lower()}_auto",
            "risk_level":     risk_level,
            "text":           text,
            "tags":           [category, country, risk_level.lower()],
            "source":         "event_cluster",
        }

    def _persist_corpus_entries(self, entries: list[dict]) -> None:
        sql = """
            INSERT INTO advisory_corpus (situation_type, risk_level, text, tags, source)
            VALUES (%(situation_type)s, %(risk_level)s, %(text)s, %(tags)s, %(source)s)
            ON CONFLICT DO NOTHING
        """
        try:
            conn = psycopg2.connect(self._dsn)
            with conn, conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, sql, entries)
            conn.close()
        except Exception as exc:
            logger.warning("Failed to persist corpus entries: %s", exc)
