"""
POLECAT event data parser.

Reads POLECAT tab-separated TXT files (ngecEvents.DV.YYYY.txt),
maps ISO-3 country codes to FIPS 10-4, and aggregates per-(country, date)
feature rows matching the country_daily_features schema.

Event type → feature mapping (mirrors feature_extractor.py conventions):
  PROTEST                            → protest_score
  ASSAULT, MOBILIZE, RETREAT        → violence_score
  ASSAULT in MATERIAL CONFLICT       → terrorism_score
  THREATEN, COERCE, ACCUSE, REJECT   → diplomatic_stress (via negative intensity)
  SANCTION                           → economic_stress
  Event Intensity (-10..+10)         → avg_goldstein (normalized 0..1)
  Quad Code CONFLICT/COOPERATION     → avg_sentiment proxy
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Iterator

from data.iso3_to_fips import iso3_to_fips

logger = logging.getLogger("ingestion.polecat_parser")

# POLECAT event types mapped to pipeline feature dimensions
PROTEST_TYPES   = {"PROTEST"}
VIOLENCE_TYPES  = {"ASSAULT", "MOBILIZE", "RETREAT"}
TERROR_TYPES    = {"ASSAULT"}        # only when Quad Code = MATERIAL CONFLICT
STRESS_TYPES    = {"THREATEN", "COERCE", "ACCUSE", "REJECT"}
ECONOMIC_TYPES  = {"SANCTION"}

# Quad Code prefix identifying cooperative vs conflict events
CONFLICT_QUAD   = "CONFLICT"
MATERIAL_CONF   = "MATERIAL CONFLICT"


def _parse_intensity(raw: str) -> float | None:
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def _norm_goldstein(intensity: float) -> float:
    """Map -10..+10 to 0..1 (same formula as feature_extractor.py)."""
    return (intensity + 10.0) / 20.0


def parse_file(
    txt_file: io.TextIOWrapper,
    target_date: date | None = None,
) -> dict[tuple[str, date], dict]:
    """
    Parse one POLECAT TXT file and return aggregated feature rows keyed by
    (fips_country, event_date).

    Args:
        txt_file:    File-like object (text mode) for a POLECAT TXT.
        target_date: If set, only events on this date are included.

    Returns:
        Dict mapping (fips_code, date) → feature accumulator dict.
    """
    reader = csv.DictReader(txt_file, delimiter="\t")

    # Accumulators: (fips, date) → counters
    agg: dict[tuple[str, date], dict] = defaultdict(lambda: {
        "total": 0,
        "protest": 0,
        "violence": 0,
        "terror": 0,
        "stress": 0,
        "economic": 0,
        "conflict": 0,
        "coop": 0,
        "goldstein_sum": 0.0,
        "goldstein_n": 0,
    })

    skipped_iso3 = set()

    for row in reader:
        raw_date = row.get("Event Date", "").strip()
        try:
            ev_date = date.fromisoformat(raw_date)
        except ValueError:
            continue

        if target_date is not None and ev_date != target_date:
            continue

        iso3 = row.get("Country", "").strip()
        fips = iso3_to_fips(iso3)
        if fips is None:
            if iso3 and iso3 != "None":
                skipped_iso3.add(iso3)
            continue

        ev_type = row.get("Event Type", "").strip().upper()
        quad    = row.get("Quad Code", "").strip().upper()
        raw_int = row.get("Event Intensity", "").strip()
        intensity = _parse_intensity(raw_int)

        key = (fips, ev_date)
        acc = agg[key]
        acc["total"] += 1

        if ev_type in PROTEST_TYPES:
            acc["protest"] += 1
        if ev_type in VIOLENCE_TYPES:
            acc["violence"] += 1
        if ev_type in TERROR_TYPES and MATERIAL_CONF in quad:
            acc["terror"] += 1
        if ev_type in STRESS_TYPES:
            acc["stress"] += 1
        if ev_type in ECONOMIC_TYPES:
            acc["economic"] += 1

        if CONFLICT_QUAD in quad:
            acc["conflict"] += 1
        else:
            acc["coop"] += 1

        if intensity is not None:
            acc["goldstein_sum"] += intensity
            acc["goldstein_n"]   += 1

    if skipped_iso3:
        logger.debug("Skipped unmapped ISO-3 codes: %s", sorted(skipped_iso3))

    # Convert accumulators to feature rows
    rows: dict[tuple[str, date], dict] = {}
    for (fips, ev_date), acc in agg.items():
        n = acc["total"]
        if n == 0:
            continue

        def ratio(count: int) -> float:
            return round(min(count / n, 1.0), 6)

        if acc["goldstein_n"] > 0:
            avg_intensity = acc["goldstein_sum"] / acc["goldstein_n"]
        else:
            avg_intensity = 0.0

        goldstein_norm = round(_norm_goldstein(avg_intensity), 6)
        # diplomatic_stress: high stress when avg intensity is negative
        diplomatic_stress = round(1.0 - goldstein_norm, 6)
        # avg_sentiment: cooperation fraction as proxy
        total_coded = acc["conflict"] + acc["coop"]
        avg_sentiment = round(acc["coop"] / total_coded, 6) if total_coded else 0.5

        rows[(fips, ev_date)] = {
            "country":            fips,
            "feature_date":       ev_date,
            "total_events":       n,
            "conflict_events":    acc["conflict"],
            "cooperation_events": acc["coop"],
            "protest_score":      ratio(acc["protest"]),
            "violence_score":     ratio(acc["violence"]),
            "diplomatic_stress":  diplomatic_stress,
            "economic_stress":    ratio(acc["economic"]),
            "terrorism_score":    ratio(acc["terror"]),
            "avg_sentiment":      avg_sentiment,
            "avg_goldstein":      goldstein_norm,
        }

    return rows


def iter_zip(
    zip_path: Path,
    target_date: date | None = None,
    years: list[int] | None = None,
) -> Iterator[dict]:
    """
    Yield feature row dicts from all POLECAT TXT files in a zip archive.

    Args:
        zip_path:    Path to dataverse_files.zip (or any zip of POLECAT TXTs).
        target_date: Filter to a single date if set.
        years:       Restrict to specific years (e.g. [2022, 2023]).
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        txt_names = [n for n in zf.namelist() if n.endswith(".txt")]
        if years:
            txt_names = [
                n for n in txt_names
                if any(str(y) in n for y in years)
            ]
        logger.info("Reading %d POLECAT files from %s", len(txt_names), zip_path.name)

        for name in sorted(txt_names):
            logger.info("Parsing %s", name)
            with zf.open(name) as raw:
                txt = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                rows = parse_file(txt, target_date=target_date)
            for row in rows.values():
                yield row
