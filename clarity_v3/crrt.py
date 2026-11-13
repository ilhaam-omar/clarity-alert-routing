"""Time-scoped CRRT context derived from MIMIC-IV ``procedureevents``.

A medication order is classified as occurring during CRRT only when its
timestamp falls inside a non-cancelled, valid procedure interval for the same
patient and admission. Admission-wide CRRT flags are deliberately not exposed
in this clean V2 API because they can leak context to orders before treatment
starts or after it ends.

The item registry is an explicit prototype mapping and must be checked against
the local MIMIC dictionary before full-cohort interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

import pandas as pd


CRRT_ITEM_REGISTRY: dict[int, dict[str, str]] = {
    225802: {
        "modality": "CRRT (generic)",
        "expected_label_regex": r"\bCRRT\b",
        "scope_basis": "continuous renal replacement therapy",
    },
    225803: {
        "modality": "CVVHD",
        "expected_label_regex": r"\bCVVHD\b",
        "scope_basis": "continuous veno-venous hemodialysis",
    },
    225809: {
        "modality": "CVVHDF",
        "expected_label_regex": r"\bCVVHDF\b",
        "scope_basis": "continuous veno-venous hemodiafiltration",
    },
    225955: {
        "modality": "SCUF",
        "expected_label_regex": r"\bSCUF\b",
        "scope_basis": "slow continuous ultrafiltration; disclosed CRRT-scope choice",
    },
}

CRRT_ITEMIDS = frozenset(CRRT_ITEM_REGISTRY)

KNOWN_EXCLUDED_DIALYSIS_ITEMIDS: dict[int, dict[str, str]] = {
    225441: {
        "reason": "standard/intermittent hemodialysis",
        "expected_label_regex": r"hemodialysis|dialysis",
    },
    225805: {
        "reason": "peritoneal dialysis",
        "expected_label_regex": r"peritoneal",
    },
}

CRRT_STRUCTURED_CONFIDENCE = 0.98


@dataclass(frozen=True, order=True)
class CrrtInterval:
    """One non-cancelled CRRT procedure interval."""

    start: pd.Timestamp
    end: pd.Timestamp
    itemid: int

    def contains(self, timestamp: pd.Timestamp) -> bool:
        # Closed interval: orders stamped exactly at start/end are treated as
        # contemporaneous with the recorded procedure.
        return self.start <= timestamp <= self.end


def _read_procedureevents(
    procedureevents_path: str, required_columns: set[str]
) -> pd.DataFrame:
    if not os.path.exists(procedureevents_path):
        raise FileNotFoundError(f"procedureevents.csv not found at {procedureevents_path}")

    header = pd.read_csv(procedureevents_path, nrows=0)
    available = set(header.columns)
    missing = required_columns - available
    if missing:
        raise ValueError(
            f"procedureevents.csv is missing expected column(s): {sorted(missing)}. "
            f"Found columns: {list(header.columns)}. Resolve the schema rather than "
            "silently guessing a replacement."
        )
    optional = {"cancelreason", "statusdescription"}.intersection(available)
    usecols = sorted(required_columns | optional)
    return pd.read_csv(procedureevents_path, usecols=usecols, low_memory=False)


def _require_columns(df: pd.DataFrame, required: set[str], source_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{source_name} is missing expected column(s): {sorted(missing)}. "
            f"Found columns: {list(df.columns)}. Resolve the schema rather than "
            "silently guessing a replacement."
        )


def _cancelled_mask(df: pd.DataFrame) -> pd.Series:
    """Conservatively identify explicitly cancelled procedure rows."""
    cancelled = pd.Series(False, index=df.index)
    if "cancelreason" in df.columns:
        values = pd.to_numeric(df["cancelreason"], errors="coerce").fillna(0)
        cancelled |= values.ne(0)
    if "statusdescription" in df.columns:
        cancelled |= (
            df["statusdescription"]
            .astype("string")
            .str.contains(r"cancel", case=False, na=False, regex=True)
        )
    return cancelled


def _selected_rows(df: pd.DataFrame) -> pd.DataFrame:
    itemids = pd.to_numeric(df["itemid"], errors="coerce")
    selected = df[itemids.isin(CRRT_ITEMIDS)].copy()
    if not selected.empty:
        selected["itemid"] = pd.to_numeric(selected["itemid"], errors="raise").astype(int)
    return selected


def _merge_intervals(intervals: Iterable[CrrtInterval]) -> tuple[CrrtInterval, ...]:
    """Merge overlapping intervals without expanding their recorded union."""
    ordered = sorted(intervals, key=lambda value: (value.start, value.end, value.itemid))
    if not ordered:
        return ()

    merged: list[CrrtInterval] = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.start <= previous.end:
            merged[-1] = CrrtInterval(
                start=previous.start,
                end=max(previous.end, current.end),
                itemid=previous.itemid,
            )
        else:
            merged.append(current)
    return tuple(merged)


def load_icu_crrt_intervals(
    procedureevents_path: str,
) -> dict[tuple[int, int], tuple[CrrtInterval, ...]]:
    """Load time-aware CRRT intervals keyed by ``(subject_id, hadm_id)``.

    Non-cancelled selected rows with missing identifiers, unparseable
    timestamps, or end times preceding start times cause a loud failure. This
    prevents a longitudinal risk adjustment from being driven by silently
    malformed timing data.
    """
    required = {"subject_id", "hadm_id", "itemid", "starttime", "endtime"}
    df = _read_procedureevents(procedureevents_path, required)
    _require_columns(df, required, "procedureevents.csv")

    selected = _selected_rows(df)
    if selected.empty:
        return {}

    selected = selected.loc[~_cancelled_mask(selected)].copy()
    if selected.empty:
        return {}

    invalid_identifiers = selected[["subject_id", "hadm_id"]].isna().any(axis=1)
    if invalid_identifiers.any():
        raise ValueError(
            f"{int(invalid_identifiers.sum())} non-cancelled CRRT row(s) lack "
            "subject_id or hadm_id and cannot be admission-scoped safely."
        )

    selected["starttime"] = pd.to_datetime(selected["starttime"], errors="coerce")
    selected["endtime"] = pd.to_datetime(selected["endtime"], errors="coerce")
    invalid_time = selected[["starttime", "endtime"]].isna().any(axis=1)
    if invalid_time.any():
        raise ValueError(
            f"{int(invalid_time.sum())} non-cancelled CRRT row(s) have missing or "
            "unparseable start/end timestamps."
        )

    reversed_time = selected["endtime"] < selected["starttime"]
    if reversed_time.any():
        raise ValueError(
            f"{int(reversed_time.sum())} CRRT row(s) have endtime before starttime."
        )

    selected["subject_id"] = pd.to_numeric(
        selected["subject_id"], errors="raise"
    ).astype(int)
    selected["hadm_id"] = pd.to_numeric(selected["hadm_id"], errors="raise").astype(int)

    grouped: dict[tuple[int, int], list[CrrtInterval]] = {}
    for row in selected.itertuples(index=False):
        key = (int(row.subject_id), int(row.hadm_id))
        grouped.setdefault(key, []).append(
            CrrtInterval(
                start=pd.Timestamp(row.starttime),
                end=pd.Timestamp(row.endtime),
                itemid=int(row.itemid),
            )
        )

    return {key: _merge_intervals(values) for key, values in grouped.items()}


def get_icu_crrt_signal_at_time(
    intervals: dict[tuple[int, int], tuple[CrrtInterval, ...]],
    subject_id,
    hadm_id,
    timestamp,
) -> dict:
    """Return time-scoped CRRT context for one medication order."""
    try:
        key = (int(subject_id), int(hadm_id))
    except (TypeError, ValueError):
        return {
            "crrt_active": False,
            "crrt_confidence": None,
            "crrt_event_in_admission": False,
            "crrt_temporal_scope": "invalid_identifier",
        }

    ts = pd.to_datetime(timestamp, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Medication-order timestamp is missing or unparseable: {timestamp!r}")
    ts = pd.Timestamp(ts)

    admission_intervals = intervals.get(key, ())
    active = any(interval.contains(ts) for interval in admission_intervals)
    if active:
        temporal_scope = "inside_recorded_interval"
    elif admission_intervals:
        temporal_scope = "outside_recorded_interval"
    else:
        temporal_scope = "no_recorded_interval"
    return {
        "crrt_active": active,
        "crrt_confidence": CRRT_STRUCTURED_CONFIDENCE if active else None,
        "crrt_event_in_admission": bool(admission_intervals),
        "crrt_temporal_scope": temporal_scope,
    }
