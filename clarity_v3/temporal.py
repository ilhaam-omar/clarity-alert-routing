from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Dict, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class RenalSnapshot:
    current_creatinine: Optional[float]
    current_time: Optional[pd.Timestamp]
    current_hadm_id: Optional[int]
    current_same_admission: bool
    hours_since_current: Optional[float]
    baseline_48h_creatinine: Optional[float]
    baseline_48h_time: Optional[pd.Timestamp]
    baseline_7d_creatinine: Optional[float]
    baseline_7d_time: Optional[pd.Timestamp]
    creatinine_delta_48h: Optional[float]
    creatinine_ratio_7d: Optional[float]
    acute_deterioration_48h: bool
    acute_deterioration_7d: bool
    acute_renal_deterioration: bool
    acute_deterioration_basis: str
    egfr_current: Optional[float]


def compute_egfr_ckdepi2021(
    creatinine_mg_dl: Optional[float], age: int, gender: Optional[str]
) -> Optional[float]:
    if creatinine_mg_dl is None or pd.isna(creatinine_mg_dl) or creatinine_mg_dl <= 0:
        return None
    if str(gender).upper() == "F":
        kappa, alpha, sex_factor = 0.7, -0.241, 1.012
    else:
        kappa, alpha, sex_factor = 0.9, -0.302, 1.0
    ratio = float(creatinine_mg_dl) / kappa
    return round(
        142
        * min(ratio, 1.0) ** alpha
        * max(ratio, 1.0) ** -1.2
        * 0.9938 ** int(age)
        * sex_factor,
        1,
    )


def _minimum_prior(
    prior: pd.DataFrame,
    timestamp: pd.Timestamp,
    hours: float,
) -> tuple[Optional[float], Optional[pd.Timestamp]]:
    start = pd.Timestamp(timestamp) - pd.Timedelta(hours=hours)
    window = prior[(prior["charttime"] >= start) & (prior["charttime"] < timestamp)]
    if window.empty:
        return None, None
    index = pd.to_numeric(window["valuenum"], errors="coerce").idxmin()
    if pd.isna(index):
        return None, None
    row = window.loc[index]
    return float(row["valuenum"]), pd.Timestamp(row["charttime"])


def renal_snapshot(
    labs: pd.DataFrame,
    timestamp: pd.Timestamp,
    hadm_id: int,
    age: int,
    gender: Optional[str],
) -> RenalSnapshot:
    """Return time-causal renal state using KDIGO-aligned windows.

    The function does not diagnose AKI. It constructs a medication-safety
    context flag using the published serum-creatinine windows: an increase of
    at least 0.3 mg/dL within 48 hours or at least 1.5-fold relative to a prior
    value within seven days. All values must precede or equal the order time.
    """
    if labs.empty:
        return RenalSnapshot(
            None, None, None, False, None,
            None, None, None, None, None, None,
            False, False, False, "no_prior_creatinine", None,
        )
    working = labs.copy()
    working["charttime"] = pd.to_datetime(working["charttime"], errors="coerce")
    working["valuenum"] = pd.to_numeric(working["valuenum"], errors="coerce")
    working["hadm_id"] = pd.to_numeric(working.get("hadm_id"), errors="coerce")
    prior = working[
        working["charttime"].notna()
        & working["valuenum"].notna()
        & (working["charttime"] <= pd.Timestamp(timestamp))
        & (working["valuenum"] > 0)
    ].sort_values("charttime", kind="stable")
    if prior.empty:
        return RenalSnapshot(
            None, None, None, False, None,
            None, None, None, None, None, None,
            False, False, False, "no_prior_creatinine", None,
        )
    current = prior.iloc[-1]
    current_value = float(current["valuenum"])
    current_time = pd.Timestamp(current["charttime"])
    current_hadm = None if pd.isna(current.get("hadm_id")) else int(current["hadm_id"])
    hours_since = max(
        0.0,
        (pd.Timestamp(timestamp) - current_time).total_seconds() / 3600.0,
    )
    baseline_48h, baseline_48h_time = _minimum_prior(prior, current_time, 48.0)
    baseline_7d, baseline_7d_time = _minimum_prior(prior, current_time, 24.0 * 7.0)
    delta_48h = None if baseline_48h is None else current_value - baseline_48h
    ratio_7d = (
        None
        if baseline_7d is None or baseline_7d <= 0
        else current_value / baseline_7d
    )
    acute_48h = bool(delta_48h is not None and delta_48h >= 0.3 - 1e-12)
    acute_7d = bool(ratio_7d is not None and ratio_7d >= 1.5 - 1e-12)
    if acute_48h and acute_7d:
        basis = "creatinine_increase_0.3_within_48h_and_1.5x_within_7d"
    elif acute_48h:
        basis = "creatinine_increase_0.3_within_48h"
    elif acute_7d:
        basis = "creatinine_increase_1.5x_within_7d"
    elif baseline_48h is None and baseline_7d is None:
        basis = "insufficient_acute_baseline"
    else:
        basis = "no_kdigo_aligned_acute_deterioration"
    return RenalSnapshot(
        current_creatinine=current_value,
        current_time=current_time,
        current_hadm_id=current_hadm,
        current_same_admission=(current_hadm == int(hadm_id)),
        hours_since_current=hours_since,
        baseline_48h_creatinine=baseline_48h,
        baseline_48h_time=baseline_48h_time,
        baseline_7d_creatinine=baseline_7d,
        baseline_7d_time=baseline_7d_time,
        creatinine_delta_48h=delta_48h,
        creatinine_ratio_7d=ratio_7d,
        acute_deterioration_48h=acute_48h,
        acute_deterioration_7d=acute_7d,
        acute_renal_deterioration=(acute_48h or acute_7d),
        acute_deterioration_basis=basis,
        egfr_current=compute_egfr_ckdepi2021(current_value, age, gender),
    )


def evidence_recency_class(hours_since_lab: Optional[float]) -> str:
    if hours_since_lab is None or pd.isna(hours_since_lab):
        return "missing"
    value = float(hours_since_lab)
    if value <= 24:
        return "within_24h"
    if value <= 48:
        return "within_48h"
    if value <= 72:
        return "within_72h"
    return "older_than_72h"


def evidence_recency_index(hours_since_lab: Optional[float], half_life_hours: float = 48.0) -> float:
    """Prototype completeness index, not a calibrated probability."""
    if hours_since_lab is None or pd.isna(hours_since_lab):
        return 0.0
    if half_life_hours <= 0:
        raise ValueError("half_life_hours must be positive")
    return max(0.05, min(1.0, exp(-float(hours_since_lab) * 0.69314718056 / half_life_hours)))


@dataclass
class EpisodeRecord:
    episode_id: str
    occurrence_count: int
    first_time: pd.Timestamp
    last_time: pd.Timestamp
    effective_end: Optional[pd.Timestamp]
    last_signature: Tuple[str, ...]


class EpisodeMemory:
    """Track signal episodes without treating recurrence as tolerance.

    A new episode begins after the previous implicated medication overlap has
    ended or at a new admission. Within an episode, a material state change is
    recorded separately so that a repeat can re-alert even when the rule ID is
    unchanged.
    """

    def __init__(self) -> None:
        self._active: Dict[Tuple[int, int, str], EpisodeRecord] = {}
        self._lifetime_counts: Dict[Tuple[int, str], int] = {}
        self._admission_counts: Dict[Tuple[int, int, str], int] = {}
        self._last_seen: Dict[Tuple[int, str], pd.Timestamp] = {}
        self._episode_sequence: Dict[Tuple[int, str], int] = {}

    def observe(
        self,
        subject_id: int,
        hadm_id: int,
        episode_key: str,
        timestamp: pd.Timestamp,
        effective_end: Optional[pd.Timestamp],
        material_signature: Tuple[str, ...],
    ) -> dict:
        patient_key = (int(subject_id), str(episode_key))
        admission_key = (int(subject_id), int(hadm_id), str(episode_key))
        active_key = admission_key
        prior_lifetime = self._lifetime_counts.get(patient_key, 0)
        prior_admission = self._admission_counts.get(admission_key, 0)
        last_seen = self._last_seen.get(patient_key)
        record = self._active.get(active_key)
        continuation = False
        if record is not None:
            if record.effective_end is None or pd.Timestamp(timestamp) <= record.effective_end:
                continuation = True
        if not continuation:
            sequence = self._episode_sequence.get(patient_key, 0) + 1
            self._episode_sequence[patient_key] = sequence
            episode_id = f"{subject_id}-{episode_key}-E{sequence}"
            occurrence_index = 1
            material_change = False
            record = EpisodeRecord(
                episode_id=episode_id,
                occurrence_count=1,
                first_time=pd.Timestamp(timestamp),
                last_time=pd.Timestamp(timestamp),
                effective_end=effective_end,
                last_signature=material_signature,
            )
            self._active[active_key] = record
            repeat_status = "new_episode"
        else:
            occurrence_index = record.occurrence_count + 1
            material_change = record.last_signature != material_signature
            repeat_status = (
                "repeat_with_material_change"
                if material_change
                else "repeat_unchanged_state"
            )
            record.occurrence_count = occurrence_index
            record.last_time = pd.Timestamp(timestamp)
            if effective_end is not None:
                if record.effective_end is None or effective_end > record.effective_end:
                    record.effective_end = effective_end
            record.last_signature = material_signature
            episode_id = record.episode_id
        self._lifetime_counts[patient_key] = prior_lifetime + 1
        self._admission_counts[admission_key] = prior_admission + 1
        self._last_seen[patient_key] = pd.Timestamp(timestamp)
        hours_since_prior = (
            None
            if last_seen is None
            else max(0.0, (pd.Timestamp(timestamp) - last_seen).total_seconds() / 3600.0)
        )
        return {
            "episode_id": episode_id,
            "episode_occurrence_index": occurrence_index,
            "prior_lifetime_occurrences": prior_lifetime,
            "prior_admission_occurrences": prior_admission,
            "hours_since_prior_same_signal": hours_since_prior,
            "material_change_since_prior": material_change,
            "repeat_status": repeat_status,
        }

class RenalTimeline:
    """Efficient per-patient creatinine lookup for repeated order timestamps."""

    def __init__(self, labs: pd.DataFrame) -> None:
        import numpy as np

        if labs.empty:
            self.times = np.array([], dtype="datetime64[ns]")
            self.values = np.array([], dtype=float)
            self.hadm_ids = np.array([], dtype=float)
            return
        working = labs.copy()
        working["charttime"] = pd.to_datetime(working["charttime"], errors="coerce")
        working["valuenum"] = pd.to_numeric(working["valuenum"], errors="coerce")
        working["hadm_id"] = pd.to_numeric(working.get("hadm_id"), errors="coerce")
        working = working.dropna(subset=["charttime", "valuenum"])
        working = working[working["valuenum"] > 0].sort_values("charttime", kind="stable")
        self.times = working["charttime"].to_numpy(dtype="datetime64[ns]")
        self.values = working["valuenum"].to_numpy(dtype=float)
        self.hadm_ids = working["hadm_id"].to_numpy(dtype=float)

    def snapshot(
        self,
        timestamp: pd.Timestamp,
        hadm_id: int,
        age: int,
        gender: Optional[str],
    ) -> RenalSnapshot:
        import numpy as np

        if self.times.size == 0:
            return renal_snapshot(pd.DataFrame(), timestamp, hadm_id, age, gender)
        ts64 = np.datetime64(pd.Timestamp(timestamp).to_datetime64())
        end = int(np.searchsorted(self.times, ts64, side="right"))
        if end == 0:
            return renal_snapshot(pd.DataFrame(), timestamp, hadm_id, age, gender)
        current_index = end - 1
        current_time = pd.Timestamp(self.times[current_index])
        current_value = float(self.values[current_index])
        current_hadm_value = self.hadm_ids[current_index]
        current_hadm = None if np.isnan(current_hadm_value) else int(current_hadm_value)
        hours_since = max(0.0, (pd.Timestamp(timestamp) - current_time).total_seconds() / 3600.0)

        def minimum_window(hours: float) -> tuple[Optional[float], Optional[pd.Timestamp]]:
            start64 = np.datetime64((current_time - pd.Timedelta(hours=hours)).to_datetime64())
            start = int(np.searchsorted(self.times, start64, side="left"))
            stop = current_index
            if stop <= start:
                return None, None
            window_values = self.values[start:stop]
            if window_values.size == 0:
                return None, None
            local = int(np.argmin(window_values))
            index = start + local
            return float(self.values[index]), pd.Timestamp(self.times[index])

        baseline_48h, baseline_48h_time = minimum_window(48.0)
        baseline_7d, baseline_7d_time = minimum_window(24.0 * 7.0)
        delta_48h = None if baseline_48h is None else current_value - baseline_48h
        ratio_7d = None if baseline_7d is None or baseline_7d <= 0 else current_value / baseline_7d
        acute_48h = bool(delta_48h is not None and delta_48h >= 0.3 - 1e-12)
        acute_7d = bool(ratio_7d is not None and ratio_7d >= 1.5 - 1e-12)
        if acute_48h and acute_7d:
            basis = "creatinine_increase_0.3_within_48h_and_1.5x_within_7d"
        elif acute_48h:
            basis = "creatinine_increase_0.3_within_48h"
        elif acute_7d:
            basis = "creatinine_increase_1.5x_within_7d"
        elif baseline_48h is None and baseline_7d is None:
            basis = "insufficient_acute_baseline"
        else:
            basis = "no_kdigo_aligned_acute_deterioration"
        return RenalSnapshot(
            current_creatinine=current_value,
            current_time=current_time,
            current_hadm_id=current_hadm,
            current_same_admission=(current_hadm == int(hadm_id)),
            hours_since_current=hours_since,
            baseline_48h_creatinine=baseline_48h,
            baseline_48h_time=baseline_48h_time,
            baseline_7d_creatinine=baseline_7d,
            baseline_7d_time=baseline_7d_time,
            creatinine_delta_48h=delta_48h,
            creatinine_ratio_7d=ratio_7d,
            acute_deterioration_48h=acute_48h,
            acute_deterioration_7d=acute_7d,
            acute_renal_deterioration=(acute_48h or acute_7d),
            acute_deterioration_basis=basis,
            egfr_current=compute_egfr_ckdepi2021(current_value, age, gender),
        )
