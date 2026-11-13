from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import pandas as pd

from .crrt import get_icu_crrt_signal_at_time, load_icu_crrt_intervals
from .detector import DeterministicDetector
from .models import ActiveMedication, OrderEvent
from .reference import ReferenceStore
from .temporal import RenalTimeline
from .transplant_phase import (
    PERI_TRANSPLANT_PHASES,
    classify_transplant_phase,
    load_transplant_episode_assignments,
    post_transplant_phase_context_allowed,
    protocol_phase_for_context,
)

CREATININE_ITEMID = 50912
ProgressCallback = Callable[[int, int, int], None]


def load_cohort(path: str | Path, expected_patients: Optional[int] = None) -> pd.DataFrame:
    cohort = pd.read_csv(path)
    required = {"subject_id", "hadm_id"}
    missing = required - set(cohort.columns)
    if missing:
        raise ValueError(f"Cohort file missing columns: {sorted(missing)}")
    for column in required:
        cohort[column] = pd.to_numeric(cohort[column], errors="coerce")
    cohort = cohort.dropna(subset=list(required)).copy()
    cohort["subject_id"] = cohort["subject_id"].astype(int)
    cohort["hadm_id"] = cohort["hadm_id"].astype(int)
    cohort = cohort.drop_duplicates(["subject_id", "hadm_id"]).reset_index(drop=True)
    if expected_patients is not None and cohort["subject_id"].nunique() != expected_patients:
        raise ValueError(
            f"Expected {expected_patients} patients, found {cohort['subject_id'].nunique()}"
        )
    return cohort


def _read_required(path: str | Path, required: set[str], low_memory: bool = False) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    header = pd.read_csv(source, nrows=0)
    missing = required - set(header.columns)
    if missing:
        raise ValueError(f"{source.name} missing required columns: {sorted(missing)}")
    return pd.read_csv(source, low_memory=low_memory)


def _safe_text(value: object) -> Optional[str]:
    return str(value) if pd.notna(value) and str(value).strip() else None


def _safe_float(value: object) -> Optional[float]:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else float(parsed)


def _age_by_admission(
    cohort: pd.DataFrame,
    patients_file: str | Path,
    admissions_file: str | Path,
) -> pd.DataFrame:
    patients = _read_required(
        patients_file, {"subject_id", "anchor_age", "anchor_year", "gender"}
    )
    admissions = _read_required(
        admissions_file,
        {"subject_id", "hadm_id", "admittime", "dischtime"},
    )
    patients["subject_id"] = pd.to_numeric(patients["subject_id"], errors="coerce")
    admissions["subject_id"] = pd.to_numeric(admissions["subject_id"], errors="coerce")
    admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="coerce")
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
    patients = patients.dropna(subset=["subject_id", "anchor_age", "anchor_year"]).copy()
    admissions = admissions.dropna(subset=["subject_id", "hadm_id", "admittime"]).copy()
    patients["subject_id"] = patients["subject_id"].astype(int)
    admissions["subject_id"] = admissions["subject_id"].astype(int)
    admissions["hadm_id"] = admissions["hadm_id"].astype(int)
    merged = (
        cohort.merge(admissions, on=["subject_id", "hadm_id"], how="left", validate="one_to_one")
        .merge(patients, on="subject_id", how="left", validate="many_to_one")
    )
    required = ["admittime", "anchor_age", "anchor_year", "gender"]
    if merged[required].isna().any(axis=1).any():
        bad = int(merged[required].isna().any(axis=1).sum())
        raise ValueError(f"Unable to derive age/gender for {bad} cohort admission(s)")
    merged["age"] = (
        pd.to_numeric(merged["anchor_age"], errors="raise")
        + merged["admittime"].dt.year
        - pd.to_numeric(merged["anchor_year"], errors="raise")
    ).round().astype(int).clip(lower=0)
    return merged[
        ["subject_id", "hadm_id", "age", "gender", "admittime", "dischtime"]
    ]


class IntervalLookup:
    def __init__(
        self,
        path: str | Path | None,
        value_column: str,
        start_column: str,
        end_column: str,
    ) -> None:
        self.by_admission: Dict[int, pd.DataFrame] = {}
        self.value_column = value_column
        if path is None or not Path(path).exists():
            return
        header = pd.read_csv(path, nrows=0)
        required = {"hadm_id", value_column, start_column, end_column}
        if not required.issubset(header.columns):
            return
        frame = pd.read_csv(path, usecols=list(required), low_memory=False)
        frame["hadm_id"] = pd.to_numeric(frame["hadm_id"], errors="coerce")
        frame[start_column] = pd.to_datetime(frame[start_column], errors="coerce")
        frame[end_column] = pd.to_datetime(frame[end_column], errors="coerce")
        frame = frame.dropna(subset=["hadm_id", start_column]).copy()
        frame["hadm_id"] = frame["hadm_id"].astype(int)
        frame[end_column] = frame[end_column].fillna(pd.Timestamp.max)
        frame = frame.rename(columns={start_column: "_start", end_column: "_end"})
        self.by_admission = {
            int(hadm): group.sort_values("_start", kind="stable")
            for hadm, group in frame.groupby("hadm_id")
        }

    def at(self, hadm_id: int, timestamp: pd.Timestamp) -> Optional[str]:
        group = self.by_admission.get(int(hadm_id))
        if group is None or group.empty:
            return None
        matches = group[(group["_start"] <= timestamp) & (group["_end"] > timestamp)]
        if matches.empty:
            return None
        return _safe_text(matches.iloc[-1][self.value_column])


class ServiceLookup:
    def __init__(self, path: str | Path | None) -> None:
        self.by_admission: Dict[int, pd.DataFrame] = {}
        if path is None or not Path(path).exists():
            return
        header = pd.read_csv(path, nrows=0)
        required = {"hadm_id", "transfertime", "curr_service"}
        if not required.issubset(header.columns):
            return
        frame = pd.read_csv(path, usecols=list(required), low_memory=False)
        frame["hadm_id"] = pd.to_numeric(frame["hadm_id"], errors="coerce")
        frame["transfertime"] = pd.to_datetime(frame["transfertime"], errors="coerce")
        frame = frame.dropna(subset=["hadm_id", "transfertime"]).copy()
        frame["hadm_id"] = frame["hadm_id"].astype(int)
        self.by_admission = {
            int(hadm): group.sort_values("transfertime", kind="stable")
            for hadm, group in frame.groupby("hadm_id")
        }

    def at(self, hadm_id: int, timestamp: pd.Timestamp) -> Optional[str]:
        group = self.by_admission.get(int(hadm_id))
        if group is None or group.empty:
            return None
        matches = group[group["transfertime"] <= timestamp]
        if matches.empty:
            return None
        return _safe_text(matches.iloc[-1]["curr_service"])


def build_order_timeline(
    cohort_file: str | Path,
    patients_file: str | Path,
    admissions_file: str | Path,
    prescriptions_file: str | Path,
    labevents_file: str | Path,
    procedures_icd_file: str | Path | None,
    transfers_file: str | Path | None,
    services_file: str | Path | None,
    procedureevents_file: str | Path | None,
    reference: ReferenceStore,
    expected_patients: Optional[int] = None,
    progress_callback: Optional[ProgressCallback] = None,
    progress_every: int = 5000,
    poe_file: str | Path | None = None,
    decision_time_mode: str = "prescription_starttime",
) -> pd.DataFrame:
    cohort = load_cohort(cohort_file, expected_patients)
    admission_context = _age_by_admission(cohort, patients_file, admissions_file)
    context_lookup = admission_context.set_index("hadm_id")
    subjects = set(cohort["subject_id"])
    admissions = set(cohort["hadm_id"])
    episode_assignments = load_transplant_episode_assignments(
        cohort, procedures_icd_file, admissions_file
    )

    rx = _read_required(
        prescriptions_file,
        {"subject_id", "hadm_id", "drug", "starttime", "stoptime", "route"},
        low_memory=True,
    )
    rx["subject_id"] = pd.to_numeric(rx["subject_id"], errors="coerce")
    rx["hadm_id"] = pd.to_numeric(rx["hadm_id"], errors="coerce")
    rx = rx.dropna(subset=["subject_id", "hadm_id"]).copy()
    rx["subject_id"] = rx["subject_id"].astype(int)
    rx["hadm_id"] = rx["hadm_id"].astype(int)
    rx = rx[rx["hadm_id"].isin(admissions)].copy()
    rx["starttime"] = pd.to_datetime(rx["starttime"], errors="coerce")
    rx["stoptime"] = pd.to_datetime(rx["stoptime"], errors="coerce")
    rx = rx.dropna(subset=["starttime"]).copy()
    rx["source_rx_row_index"] = rx.index.astype(int)

    if "poe_id" not in rx.columns:
        rx["poe_id"] = pd.NA
    rx["poe_key"] = rx["poe_id"].astype("string").str.strip()
    rx.loc[rx["poe_key"].isin(["", "<NA>", "nan", "None"]), "poe_key"] = pd.NA
    rx["provider_ordertime"] = pd.NaT
    rx["poe_link_status"] = "not_requested"

    if decision_time_mode == "poe_ordertime_with_starttime_fallback":
        if poe_file is None or not Path(poe_file).exists():
            raise FileNotFoundError(
                "A valid poe.csv path is required for provider-order-time replay"
            )
        poe = _read_required(
            poe_file, {"subject_id", "hadm_id", "poe_id", "ordertime"}, low_memory=True
        )
        poe["subject_id"] = pd.to_numeric(poe["subject_id"], errors="coerce")
        poe["hadm_id"] = pd.to_numeric(poe["hadm_id"], errors="coerce")
        poe["ordertime"] = pd.to_datetime(poe["ordertime"], errors="coerce")
        poe["poe_key"] = poe["poe_id"].astype("string").str.strip()
        poe = poe.dropna(subset=["subject_id", "hadm_id", "poe_key", "ordertime"]).copy()
        poe["subject_id"] = poe["subject_id"].astype(int)
        poe["hadm_id"] = poe["hadm_id"].astype(int)
        if poe.duplicated(["subject_id", "hadm_id", "poe_key"], keep=False).any():
            duplicate = poe[poe.duplicated(["subject_id", "hadm_id", "poe_key"], keep=False)]
            varying = duplicate.groupby(["subject_id", "hadm_id", "poe_key"])["ordertime"].nunique()
            if (varying > 1).any():
                raise ValueError("A provider-order key maps to multiple ordertimes")
            poe = poe.drop_duplicates(["subject_id", "hadm_id", "poe_key"], keep="first")
        known_poe_keys = set(poe["poe_key"].dropna().astype(str))
        rx = rx.merge(
            poe[["subject_id", "hadm_id", "poe_key", "ordertime"]].rename(
                columns={"ordertime": "_matched_ordertime"}
            ),
            on=["subject_id", "hadm_id", "poe_key"],
            how="left",
            validate="many_to_one",
        )
        exact = rx["_matched_ordertime"].notna()
        missing = rx["poe_key"].isna()
        known_elsewhere = (~missing) & (~exact) & rx["poe_key"].astype(str).isin(known_poe_keys)
        rx.loc[exact, "poe_link_status"] = "exact_subject_admission_order_match"
        rx.loc[missing, "poe_link_status"] = "missing_poe_id"
        rx.loc[known_elsewhere, "poe_link_status"] = "poe_id_found_other_admission"
        rx.loc[(~missing) & (~exact) & (~known_elsewhere), "poe_link_status"] = "poe_id_not_found"
        rx.loc[exact, "provider_ordertime"] = rx.loc[exact, "_matched_ordertime"]
        rx = rx.drop(columns=["_matched_ordertime"])
    elif decision_time_mode == "prescription_starttime":
        rx.loc[rx["poe_key"].notna(), "poe_link_status"] = "poe_id_unvalidated_starttime_mode"
        rx.loc[rx["poe_key"].isna(), "poe_link_status"] = "missing_poe_id"
    else:
        raise ValueError(f"Unsupported decision_time_mode: {decision_time_mode}")

    exact_poe = rx["poe_link_status"].eq("exact_subject_admission_order_match")
    if decision_time_mode == "poe_ordertime_with_starttime_fallback":
        rx["decision_timestamp"] = rx["provider_ordertime"].where(exact_poe, rx["starttime"])
        rx["decision_time_source"] = "prescription_starttime_fallback"
        rx.loc[exact_poe, "decision_time_source"] = "poe_ordertime"
    else:
        rx["decision_timestamp"] = rx["starttime"]
        rx["decision_time_source"] = "prescription_starttime"
    rx["exposure_starttime"] = rx["starttime"]

    labs = _read_required(
        labevents_file,
        {"subject_id", "hadm_id", "itemid", "charttime", "valuenum"},
        low_memory=True,
    )
    labs["subject_id"] = pd.to_numeric(labs["subject_id"], errors="coerce")
    labs["itemid"] = pd.to_numeric(labs["itemid"], errors="coerce")
    labs = labs[labs["subject_id"].isin(subjects) & labs["itemid"].eq(CREATININE_ITEMID)].copy()
    labs["subject_id"] = labs["subject_id"].astype(int)
    renal_timelines = {
        int(sid): RenalTimeline(group)
        for sid, group in labs.groupby("subject_id")
    }

    crrt_intervals = {}
    if procedureevents_file is not None and Path(procedureevents_file).exists():
        crrt_intervals = load_icu_crrt_intervals(str(procedureevents_file))
    care_lookup = IntervalLookup(
        transfers_file, "careunit", "intime", "outtime"
    )
    service_lookup = ServiceLookup(services_file)

    rows: list[dict] = []
    ordered_rx = rx.sort_values(
        ["subject_id", "decision_timestamp", "hadm_id", "source_rx_row_index"], kind="stable"
    )
    total = len(ordered_rx)
    for position, row in enumerate(ordered_rx.itertuples(index=False), start=1):
        sid = int(row.subject_id)
        hadm = int(row.hadm_id)
        if hadm not in context_lookup.index:
            continue
        age = int(context_lookup.loc[hadm, "age"])
        gender = _safe_text(context_lookup.loc[hadm, "gender"])
        timestamp = pd.Timestamp(row.decision_timestamp)
        exposure_start = pd.Timestamp(row.exposure_starttime)
        dischtime = context_lookup.loc[hadm, "dischtime"]
        effective_stop = (
            pd.Timestamp(row.stoptime)
            if pd.notna(row.stoptime)
            else (pd.Timestamp(dischtime) if pd.notna(dischtime) else None)
        )
        renal = renal_timelines.get(sid, RenalTimeline(pd.DataFrame())).snapshot(
            timestamp, hadm, age, gender
        )
        crrt = get_icu_crrt_signal_at_time(crrt_intervals, sid, hadm, timestamp)
        source_index = int(row.source_rx_row_index)
        poe_id = getattr(row, "poe_key", None)
        valid_provider_order = str(row.poe_link_status) in {
            "exact_subject_admission_order_match",
            "poe_id_unvalidated_starttime_mode",
        } and pd.notna(poe_id) and str(poe_id).strip()
        base_order_id = str(poe_id) if valid_provider_order else "RXROW"
        order_id = f"{sid}-{hadm}-{base_order_id}-{source_index}"
        bundle_id = (
            f"{sid}-{hadm}-POE-{base_order_id}"
            if base_order_id != "RXROW"
            else f"{sid}-{hadm}-RXROW-{source_index}"
        )
        classification = reference.classify(row.drug)
        anchor = episode_assignments[hadm]
        phase, days_since = classify_transplant_phase(timestamp, hadm, anchor)
        post_phase_allowed = post_transplant_phase_context_allowed(phase, anchor)
        protocol_phase = protocol_phase_for_context(phase, post_phase_allowed)
        peri_transplant_order = phase in PERI_TRANSPLANT_PHASES
        rows.append(
            {
                "subject_id": sid,
                "hadm_id": hadm,
                "order_id": order_id,
                "bundle_id": bundle_id,
                "source_rx_row_index": source_index,
                "timestamp": timestamp,
                "decision_timestamp": timestamp,
                "decision_time_source": str(row.decision_time_source),
                "provider_ordertime": (pd.Timestamp(row.provider_ordertime) if pd.notna(row.provider_ordertime) else pd.NaT),
                "poe_link_status": str(row.poe_link_status),
                "source_poe_id": (_safe_text(getattr(row, "poe_id", None))),
                "exposure_starttime": exposure_start,
                "stoptime": effective_stop,
                "drug": str(row.drug),
                "ingredient_key": classification.ingredient_key,
                "drug_category": classification.category,
                "route": _safe_text(getattr(row, "route", None)),
                "dose_val_rx": _safe_float(getattr(row, "dose_val_rx", None)),
                "dose_unit_rx": _safe_text(getattr(row, "dose_unit_rx", None)),
                "frequency": _safe_text(getattr(row, "doses_per_24_hrs", None)),
                "age": age,
                "gender": gender,
                "admittime": context_lookup.loc[hadm, "admittime"],
                "dischtime": dischtime,
                "careunit": care_lookup.at(hadm, timestamp),
                "clinical_service": service_lookup.at(hadm, timestamp),
                "transplant_date": anchor.transplant_date,
                "transplant_date_source": anchor.source,
                "transplant_procedure_code": anchor.procedure_code,
                "transplant_procedure_version": anchor.procedure_version,
                "transplant_episode_id": anchor.transplant_episode_id,
                "transplant_episode_number": anchor.episode_number,
                "transplant_hadm_id": anchor.transplant_hadm_id,
                "transplant_anchor_quality": anchor.anchor_quality,
                "transplant_anchor_quality_level": anchor.anchor_quality_level,
                # Admission-level permission retained for audit.
                "episode_phase_specific_context_allowed": (
                    anchor.phase_specific_context_allowed
                ),
                # Backward-compatible field now means post-transplant phase
                # narrowing is permitted for this exact order.
                "phase_specific_context_allowed": post_phase_allowed,
                "post_transplant_phase_context_allowed": post_phase_allowed,
                "peri_transplant_order": peri_transplant_order,
                "transplant_protocol_phase": protocol_phase,
                "short_gap_repeat_transplant_candidate": (
                    anchor.short_gap_repeat_transplant_candidate
                ),
                "transplant_phase": phase,
                "days_since_transplant": days_since,
                "current_creatinine": renal.current_creatinine,
                "current_creatinine_time": renal.current_time,
                "current_creatinine_hadm_id": renal.current_hadm_id,
                "current_creatinine_same_admission": renal.current_same_admission,
                "hours_since_current_creatinine": renal.hours_since_current,
                "baseline_48h_creatinine": renal.baseline_48h_creatinine,
                "baseline_48h_time": renal.baseline_48h_time,
                "baseline_7d_creatinine": renal.baseline_7d_creatinine,
                "baseline_7d_time": renal.baseline_7d_time,
                "creatinine_delta_48h": renal.creatinine_delta_48h,
                "creatinine_ratio_7d": renal.creatinine_ratio_7d,
                "acute_deterioration_48h": renal.acute_deterioration_48h,
                "acute_deterioration_7d": renal.acute_deterioration_7d,
                "acute_renal_deterioration": renal.acute_renal_deterioration,
                "acute_deterioration_basis": renal.acute_deterioration_basis,
                "egfr_current": renal.egfr_current,
                "renal_metric_method": "CKD-EPI-2021 eGFR screening proxy",
                **crrt,
            }
        )
        if progress_callback and (position % progress_every == 0 or position == total):
            progress_callback(position, total, len(rows))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["subject_id", "timestamp", "hadm_id", "source_rx_row_index"], kind="stable"
    ).reset_index(drop=True)


def _active_medication(row: pd.Series) -> Optional[ActiveMedication]:
    if str(row.get("drug_category")) != "DRUG" or not row.get("ingredient_key"):
        return None
    return ActiveMedication(
        raw_name=str(row["drug"]),
        ingredient_key=str(row["ingredient_key"]),
        start_time=pd.Timestamp(row.get("exposure_starttime", row["timestamp"])),
        stop_time=(pd.Timestamp(row["stoptime"]) if pd.notna(row.get("stoptime")) else None),
        route=_safe_text(row.get("route")),
        dose=_safe_float(row.get("dose_val_rx")),
        dose_unit=_safe_text(row.get("dose_unit_rx")),
    )


def _intervals_overlap(
    first_start: pd.Timestamp,
    first_stop: Optional[pd.Timestamp],
    second_start: pd.Timestamp,
    second_stop: Optional[pd.Timestamp],
) -> bool:
    if first_stop is not None and first_stop <= second_start:
        return False
    if second_stop is not None and second_stop <= first_start:
        return False
    return True


def _overlapping_medications(
    medications: Iterable[ActiveMedication],
    exposure_start: pd.Timestamp,
    exposure_stop: Optional[pd.Timestamp],
) -> list[ActiveMedication]:
    return [
        medication
        for medication in medications
        if _intervals_overlap(
            medication.start_time,
            medication.stop_time,
            exposure_start,
            exposure_stop,
        )
    ]


def build_order_active_snapshots(
    orders: pd.DataFrame,
    relevant_order_ids: Optional[set[str]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    progress_every: int = 5000,
) -> Dict[str, tuple[ActiveMedication, ...]]:
    """Build prior-known, planned-overlap medication snapshots.

    Provider ordertime determines what was known at the decision. Prescription
    start/stop times determine whether medication exposures are planned to
    overlap. Separate provider orders with the same ordertime are not given an
    invented sequence. Rows within the same provider-order bundle are evaluated
    in stable source-row order so a pair can fire once rather than twice.
    """
    snapshots: Dict[str, tuple[ActiveMedication, ...]] = {}
    if orders.empty:
        return snapshots
    total = len(orders)
    processed = 0
    relevant = relevant_order_ids
    ordered = orders.sort_values(
        ["hadm_id", "timestamp", "bundle_id", "source_rx_row_index"], kind="stable"
    )
    for _, admission in ordered.groupby("hadm_id", sort=False):
        known_prior: list[ActiveMedication] = []
        for _, time_group in admission.groupby("timestamp", sort=False):
            pending_time_medications: list[ActiveMedication] = []
            for _, bundle_group in time_group.groupby("bundle_id", sort=False):
                prior_within_bundle: list[ActiveMedication] = []
                for _, row in bundle_group.sort_values("source_rx_row_index", kind="stable").iterrows():
                    exposure_start = pd.Timestamp(row.get("exposure_starttime", row["timestamp"]))
                    exposure_stop = (
                        pd.Timestamp(row["stoptime"])
                        if pd.notna(row.get("stoptime"))
                        else None
                    )
                    order_id = str(row["order_id"])
                    if relevant is None or order_id in relevant:
                        overlapping = _overlapping_medications(
                            known_prior + prior_within_bundle,
                            exposure_start,
                            exposure_stop,
                        )
                        snapshots[order_id] = tuple(overlapping)
                    medication = _active_medication(row)
                    if medication is not None:
                        prior_within_bundle.append(medication)
                        pending_time_medications.append(medication)
                    processed += 1
                    if progress_callback and (
                        processed % progress_every == 0 or processed == total
                    ):
                        progress_callback(processed, total, len(snapshots))
            known_prior.extend(pending_time_medications)
    return snapshots


def detect_signals_from_orders(
    orders: pd.DataFrame,
    reference: ReferenceStore,
    progress_callback: Optional[ProgressCallback] = None,
    progress_every: int = 5000,
) -> pd.DataFrame:
    """Evaluate every provider-order decision using planned exposure overlap."""
    if orders.empty:
        return pd.DataFrame()
    detector = DeterministicDetector(reference)
    snapshots = build_order_active_snapshots(orders)
    rows: list[dict] = []
    total = len(orders)
    ordered = orders.sort_values(
        ["hadm_id", "timestamp", "bundle_id", "source_rx_row_index"], kind="stable"
    )
    for position, (_, row) in enumerate(ordered.iterrows(), start=1):
        timestamp = pd.Timestamp(row["timestamp"])
        exposure_start = pd.Timestamp(row.get("exposure_starttime", timestamp))
        exposure_stop = (
            pd.Timestamp(row["stoptime"]) if pd.notna(row.get("stoptime")) else None
        )
        order = OrderEvent(
            subject_id=int(row["subject_id"]),
            hadm_id=int(row["hadm_id"]),
            order_id=str(row["order_id"]),
            bundle_id=str(row["bundle_id"]),
            raw_name=str(row["drug"]),
            start_time=timestamp,
            stop_time=exposure_stop,
            route=_safe_text(row.get("route")),
            dose=_safe_float(row.get("dose_val_rx")),
            dose_unit=_safe_text(row.get("dose_unit_rx")),
            frequency=_safe_text(row.get("frequency")),
            source_row_index=int(row["source_rx_row_index"]),
        )
        signals = detector.detect(
            order,
            snapshots.get(str(row["order_id"]), ()),
            patient_age=int(row["age"]),
            renal_metric=_safe_float(row.get("egfr_current")),
        )
        for signal in signals:
            record = signal.as_dict()
            for column in (
                "stoptime", "exposure_starttime", "decision_timestamp",
                "decision_time_source", "provider_ordertime", "poe_link_status",
                "source_poe_id", "transplant_phase", "days_since_transplant",
                "transplant_date_source", "transplant_episode_id",
                "transplant_episode_number", "transplant_hadm_id",
                "transplant_anchor_quality", "transplant_anchor_quality_level",
                "episode_phase_specific_context_allowed",
                "phase_specific_context_allowed",
                "post_transplant_phase_context_allowed",
                "peri_transplant_order", "transplant_protocol_phase",
                "short_gap_repeat_transplant_candidate", "current_creatinine",
                "hours_since_current_creatinine", "acute_renal_deterioration",
                "acute_deterioration_basis", "crrt_active", "careunit",
                "clinical_service", "route", "dose_val_rx", "dose_unit_rx",
                "frequency", "current_creatinine_same_admission",
            ):
                record[column] = row.get(column)
            record["new_drug_start_time"] = exposure_start
            rows.append(record)
        if progress_callback and (position % progress_every == 0 or position == total):
            progress_callback(position, total, len(rows))
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    duplicate = result.duplicated(["order_id", "rule_id"])
    if duplicate.any():
        examples = result.loc[duplicate, ["order_id", "rule_id"]].head().to_dict("records")
        raise ValueError(f"Duplicate order_id/rule_id firings detected: {examples}")
    return result.sort_values(
        ["subject_id", "timestamp", "hadm_id", "order_id", "rule_id"], kind="stable"
    ).reset_index(drop=True)

