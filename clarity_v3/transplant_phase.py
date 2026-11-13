from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# ICD-9-CM 55.69: other kidney transplantation.
# ICD-10-PCS codes below cover right/left kidney transplantation with
# allogeneic, syngeneic, or zooplastic qualifiers. The prototype reports the
# exact code and never infers donor type beyond the code itself.
ICD9_KIDNEY_TRANSPLANT = {"5569"}
ICD10_KIDNEY_TRANSPLANT = {
    "0TY00Z0", "0TY00Z1", "0TY00Z2",
    "0TY10Z0", "0TY10Z1", "0TY10Z2",
}

EPISODE_COHORT_COLUMNS = {
    "transplant_episode_id",
    "episode_number",
    "transplant_hadm_id",
    "transplant_date",
    "anchor_quality",
    "anchor_quality_level",
    "phase_specific_context_allowed",
    "short_gap_repeat_transplant_candidate",
}


@dataclass(frozen=True)
class TransplantAnchor:
    subject_id: int
    transplant_date: Optional[pd.Timestamp]
    transplant_hadm_id: Optional[int]
    source: str
    procedure_code: Optional[str]
    procedure_version: Optional[int]
    transplant_episode_id: Optional[str] = None
    episode_number: Optional[int] = None
    anchor_quality: str = "not_applicable"
    anchor_quality_level: str = "unknown"
    phase_specific_context_allowed: bool = True
    short_gap_repeat_transplant_candidate: bool = False


def normalize_icd_code(value: object) -> str:
    return str(value).replace(".", "").replace(" ", "").upper().strip()


def is_kidney_transplant_procedure(code: object, version: object) -> bool:
    normalized = normalize_icd_code(code)
    try:
        version_int = int(version)
    except (TypeError, ValueError):
        return False
    if version_int == 9:
        return normalized in ICD9_KIDNEY_TRANSPLANT
    if version_int == 10:
        return normalized in ICD10_KIDNEY_TRANSPLANT
    return False


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _load_admissions(admissions_file: str | Path) -> pd.DataFrame:
    admissions = pd.read_csv(
        admissions_file,
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
    )
    admissions["subject_id"] = pd.to_numeric(admissions["subject_id"], errors="coerce")
    admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="coerce")
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
    admissions = admissions.dropna(subset=["subject_id", "hadm_id", "admittime"])
    admissions = admissions.astype({"subject_id": "int64", "hadm_id": "int64"})
    if admissions["hadm_id"].duplicated().any():
        raise ValueError("admissions_file contains duplicate hadm_id values")
    return admissions


def _procedure_metadata_by_admission(
    procedures_icd_file: str | Path | None,
    cohort_subjects: set[int],
) -> dict[int, tuple[Optional[str], Optional[int]]]:
    if procedures_icd_file is None or not Path(procedures_icd_file).exists():
        return {}
    header = pd.read_csv(procedures_icd_file, nrows=0)
    required = {"subject_id", "hadm_id", "icd_code", "icd_version"}
    missing = required - set(header.columns)
    if missing:
        raise ValueError(f"procedures_icd missing columns: {sorted(missing)}")
    procedures = pd.read_csv(
        procedures_icd_file,
        usecols=list(required),
        low_memory=False,
    )
    procedures["subject_id"] = pd.to_numeric(procedures["subject_id"], errors="coerce")
    procedures["hadm_id"] = pd.to_numeric(procedures["hadm_id"], errors="coerce")
    procedures = procedures.dropna(subset=["subject_id", "hadm_id"]).copy()
    procedures = procedures.astype({"subject_id": "int64", "hadm_id": "int64"})
    procedures = procedures[procedures["subject_id"].isin(cohort_subjects)].copy()
    procedures["is_kidney_transplant"] = [
        is_kidney_transplant_procedure(code, version)
        for code, version in zip(procedures["icd_code"], procedures["icd_version"])
    ]
    procedures = procedures[procedures["is_kidney_transplant"]].copy()
    result: dict[int, tuple[Optional[str], Optional[int]]] = {}
    for hadm_id, group in procedures.groupby("hadm_id", sort=False):
        row = group.iloc[0]
        result[int(hadm_id)] = (
            normalize_icd_code(row["icd_code"]),
            int(row["icd_version"]),
        )
    return result


def load_transplant_anchors(
    cohort: pd.DataFrame,
    procedures_icd_file: str | Path | None,
    admissions_file: str | Path,
) -> Dict[int, TransplantAnchor]:
    """Legacy one-anchor-per-patient loader.

    This remains available for the frozen synthetic demonstration and for
    backward-compatible configurations that do not contain episode metadata.
    Episode-based real cohorts should use ``load_transplant_episode_assignments``.
    """
    admissions = _load_admissions(admissions_file)
    admission_lookup = admissions.set_index("hadm_id")["admittime"].to_dict()
    anchors: Dict[int, TransplantAnchor] = {
        int(sid): TransplantAnchor(
            subject_id=int(sid),
            transplant_date=None,
            transplant_hadm_id=None,
            source="prevalent_or_unknown_no_observed_procedure",
            procedure_code=None,
            procedure_version=None,
            phase_specific_context_allowed=False,
        )
        for sid in cohort["subject_id"].drop_duplicates()
    }
    if procedures_icd_file is None or not Path(procedures_icd_file).exists():
        return anchors
    header = pd.read_csv(procedures_icd_file, nrows=0)
    required = {"subject_id", "hadm_id", "icd_code", "icd_version"}
    missing = required - set(header.columns)
    if missing:
        raise ValueError(f"procedures_icd missing columns: {sorted(missing)}")
    usecols = list(required | ({"chartdate"} if "chartdate" in header.columns else set()))
    procedures = pd.read_csv(procedures_icd_file, usecols=usecols, low_memory=False)
    procedures["subject_id"] = pd.to_numeric(procedures["subject_id"], errors="coerce")
    procedures["hadm_id"] = pd.to_numeric(procedures["hadm_id"], errors="coerce")
    procedures = procedures.dropna(subset=["subject_id", "hadm_id"])
    procedures = procedures.astype({"subject_id": "int64", "hadm_id": "int64"})
    cohort_subjects = set(int(value) for value in cohort["subject_id"])
    procedures = procedures[procedures["subject_id"].isin(cohort_subjects)].copy()
    procedures["is_kidney_transplant"] = [
        is_kidney_transplant_procedure(code, version)
        for code, version in zip(procedures["icd_code"], procedures["icd_version"])
    ]
    procedures = procedures[procedures["is_kidney_transplant"]].copy()
    if procedures.empty:
        return anchors
    if "chartdate" in procedures.columns:
        procedures["event_date"] = pd.to_datetime(procedures["chartdate"], errors="coerce")
    else:
        procedures["event_date"] = pd.NaT
    procedures["event_date"] = procedures.apply(
        lambda row: row["event_date"]
        if pd.notna(row["event_date"])
        else admission_lookup.get(int(row["hadm_id"])),
        axis=1,
    )
    procedures = procedures.dropna(subset=["event_date"]).sort_values(
        ["subject_id", "event_date", "hadm_id"], kind="stable"
    )
    for sid, group in procedures.groupby("subject_id", sort=False):
        row = group.iloc[0]
        anchors[int(sid)] = TransplantAnchor(
            subject_id=int(sid),
            transplant_date=pd.Timestamp(row["event_date"]),
            transplant_hadm_id=int(row["hadm_id"]),
            source="observed_kidney_transplant_procedure_icd_legacy_first_anchor",
            procedure_code=normalize_icd_code(row["icd_code"]),
            procedure_version=int(row["icd_version"]),
            transplant_episode_id=f"{int(sid)}-TX1",
            episode_number=1,
            anchor_quality="legacy_unclassified",
            anchor_quality_level="unknown",
            phase_specific_context_allowed=True,
        )
    return anchors


def load_transplant_episode_assignments(
    cohort: pd.DataFrame,
    procedures_icd_file: str | Path | None,
    admissions_file: str | Path,
) -> Dict[int, TransplantAnchor]:
    """Return the transplant episode assigned to each cohort admission.

    Real Phase-1 longitudinal cohorts created by Stage 00A.3 carry explicit
    episode metadata. Each admission is therefore evaluated relative to the
    most recent preceding transplant episode, allowing the transplant clock to
    reset after retransplantation. Legacy cohort files without these columns
    retain the historical one-anchor-per-patient behavior for compatibility.
    """
    if not EPISODE_COHORT_COLUMNS.issubset(cohort.columns):
        legacy = load_transplant_anchors(cohort, procedures_icd_file, admissions_file)
        return {
            int(row.hadm_id): legacy[int(row.subject_id)]
            for row in cohort[["subject_id", "hadm_id"]].itertuples(index=False)
        }

    admissions = _load_admissions(admissions_file)
    admission_lookup = admissions.set_index("hadm_id")
    cohort_subjects = set(int(value) for value in cohort["subject_id"])
    procedure_lookup = _procedure_metadata_by_admission(
        procedures_icd_file,
        cohort_subjects,
    )

    frame = cohort.copy()
    frame["subject_id"] = pd.to_numeric(frame["subject_id"], errors="coerce")
    frame["hadm_id"] = pd.to_numeric(frame["hadm_id"], errors="coerce")
    frame["transplant_hadm_id"] = pd.to_numeric(
        frame["transplant_hadm_id"], errors="coerce"
    )
    frame["episode_number"] = pd.to_numeric(frame["episode_number"], errors="coerce")
    frame["transplant_date"] = pd.to_datetime(frame["transplant_date"], errors="coerce")
    required_nonmissing = [
        "subject_id", "hadm_id", "transplant_hadm_id", "episode_number",
        "transplant_date", "transplant_episode_id",
    ]
    if frame[required_nonmissing].isna().any(axis=1).any():
        bad = int(frame[required_nonmissing].isna().any(axis=1).sum())
        raise ValueError(f"Episode-based cohort contains {bad} incomplete assignment row(s)")
    frame = frame.astype(
        {
            "subject_id": "int64",
            "hadm_id": "int64",
            "transplant_hadm_id": "int64",
            "episode_number": "int64",
        }
    )
    if frame["hadm_id"].duplicated().any():
        raise ValueError("Episode-based cohort contains duplicate hadm_id assignments")

    assignments: Dict[int, TransplantAnchor] = {}
    for row in frame.itertuples(index=False):
        hadm_id = int(row.hadm_id)
        transplant_hadm_id = int(row.transplant_hadm_id)
        subject_id = int(row.subject_id)
        if hadm_id not in admission_lookup.index:
            raise ValueError(f"Cohort hadm_id {hadm_id} is absent from admissions.csv")
        if transplant_hadm_id not in admission_lookup.index:
            raise ValueError(
                f"Transplant anchor hadm_id {transplant_hadm_id} is absent from admissions.csv"
            )
        anchor_subject = int(admission_lookup.loc[transplant_hadm_id, "subject_id"])
        if anchor_subject != subject_id:
            raise ValueError(
                f"Transplant anchor admission {transplant_hadm_id} belongs to subject "
                f"{anchor_subject}, not cohort subject {subject_id}"
            )
        current_subject = int(admission_lookup.loc[hadm_id, "subject_id"])
        if current_subject != subject_id:
            raise ValueError(
                f"Cohort admission {hadm_id} belongs to subject {current_subject}, "
                f"not assigned subject {subject_id}"
            )
        procedure_code, procedure_version = procedure_lookup.get(
            transplant_hadm_id,
            (None, None),
        )
        allowed = _as_bool(getattr(row, "phase_specific_context_allowed"))
        short_gap = _as_bool(getattr(row, "short_gap_repeat_transplant_candidate"))
        # A short-gap repeat transplant code is retained as a candidate episode,
        # but phase-specific protocol narrowing is disabled until adjudication.
        if short_gap:
            allowed = False
        assignments[hadm_id] = TransplantAnchor(
            subject_id=subject_id,
            transplant_date=pd.Timestamp(row.transplant_date),
            transplant_hadm_id=transplant_hadm_id,
            source="stage00a3_episode_assignment",
            procedure_code=procedure_code,
            procedure_version=procedure_version,
            transplant_episode_id=str(row.transplant_episode_id),
            episode_number=int(row.episode_number),
            anchor_quality=str(getattr(row, "anchor_quality")),
            anchor_quality_level=str(getattr(row, "anchor_quality_level")),
            phase_specific_context_allowed=allowed,
            short_gap_repeat_transplant_candidate=short_gap,
        )
    return assignments


PERI_TRANSPLANT_PHASES = {
    "index_admission_preprocedure",
    "transplant_day_time_uncertain",
}


def classify_transplant_phase(
    timestamp: pd.Timestamp,
    hadm_id: int,
    anchor: TransplantAnchor,
) -> tuple[str, Optional[int]]:
    """Classify an order relative to its assigned transplant episode.

    ``procedures_icd.chartdate`` provides a calendar date rather than a
    reliable operation timestamp. Orders before that date in the index
    admission are therefore retained as peri-transplant pre-procedure events;
    orders on the procedure date are marked time-uncertain; and orders on a
    later date in the index admission are marked post-procedure.  This avoids
    treating pre-procedure orders as ordinary post-transplant phase evidence.
    """
    if anchor.transplant_date is None:
        return "prevalent_phase_unknown", None

    event_date = pd.Timestamp(anchor.transplant_date).normalize()
    current_date = pd.Timestamp(timestamp).normalize()
    days = int((current_date - event_date).days)
    is_index_admission = int(hadm_id) == int(anchor.transplant_hadm_id or -1)

    if days < 0:
        if is_index_admission:
            return "index_admission_preprocedure", days
        return "pre_transplant_observed", days

    if is_index_admission and days == 0:
        return "transplant_day_time_uncertain", days

    if is_index_admission:
        return "index_admission_postprocedure", days

    if days <= 30:
        return "early_0_30d", days
    if days <= 90:
        return "early_31_90d", days
    if days <= 180:
        return "transition_91_180d", days
    if days <= 365:
        return "first_year_181_365d", days
    return "maintenance_gt365d", days


def post_transplant_phase_context_allowed(
    phase: str,
    anchor: TransplantAnchor,
) -> bool:
    """Whether post-transplant phase-specific protocol narrowing is allowed.

    Admission-level anchor permission remains necessary but is not sufficient:
    pre-procedure and same-day time-uncertain orders must not inherit ordinary
    post-transplant phase narrowing.
    """
    if not anchor.phase_specific_context_allowed:
        return False
    return phase not in {
        "prevalent_phase_unknown",
        "pre_transplant_observed",
        "index_admission_preprocedure",
        "transplant_day_time_uncertain",
    }


def protocol_phase_for_context(
    actual_phase: str,
    post_transplant_context_allowed: bool,
) -> str:
    """Map the fine-grained order phase to the registered protocol phase.

    The reference table retains ``index_transplant_admission`` as the
    post-procedure index-admission scope.  Peri-transplant orders are mapped to
    ``prevalent_phase_unknown`` until separately sourced peri-transplant rule
    scopes are introduced.
    """
    if not post_transplant_context_allowed:
        return "prevalent_phase_unknown"
    if actual_phase == "index_admission_postprocedure":
        return "index_transplant_admission"
    return actual_phase
