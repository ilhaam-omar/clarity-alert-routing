from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clarity_v3.mimic import load_cohort
from clarity_v3.project import ensure_output_dirs, load_project_paths
from clarity_v3.transplant_phase import (
    classify_transplant_phase,
    load_transplant_episode_assignments,
)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the episode-aware transplant phase engine before the real "
            "MIMIC timeline is built."
        )
    )
    parser.add_argument("--paths", required=True)
    args = parser.parse_args()

    paths = load_project_paths(Path(args.paths).resolve())
    ensure_output_dirs(paths)
    cohort = load_cohort(paths.cohort_file, paths.expected_patients)
    required = {
        "transplant_episode_id",
        "episode_number",
        "transplant_hadm_id",
        "transplant_date",
        "anchor_quality",
        "anchor_quality_level",
        "phase_specific_context_allowed",
        "short_gap_repeat_transplant_candidate",
        "relation",
        "assigned_phase",
    }
    missing = required - set(cohort.columns)
    if missing:
        raise ValueError(
            "The active cohort is not the Stage 00A.3 episode-based cohort. "
            f"Missing columns: {sorted(missing)}"
        )

    admissions = pd.read_csv(
        paths.admissions_file,
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        low_memory=False,
    )
    admissions["subject_id"] = pd.to_numeric(admissions["subject_id"], errors="coerce")
    admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="coerce")
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
    admissions = admissions.dropna(subset=["subject_id", "hadm_id", "admittime"]).copy()
    admissions = admissions.astype({"subject_id": "int64", "hadm_id": "int64"})
    admission_lookup = admissions.set_index("hadm_id")

    assignments = load_transplant_episode_assignments(
        cohort,
        paths.procedures_icd_file,
        paths.admissions_file,
    )

    failures: list[str] = []
    if len(assignments) != len(cohort):
        failures.append(
            f"episode assignment count mismatch: {len(assignments)} for {len(cohort)} admissions"
        )

    episode_rows = cohort[cohort["relation"].eq("transplant_episode_admission")].copy()
    unique_episodes = int(cohort["transplant_episode_id"].nunique())
    if len(episode_rows) != unique_episodes:
        failures.append(
            "each transplant_episode_id must have exactly one transplant episode admission"
        )

    phase_mismatches = 0
    not_most_recent = 0
    anchor_subject_mismatches = 0
    procedure_metadata_missing = 0

    episode_dates_by_subject: dict[int, list[pd.Timestamp]] = {}
    for sid, group in episode_rows.groupby("subject_id"):
        episode_dates_by_subject[int(sid)] = sorted(
            pd.to_datetime(group["transplant_date"], errors="coerce").dropna().tolist()
        )

    for row in cohort.itertuples(index=False):
        hadm_id = int(row.hadm_id)
        sid = int(row.subject_id)
        anchor = assignments.get(hadm_id)
        if anchor is None:
            failures.append(f"missing assignment for hadm_id {hadm_id}")
            continue
        if anchor.subject_id != sid:
            anchor_subject_mismatches += 1
        if anchor.procedure_code is None or anchor.procedure_version is None:
            procedure_metadata_missing += 1

        if hadm_id not in admission_lookup.index:
            failures.append(f"hadm_id {hadm_id} absent from admissions.csv")
            continue
        admittime = pd.Timestamp(admission_lookup.loc[hadm_id, "admittime"])
        relation = str(row.relation)
        expected_phase = str(row.assigned_phase)
        computed_phase, _ = classify_transplant_phase(admittime, hadm_id, anchor)

        if relation == "transplant_episode_admission":
            if hadm_id != int(anchor.transplant_hadm_id or -1):
                phase_mismatches += 1
        elif computed_phase != expected_phase:
            phase_mismatches += 1

        if relation == "post_transplant_nonindex_admission":
            candidate_dates = [
                value
                for value in episode_dates_by_subject.get(sid, [])
                if pd.Timestamp(value).normalize() <= admittime.normalize()
            ]
            if not candidate_dates:
                not_most_recent += 1
            else:
                latest = max(candidate_dates)
                if pd.Timestamp(anchor.transplant_date).normalize() != pd.Timestamp(latest).normalize():
                    not_most_recent += 1

    short_gap_rows = cohort[
        cohort["short_gap_repeat_transplant_candidate"].map(_as_bool)
    ]
    short_gap_assignments_allowed = sum(
        int(assignments[int(hadm)].phase_specific_context_allowed)
        for hadm in short_gap_rows["hadm_id"]
    )
    if short_gap_assignments_allowed:
        failures.append(
            "short-gap repeat-transplant candidate admissions still allow phase-specific narrowing"
        )

    low_quality_nonindex = cohort[
        cohort["anchor_quality_level"].astype(str).eq("low")
        & cohort["relation"].eq("post_transplant_nonindex_admission")
    ]
    low_quality_nonindex_allowed = sum(
        int(assignments[int(hadm)].phase_specific_context_allowed)
        for hadm in low_quality_nonindex["hadm_id"]
    )
    if low_quality_nonindex_allowed:
        failures.append(
            "low-quality non-index anchor assignments still allow phase-specific narrowing"
        )

    if anchor_subject_mismatches:
        failures.append(f"{anchor_subject_mismatches} anchor-subject mismatch(es)")
    if phase_mismatches:
        failures.append(f"{phase_mismatches} admission phase mismatch(es)")
    if not_most_recent:
        failures.append(
            f"{not_most_recent} non-index admission(s) were not assigned to the most recent preceding episode"
        )
    if procedure_metadata_missing:
        failures.append(
            f"{procedure_metadata_missing} admission assignment(s) lack exact transplant procedure metadata"
        )

    summary = {
        "cohort_patients": int(cohort["subject_id"].nunique()),
        "cohort_admissions": int(len(cohort)),
        "transplant_episodes": unique_episodes,
        "episode_assignments_loaded": int(len(assignments)),
        "transplant_episode_admissions": int(len(episode_rows)),
        "post_transplant_nonindex_admissions": int(
            cohort["relation"].eq("post_transplant_nonindex_admission").sum()
        ),
        "phase_specific_context_allowed_admissions": int(
            sum(int(item.phase_specific_context_allowed) for item in assignments.values())
        ),
        "phase_specific_context_blocked_admissions": int(
            sum(not item.phase_specific_context_allowed for item in assignments.values())
        ),
        "short_gap_candidate_episodes": int(
            episode_rows["short_gap_repeat_transplant_candidate"].map(_as_bool).sum()
        ),
        "short_gap_candidate_admissions": int(len(short_gap_rows)),
        "short_gap_assignments_allowing_phase_specific_context": int(
            short_gap_assignments_allowed
        ),
        "low_quality_nonindex_admissions": int(len(low_quality_nonindex)),
        "low_quality_nonindex_allowing_phase_specific_context": int(
            low_quality_nonindex_allowed
        ),
        "phase_mismatches": int(phase_mismatches),
        "not_most_recent_episode_assignments": int(not_most_recent),
        "anchor_subject_mismatches": int(anchor_subject_mismatches),
        "assignments_missing_procedure_metadata": int(procedure_metadata_missing),
        "status": "PASS" if not failures else "REVIEW",
        "failures": failures,
    }

    evidence_file = paths.evidence_dir / "00a4_episode_phase_engine_validation.json"
    evidence_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 78)
    print("STAGE 00A.4 - EPISODE-AWARE TRANSPLANT PHASE ENGINE GATE")
    print("=" * 78)
    for key, value in summary.items():
        if key not in {"failures", "status"}:
            print(f"{key}: {value}")
    print(f"Saved aggregate validation: {evidence_file}")
    if failures:
        print("\nSTAGE 00A.4: REVIEW REQUIRED")
        for item in failures:
            print(f" - {item}")
        raise SystemExit(1)
    print("\nSTAGE 00A.4: COMPLETE - PASS")


if __name__ == "__main__":
    main()
