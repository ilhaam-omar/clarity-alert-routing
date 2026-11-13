from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clarity_v3.mimic import load_cohort
from clarity_v3.project import ensure_output_dirs, load_project_paths
from clarity_v3.transplant_phase import is_kidney_transplant_procedure, normalize_icd_code


def _safe_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _phase_from_days(days: int) -> str:
    if days < 0:
        return "pre_transplant"
    if days <= 30:
        return "early_0_30d"
    if days <= 90:
        return "early_31_90d"
    if days <= 180:
        return "transition_91_180d"
    if days <= 365:
        return "first_year_181_365d"
    return "maintenance_gt365d"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit transplant episodes, retransplantation, date anomalies, and the "
            "amount of post-transplant inpatient follow-up available in MIMIC."
        )
    )
    parser.add_argument("--paths", required=True, help="Active project JSON configuration")
    args = parser.parse_args()

    paths = load_project_paths(args.paths)
    ensure_output_dirs(paths)

    if paths.procedures_icd_file is None or not paths.procedures_icd_file.exists():
        raise FileNotFoundError("procedures_icd.csv is required")

    cohort = load_cohort(paths.cohort_file, paths.expected_patients)
    cohort["subject_id"] = pd.to_numeric(cohort["subject_id"], errors="coerce")
    cohort["hadm_id"] = pd.to_numeric(cohort["hadm_id"], errors="coerce")
    cohort = cohort.dropna(subset=["subject_id", "hadm_id"]).astype(
        {"subject_id": "int64", "hadm_id": "int64"}
    )
    cohort_pairs = set(zip(cohort["subject_id"], cohort["hadm_id"]))
    cohort_subjects = set(cohort["subject_id"])

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
    admissions = admissions[admissions["subject_id"].isin(cohort_subjects)].copy()
    admissions["admit_day"] = admissions["admittime"].dt.normalize()
    admissions["discharge_day"] = admissions["dischtime"].dt.normalize()
    admissions = admissions.drop_duplicates(["subject_id", "hadm_id"], keep="first")

    header = pd.read_csv(paths.procedures_icd_file, nrows=0)
    required = {"subject_id", "hadm_id", "icd_code", "icd_version"}
    missing = required - set(header.columns)
    if missing:
        raise ValueError(f"procedures_icd.csv is missing columns: {sorted(missing)}")
    usecols = list(required | ({"chartdate"} if "chartdate" in header.columns else set()))
    procedures = pd.read_csv(paths.procedures_icd_file, usecols=usecols, low_memory=False)
    procedures["subject_id"] = pd.to_numeric(procedures["subject_id"], errors="coerce")
    procedures["hadm_id"] = pd.to_numeric(procedures["hadm_id"], errors="coerce")
    procedures = procedures.dropna(subset=["subject_id", "hadm_id"]).copy()
    procedures = procedures.astype({"subject_id": "int64", "hadm_id": "int64"})
    procedures = procedures[procedures["subject_id"].isin(cohort_subjects)].copy()
    procedures["normalized_code"] = procedures["icd_code"].map(normalize_icd_code)
    procedures["is_exact_transplant"] = [
        is_kidney_transplant_procedure(code, version)
        for code, version in zip(procedures["icd_code"], procedures["icd_version"])
    ]
    matched = procedures[procedures["is_exact_transplant"]].copy()
    matched["chartdate"] = pd.to_datetime(
        matched["chartdate"] if "chartdate" in matched.columns else pd.NaT,
        errors="coerce",
    )

    matched = matched.merge(
        admissions[
            [
                "subject_id",
                "hadm_id",
                "admittime",
                "dischtime",
                "admit_day",
                "discharge_day",
            ]
        ],
        on=["subject_id", "hadm_id"],
        how="left",
        validate="many_to_one",
    )
    matched["chart_day"] = matched["chartdate"].dt.normalize()
    matched["chartdate_within_admission"] = (
        matched["chart_day"].notna()
        & matched["admit_day"].notna()
        & (matched["chart_day"] >= matched["admit_day"])
        & (
            matched["discharge_day"].isna()
            | (matched["chart_day"] <= matched["discharge_day"])
        )
    )
    matched["days_before_admission"] = (
        matched["admit_day"] - matched["chart_day"]
    ).dt.days.where(matched["chart_day"] < matched["admit_day"], 0)
    matched["days_after_discharge"] = (
        matched["chart_day"] - matched["discharge_day"]
    ).dt.days.where(
        matched["discharge_day"].notna()
        & (matched["chart_day"] > matched["discharge_day"]),
        0,
    )

    episode_rows: list[dict[str, Any]] = []
    for (subject_id, hadm_id), group in matched.groupby(
        ["subject_id", "hadm_id"], sort=True
    ):
        valid_days = sorted(
            pd.Timestamp(v)
            for v in group.loc[group["chartdate_within_admission"], "chart_day"].dropna().unique()
        )
        all_days = sorted(
            pd.Timestamp(v) for v in group["chart_day"].dropna().unique()
        )
        admit_day = group["admit_day"].dropna().iloc[0] if group["admit_day"].notna().any() else pd.NaT
        discharge_day = (
            group["discharge_day"].dropna().iloc[0]
            if group["discharge_day"].notna().any()
            else pd.NaT
        )
        if len(valid_days) == 1:
            anchor_day = valid_days[0]
            anchor_quality = "exact_single_in_admission"
        elif len(valid_days) > 1:
            anchor_day = valid_days[0]
            anchor_quality = "exact_multiple_dates_same_admission_earliest_used"
        elif pd.notna(admit_day):
            anchor_day = pd.Timestamp(admit_day)
            anchor_quality = "admission_date_proxy_invalid_chartdate"
        else:
            anchor_day = pd.NaT
            anchor_quality = "unusable_missing_admission_date"

        episode_rows.append(
            {
                "subject_id": int(subject_id),
                "hadm_id": int(hadm_id),
                "raw_code_rows": int(len(group)),
                "distinct_codes": int(group["normalized_code"].nunique()),
                "distinct_chartdates": int(len(all_days)),
                "valid_in_admission_chartdates": int(len(valid_days)),
                "invalid_chartdate_rows": int((~group["chartdate_within_admission"]).sum()),
                "admit_day": admit_day,
                "discharge_day": discharge_day,
                "earliest_chart_day": all_days[0] if all_days else pd.NaT,
                "latest_chart_day": all_days[-1] if all_days else pd.NaT,
                "candidate_anchor_day": anchor_day,
                "anchor_quality": anchor_quality,
                "maximum_days_before_admission": int(
                    pd.to_numeric(group["days_before_admission"], errors="coerce")
                    .fillna(0)
                    .max()
                ),
                "maximum_days_after_discharge": int(
                    pd.to_numeric(group["days_after_discharge"], errors="coerce")
                    .fillna(0)
                    .max()
                ),
                "codes": "|".join(sorted(set(group["normalized_code"].astype(str)))),
            }
        )

    episodes = pd.DataFrame(episode_rows).sort_values(
        ["subject_id", "candidate_anchor_day", "hadm_id"], kind="stable"
    )
    episodes["episode_number"] = episodes.groupby("subject_id").cumcount() + 1
    episodes["previous_episode_anchor_day"] = episodes.groupby("subject_id")[
        "candidate_anchor_day"
    ].shift(1)
    episodes["days_since_previous_transplant_episode"] = (
        episodes["candidate_anchor_day"] - episodes["previous_episode_anchor_day"]
    ).dt.days
    episodes["in_current_cohort"] = [
        (int(sid), int(hid)) in cohort_pairs
        for sid, hid in zip(episodes["subject_id"], episodes["hadm_id"])
    ]

    episode_pair_set = set(zip(episodes["subject_id"], episodes["hadm_id"]))

    assignment_rows: list[dict[str, Any]] = []
    episodes_by_subject = {
        int(sid): group.sort_values("candidate_anchor_day", kind="stable").copy()
        for sid, group in episodes.groupby("subject_id", sort=False)
    }
    for row in admissions.itertuples(index=False):
        sid = int(row.subject_id)
        hid = int(row.hadm_id)
        admit_day = pd.Timestamp(row.admit_day)
        subject_episodes = episodes_by_subject.get(sid)
        relation = "no_transplant_episode"
        episode_number: int | None = None
        anchor_day = pd.NaT
        days_from_anchor: int | None = None
        phase = "no_transplant_episode"

        if subject_episodes is not None and not subject_episodes.empty:
            same_admission = subject_episodes[subject_episodes["hadm_id"] == hid]
            if not same_admission.empty:
                chosen = same_admission.iloc[-1]
                relation = "transplant_episode_admission"
            else:
                eligible = subject_episodes[
                    subject_episodes["candidate_anchor_day"] <= admit_day
                ]
                if eligible.empty:
                    chosen = subject_episodes.iloc[0]
                    relation = "admission_before_first_observed_transplant"
                else:
                    chosen = eligible.iloc[-1]
                    relation = "post_transplant_nonindex_admission"
            episode_number = int(chosen["episode_number"])
            anchor_day = pd.Timestamp(chosen["candidate_anchor_day"])
            days_from_anchor = int((admit_day - anchor_day).days)
            phase = (
                "index_transplant_admission"
                if relation == "transplant_episode_admission"
                else _phase_from_days(days_from_anchor)
            )

        assignment_rows.append(
            {
                "subject_id": sid,
                "hadm_id": hid,
                "admit_day": admit_day,
                "discharge_day": row.discharge_day,
                "relation": relation,
                "episode_number": episode_number,
                "assigned_anchor_day": anchor_day,
                "days_from_assigned_anchor": days_from_anchor,
                "assigned_phase": phase,
                "in_current_cohort": (sid, hid) in cohort_pairs,
                "is_exact_transplant_episode_admission": (sid, hid) in episode_pair_set,
            }
        )

    assignments = pd.DataFrame(assignment_rows)
    post_nonindex = assignments[
        assignments["relation"] == "post_transplant_nonindex_admission"
    ].copy()
    post_nonindex_365 = post_nonindex[
        post_nonindex["days_from_assigned_anchor"].between(0, 365, inclusive="both")
    ].copy()

    per_patient_episode_count = episodes.groupby("subject_id").size()
    retransplant_gaps = episodes.loc[
        episodes["episode_number"] > 1, "days_since_previous_transplant_episode"
    ].dropna()

    summary = {
        "cohort_patients": int(cohort["subject_id"].nunique()),
        "current_cohort_admissions": int(len(cohort)),
        "current_cohort_admissions_with_exact_transplant_code": int(
            sum((sid, hid) in episode_pair_set for sid, hid in cohort_pairs)
        ),
        "current_cohort_admissions_without_exact_transplant_code": int(
            sum((sid, hid) not in episode_pair_set for sid, hid in cohort_pairs)
        ),
        "collapsed_transplant_episodes": int(len(episodes)),
        "patients_with_one_transplant_episode": int((per_patient_episode_count == 1).sum()),
        "patients_with_multiple_transplant_episodes": int((per_patient_episode_count > 1).sum()),
        "maximum_transplant_episodes_for_one_patient": int(per_patient_episode_count.max())
        if not per_patient_episode_count.empty
        else 0,
        "episodes_exact_single_date_in_admission": int(
            (episodes["anchor_quality"] == "exact_single_in_admission").sum()
        ),
        "episodes_with_multiple_valid_dates_same_admission": int(
            (
                episodes["anchor_quality"]
                == "exact_multiple_dates_same_admission_earliest_used"
            ).sum()
        ),
        "episodes_using_admission_date_proxy": int(
            (
                episodes["anchor_quality"]
                == "admission_date_proxy_invalid_chartdate"
            ).sum()
        ),
        "maximum_chartdate_days_before_admission": int(
            episodes["maximum_days_before_admission"].max()
        )
        if not episodes.empty
        else 0,
        "maximum_chartdate_days_after_discharge": int(
            episodes["maximum_days_after_discharge"].max()
        )
        if not episodes.empty
        else 0,
        "retransplant_gap_days_minimum": int(retransplant_gaps.min())
        if not retransplant_gaps.empty
        else None,
        "retransplant_gap_days_median": float(retransplant_gaps.median())
        if not retransplant_gaps.empty
        else None,
        "retransplant_gap_days_maximum": int(retransplant_gaps.max())
        if not retransplant_gaps.empty
        else None,
        "all_mimic_admissions_for_cohort_patients": int(len(assignments)),
        "post_transplant_nonindex_admissions_available": int(len(post_nonindex)),
        "patients_with_any_post_transplant_nonindex_admission": int(
            post_nonindex["subject_id"].nunique()
        ),
        "post_transplant_nonindex_admissions_within_365_days": int(len(post_nonindex_365)),
        "patients_with_post_transplant_nonindex_admission_within_365_days": int(
            post_nonindex_365["subject_id"].nunique()
        ),
        "post_transplant_nonindex_admissions_after_365_days": int(
            (post_nonindex["days_from_assigned_anchor"] > 365).sum()
        ),
        "candidate_primary_365d_unique_admissions": int(
            pd.concat(
                [
                    assignments[assignments["relation"] == "transplant_episode_admission"][["subject_id", "hadm_id"]],
                    post_nonindex_365[["subject_id", "hadm_id"]],
                ],
                ignore_index=True,
            )
            .drop_duplicates()
            .shape[0]
        ),
        "candidate_all_post_transplant_unique_admissions": int(
            assignments[
                assignments["relation"].isin(
                    ["transplant_episode_admission", "post_transplant_nonindex_admission"]
                )
            ][["subject_id", "hadm_id"]]
            .drop_duplicates()
            .shape[0]
        ),
    }

    phase_counts = (
        assignments.groupby(["relation", "assigned_phase"], dropna=False)
        .size()
        .reset_index(name="admission_count")
        .sort_values(["relation", "assigned_phase"], kind="stable")
    )

    evidence_json = paths.evidence_dir / "00a2_transplant_episode_scope.json"
    evidence_phase = paths.evidence_dir / "00a2_available_followup_distribution.csv"
    private_episode = paths.private_dir / "00a2_transplant_episodes_private.csv"
    private_assignments = paths.private_dir / "00a2_all_admission_assignments_private.csv"

    evidence_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    phase_counts.to_csv(evidence_phase, index=False)
    episodes.to_csv(private_episode, index=False)
    assignments.to_csv(private_assignments, index=False)

    print("=" * 78)
    print("STAGE 00A.2 - TRANSPLANT EPISODE AND LONGITUDINAL SCOPE AUDIT")
    print("=" * 78)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nAvailable admission distribution:")
    print(phase_counts.to_string(index=False))
    print(f"\nSaved aggregate audit: {evidence_json}")
    print(f"Saved aggregate distribution: {evidence_phase}")
    print(f"Saved PRIVATE episode detail: {private_episode}")
    print(f"Saved PRIVATE admission assignments: {private_assignments}")

    review_reasons: list[str] = []
    if summary["current_cohort_admissions_without_exact_transplant_code"] > 0:
        review_reasons.append(
            "the current cohort already contains admissions without an exact transplant procedure"
        )
    if summary["episodes_using_admission_date_proxy"] > 0:
        review_reasons.append(
            "some exact transplant episodes require an admission-date proxy because chartdate is outside the stay"
        )
    if summary["episodes_with_multiple_valid_dates_same_admission"] > 0:
        review_reasons.append(
            "some transplant admissions contain more than one valid transplant chartdate"
        )

    if review_reasons:
        print("\nSTAGE 00A.2: REVIEW REQUIRED")
        for reason in review_reasons:
            print(" -", reason)
        raise SystemExit(2)

    print("\nSTAGE 00A.2: COMPLETE - PASS")


if __name__ == "__main__":
    main()