from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clarity_v3.mimic import load_cohort
from clarity_v3.project import ensure_output_dirs, load_project_paths
from clarity_v3.transplant_phase import is_kidney_transplant_procedure, normalize_icd_code


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate kidney-transplant procedure anchors before phase assignment"
    )
    parser.add_argument("--paths", required=True, help="Path to the active project JSON configuration")
    args = parser.parse_args()

    paths = load_project_paths(args.paths)
    ensure_output_dirs(paths)

    if paths.procedures_icd_file is None or not paths.procedures_icd_file.exists():
        raise FileNotFoundError("procedures_icd.csv is required for this audit")

    cohort = load_cohort(paths.cohort_file, paths.expected_patients)
    cohort_pairs = set(zip(cohort["subject_id"].astype(int), cohort["hadm_id"].astype(int)))
    cohort_subjects = set(cohort["subject_id"].astype(int))

    admissions = pd.read_csv(
        paths.admissions_file,
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        low_memory=False,
    )
    for column in ["subject_id", "hadm_id"]:
        admissions[column] = pd.to_numeric(admissions[column], errors="coerce")
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
    admissions = admissions.dropna(subset=["subject_id", "hadm_id", "admittime"]).copy()
    admissions = admissions.astype({"subject_id": "int64", "hadm_id": "int64"})
    admissions["admit_day"] = admissions["admittime"].dt.normalize()
    admissions["discharge_day"] = admissions["dischtime"].dt.normalize()

    header = pd.read_csv(paths.procedures_icd_file, nrows=0)
    required = {"subject_id", "hadm_id", "icd_code", "icd_version"}
    missing = required - set(header.columns)
    if missing:
        raise ValueError(f"procedures_icd.csv is missing columns: {sorted(missing)}")

    usecols = list(required | ({"chartdate"} if "chartdate" in header.columns else set()))
    procedures = pd.read_csv(paths.procedures_icd_file, usecols=usecols, low_memory=False)
    for column in ["subject_id", "hadm_id"]:
        procedures[column] = pd.to_numeric(procedures[column], errors="coerce")
    procedures = procedures.dropna(subset=["subject_id", "hadm_id"]).copy()
    procedures = procedures.astype({"subject_id": "int64", "hadm_id": "int64"})
    procedures = procedures[procedures["subject_id"].isin(cohort_subjects)].copy()
    procedures["normalized_code"] = procedures["icd_code"].map(normalize_icd_code)
    procedures["is_exact_kidney_transplant"] = [
        is_kidney_transplant_procedure(code, version)
        for code, version in zip(procedures["icd_code"], procedures["icd_version"])
    ]
    matched = procedures[procedures["is_exact_kidney_transplant"]].copy()

    if "chartdate" in matched.columns:
        matched["chartdate"] = pd.to_datetime(matched["chartdate"], errors="coerce")
    else:
        matched["chartdate"] = pd.NaT

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
    matched["event_date"] = matched["chartdate"].where(
        matched["chartdate"].notna(), matched["admittime"]
    )
    matched["date_source"] = matched["chartdate"].notna().map(
        {True: "chartdate", False: "admittime_fallback"}
    )
    matched = matched.dropna(subset=["event_date"]).copy()
    matched["event_day"] = matched["event_date"].dt.normalize()
    matched["within_admission_by_date"] = (
        (matched["event_day"] >= matched["admit_day"])
        & (
            matched["discharge_day"].isna()
            | (matched["event_day"] <= matched["discharge_day"])
        )
    )

    unique_events = matched.drop_duplicates(
        subset=["subject_id", "hadm_id", "event_day", "normalized_code", "icd_version"]
    ).copy()
    unique_events = unique_events.sort_values(
        ["subject_id", "event_date", "hadm_id"], kind="stable"
    )
    anchors = unique_events.groupby("subject_id", as_index=False, sort=False).first()

    anchor_map = anchors.set_index("subject_id")[["hadm_id", "event_day"]].to_dict("index")
    relation_rows: list[dict[str, object]] = []
    admissions_lookup = admissions.set_index(["subject_id", "hadm_id"])
    for row in cohort.itertuples(index=False):
        subject_id = int(row.subject_id)
        hadm_id = int(row.hadm_id)
        anchor = anchor_map.get(subject_id)
        if anchor is None:
            relation = "no_anchor"
            days_from_anchor = None
        else:
            admission_row = admissions_lookup.loc[(subject_id, hadm_id)]
            if isinstance(admission_row, pd.DataFrame):
                admission_row = admission_row.iloc[0]
            admission_day = pd.Timestamp(admission_row["admit_day"])
            anchor_day = pd.Timestamp(anchor["event_day"])
            days_from_anchor = int((admission_day - anchor_day).days)
            if hadm_id == int(anchor["hadm_id"]):
                relation = "anchor_admission"
            elif days_from_anchor < 0:
                relation = "admission_before_anchor"
            elif days_from_anchor == 0:
                relation = "same_day_non_anchor_admission"
            else:
                relation = "admission_after_anchor"
        relation_rows.append(
            {"relation": relation, "days_from_anchor": days_from_anchor}
        )
    relations = pd.DataFrame(relation_rows)

    per_patient_dates = unique_events.groupby("subject_id")["event_day"].nunique()
    per_patient_admissions = unique_events.groupby("subject_id")["hadm_id"].nunique()

    summary = {
        "cohort_patients": int(cohort["subject_id"].nunique()),
        "cohort_admissions": int(len(cohort)),
        "exact_transplant_procedure_rows": int(len(matched)),
        "unique_exact_transplant_events": int(len(unique_events)),
        "patients_with_exact_transplant_procedure": int(
            unique_events["subject_id"].nunique()
        ),
        "patients_without_exact_transplant_procedure": int(
            cohort["subject_id"].nunique() - unique_events["subject_id"].nunique()
        ),
        "anchor_admission_present_in_cohort": int(
            sum(
                (int(row.subject_id), int(row.hadm_id)) in cohort_pairs
                for row in anchors.itertuples(index=False)
            )
        ),
        "anchor_date_within_recorded_admission": int(
            anchors["within_admission_by_date"].sum()
        ),
        "anchors_using_chartdate": int((anchors["date_source"] == "chartdate").sum()),
        "anchors_using_admittime_fallback": int(
            (anchors["date_source"] == "admittime_fallback").sum()
        ),
        "patients_with_multiple_distinct_transplant_dates": int(
            (per_patient_dates > 1).sum()
        ),
        "patients_with_transplant_codes_in_multiple_admissions": int(
            (per_patient_admissions > 1).sum()
        ),
        "maximum_distinct_transplant_dates_for_one_patient": int(
            per_patient_dates.max()
        )
        if not per_patient_dates.empty
        else 0,
        "cohort_anchor_admissions": int(
            (relations["relation"] == "anchor_admission").sum()
        ),
        "cohort_admissions_after_anchor": int(
            (relations["relation"] == "admission_after_anchor").sum()
        ),
        "cohort_admissions_before_anchor": int(
            (relations["relation"] == "admission_before_anchor").sum()
        ),
        "cohort_same_day_non_anchor_admissions": int(
            (relations["relation"] == "same_day_non_anchor_admission").sum()
        ),
        "earliest_anchor_date": str(anchors["event_day"].min().date())
        if not anchors.empty
        else None,
        "latest_anchor_date": str(anchors["event_day"].max().date())
        if not anchors.empty
        else None,
    }

    code_table = (
        unique_events.groupby(["icd_version", "normalized_code"], dropna=False)
        .size()
        .reset_index(name="unique_event_count")
    )
    if (
        paths.d_icd_procedures_file is not None
        and paths.d_icd_procedures_file.exists()
    ):
        dictionary = pd.read_csv(paths.d_icd_procedures_file, low_memory=False)
        if {"icd_code", "icd_version", "long_title"}.issubset(dictionary.columns):
            dictionary["normalized_code"] = dictionary["icd_code"].map(
                normalize_icd_code
            )
            dictionary["icd_version"] = pd.to_numeric(
                dictionary["icd_version"], errors="coerce"
            )
            dictionary = dictionary.dropna(subset=["icd_version"]).copy()
            dictionary["icd_version"] = dictionary["icd_version"].astype(int)
            dictionary = dictionary[
                ["icd_version", "normalized_code", "long_title"]
            ].drop_duplicates()
            code_table = code_table.merge(
                dictionary,
                on=["icd_version", "normalized_code"],
                how="left",
                validate="one_to_one",
            )

    summary_path = paths.evidence_dir / "00a1_transplant_anchor_validity.json"
    code_path = paths.evidence_dir / "00a1_transplant_anchor_code_distribution.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    code_table.to_csv(code_path, index=False)

    print("=" * 78)
    print("STAGE 00A.1 - TRANSPLANT ANCHOR VALIDITY AUDIT")
    print("=" * 78)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nExact procedure-code distribution:")
    print(code_table.to_string(index=False))
    print(f"\nSaved aggregate audit: {summary_path}")
    print(f"Saved code distribution: {code_path}")

    review_reasons: list[str] = []
    if summary["patients_without_exact_transplant_procedure"] != 0:
        review_reasons.append("some cohort patients lack an exact transplant procedure")
    if summary["anchor_date_within_recorded_admission"] != summary["cohort_patients"]:
        review_reasons.append("some anchor dates fall outside the coded admission dates")
    if summary["patients_with_multiple_distinct_transplant_dates"] > 0:
        review_reasons.append("some patients have multiple distinct transplant dates")
    if code_table.get("long_title", pd.Series(dtype=str)).isna().any():
        review_reasons.append("some exact procedure codes lack a dictionary description")

    if review_reasons:
        print("\nSTAGE 00A.1: REVIEW REQUIRED")
        for reason in review_reasons:
            print(" -", reason)
        raise SystemExit(2)

    print("\nSTAGE 00A.1: COMPLETE - PASS")


if __name__ == "__main__":
    main()