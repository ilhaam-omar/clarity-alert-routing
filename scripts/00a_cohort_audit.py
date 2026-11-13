from __future__ import annotations

import pandas as pd

from common import atomic_json, banner, parser
from clarity_v3.mimic import load_cohort
from clarity_v3.project import ensure_output_dirs, load_project_paths
from clarity_v3.transplant_phase import load_transplant_anchors


def main() -> None:
    args = parser("Audit cohort denominator and transplant-anchor coverage").parse_args()
    banner("STAGE 00A - COHORT AND TRANSPLANT-ANCHOR AUDIT")
    paths = load_project_paths(args.paths)
    ensure_output_dirs(paths)
    raw = pd.read_csv(paths.cohort_file)
    subject = pd.to_numeric(raw["subject_id"], errors="coerce")
    hadm = pd.to_numeric(raw["hadm_id"], errors="coerce")
    valid = subject.notna() & hadm.notna()
    pairs = pd.DataFrame({
        "subject_id": subject.loc[valid].astype(int),
        "hadm_id": hadm.loc[valid].astype(int),
    })
    cohort = load_cohort(paths.cohort_file, paths.expected_patients)
    admissions = pd.read_csv(paths.admissions_file, usecols=["subject_id", "hadm_id"])
    admissions = admissions.apply(pd.to_numeric, errors="coerce").dropna().astype(int).drop_duplicates()
    matched = cohort.merge(admissions, on=["subject_id", "hadm_id"], how="left", indicator=True)
    anchors = load_transplant_anchors(cohort, paths.procedures_icd_file, paths.admissions_file)
    incident = sum(anchor.transplant_date is not None for anchor in anchors.values())
    prevalent = len(anchors) - incident
    per_patient = cohort.groupby("subject_id")["hadm_id"].nunique()
    summary = {
        "raw_cohort_rows": int(len(raw)),
        "invalid_identifier_rows": int((~valid).sum()),
        "duplicate_patient_admission_rows": int(pairs.duplicated().sum()),
        "unique_patients": int(cohort["subject_id"].nunique()),
        "unique_admissions": int(len(cohort)),
        "pairs_missing_from_admissions": int((matched["_merge"] != "both").sum()),
        "patients_with_multiple_admissions": int((per_patient > 1).sum()),
        "maximum_admissions_per_patient": int(per_patient.max()),
        "patients_with_observed_transplant_procedure_anchor": int(incident),
        "patients_prevalent_or_phase_unknown": int(prevalent),
    }
    for key, value in summary.items():
        print(f"{key}: {value:,}")
    if summary["pairs_missing_from_admissions"]:
        raise ValueError("Cohort contains admissions absent from admissions.csv")
    atomic_json(summary, paths.cohort_audit_file)
    print(f"Saved aggregate audit: {paths.cohort_audit_file}")
    print("\nSTAGE 00A: COMPLETE - PASS")


if __name__ == "__main__":
    main()
