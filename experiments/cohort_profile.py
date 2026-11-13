"""Cohort profile: how much of the care in this cohort is unrelated to the transplant itself?

Describes admissions as index (the transplant admission) or follow-up (a later admission
assigned to a transplant episode), and reports for each:
  - admissions, patients, admissions per patient
  - the clinical services of follow-up admissions (where the patient was admitted)
  - orders, order bundles, detected concerns and protected bundles
  - concerns by rule family
  - concerns involving an immunosuppressant, and how many of those were raised by a medicine
    that is not itself an immunosuppressant (a medicine for another condition checked against
    the transplant regimen)
Only aggregate counts are written; counts from 1 to 10 are masked.

Usage
  python experiments/cohort_profile.py --project . --outputs outputs_corrected
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

MASK = 11
IMMUNOSUPPRESSANTS = ("tacrolimus", "cyclosporin", "ciclosporin", "mycophenol", "azathioprine", "sirolimus",
                      "everolimus", "belatacept", "basiliximab", "thymoglobulin", "antithymocyte")
STEROIDS = ("prednisone", "prednisolone", "methylprednisolone")


def m(n) -> object:
    n = int(n)
    return n if n == 0 or n >= MASK else f"<{MASK}"


def tf(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "t"])


def has(s: pd.Series, words) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()
    return s.apply(lambda v: any(w in v for w in words))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--outputs", default="outputs_corrected")
    a = ap.parse_args()
    project = a.project.expanduser().resolve()
    private = project / a.outputs / "private"
    orders = pd.read_csv(private / "01_order_timeline.csv", low_memory=False,
                         usecols=lambda c: c in {"subject_id", "hadm_id", "order_id", "bundle_id", "transplant_hadm_id",
                                                 "clinical_service", "careunit", "drug", "ingredient_key"})
    hazards = pd.read_csv(private / "02_detected_hazards.csv", low_memory=False,
                          usecols=lambda c: c in {"signal_id", "hadm_id", "bundle_id", "rule_family", "new_drug",
                                                  "new_ingredient_key", "partner_ingredient_key", "partner_raw_name",
                                                  "overlapping_ingredient_keys"})
    bundles = pd.read_csv(private / "05_order_bundles.csv", low_memory=False)

    # admission type
    adm = orders.groupby("hadm_id").agg(subject_id=("subject_id", "first"),
                                        transplant_hadm_id=("transplant_hadm_id", "first"),
                                        clinical_service=("clinical_service", "first")).reset_index()
    t = pd.to_numeric(adm["transplant_hadm_id"], errors="coerce")
    h = pd.to_numeric(adm["hadm_id"], errors="coerce")
    adm["admission_type"] = np.where(t.isna(), "unanchored", np.where(t.eq(h), "index", "follow_up"))
    kind = adm.set_index("hadm_id")["admission_type"]

    out = project / ("cohort_profile_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True)

    per_patient = adm.groupby("subject_id").size()
    summary = {
        "patients": int(adm["subject_id"].nunique()),
        "admissions": int(len(adm)),
        "admissions_by_type": {k: m(v) for k, v in adm["admission_type"].value_counts().items()},
        "follow_up_share_percent": round(100 * float((adm["admission_type"] == "follow_up").mean()), 1),
        "admissions_per_patient_median": float(per_patient.median()),
        "admissions_per_patient_iqr": [float(per_patient.quantile(0.25)), float(per_patient.quantile(0.75))],
        "patients_with_follow_up_admission": m((adm.groupby("subject_id")["admission_type"]
                                                .apply(lambda s: (s == "follow_up").any())).sum()),
    }

    # services of follow-up admissions
    fu = adm[adm["admission_type"] == "follow_up"]
    services = fu["clinical_service"].fillna("unknown").value_counts().rename_axis("clinical_service").reset_index(name="admissions")
    services["percent"] = (100 * services["admissions"] / max(len(fu), 1)).round(1)
    services["admissions"] = services["admissions"].map(m)

    # orders, bundles, concerns, protected bundles by type
    orders["admission_type"] = orders["hadm_id"].map(kind)
    hazards["admission_type"] = hazards["hadm_id"].map(kind)
    bundles["admission_type"] = bundles["hadm_id"].map(kind)
    rows = []
    for k in ("index", "follow_up", "unanchored"):
        o = orders[orders["admission_type"] == k]; hz = hazards[hazards["admission_type"] == k]
        b = bundles[bundles["admission_type"] == k]
        pa, pr = tf(b["protected_alert"]), tf(b["protected_review"])
        rows.append({"admission_type": k, "admissions": m((adm["admission_type"] == k).sum()),
                     "prescription_records": m(len(o)), "order_bundles": m(o["bundle_id"].nunique()),
                     "affected_bundles": m(len(b)), "concerns": m(len(hz)),
                     "protected_bundles": m((pa | pr).sum()),
                     "concerns_per_admission": round(len(hz) / max((adm["admission_type"] == k).sum(), 1), 1)})
    by_type = pd.DataFrame(rows)

    fam = pd.crosstab(hazards["rule_family"].fillna("unknown"), hazards["admission_type"].fillna("unanchored"))
    fam = fam.apply(lambda col: col.map(m)).reset_index()

    # immunosuppressant involvement
    new_is = has(hazards.get("new_ingredient_key", pd.Series(dtype=str)), IMMUNOSUPPRESSANTS) | \
        has(hazards.get("new_drug", pd.Series(dtype=str)), IMMUNOSUPPRESSANTS)
    partner_cols = [c for c in ("partner_ingredient_key", "partner_raw_name", "overlapping_ingredient_keys") if c in hazards]
    partner_is = pd.Series(False, index=hazards.index)
    for c in partner_cols:
        partner_is |= has(hazards[c], IMMUNOSUPPRESSANTS)
    any_is = new_is | partner_is
    other_drug_vs_is = partner_is & ~new_is
    rows = []
    for k in ("all", "index", "follow_up"):
        sel = pd.Series(True, index=hazards.index) if k == "all" else hazards["admission_type"].eq(k)
        n = int(sel.sum())
        rows.append({"admission_type": k, "concerns": m(n),
                     "involving_immunosuppressant": m((any_is & sel).sum()),
                     "involving_immunosuppressant_percent": round(100 * (any_is & sel).sum() / max(n, 1), 1),
                     "raised_by_non_immunosuppressant_against_regimen": m((other_drug_vs_is & sel).sum()),
                     "raised_by_non_immunosuppressant_percent_of_is_concerns":
                         round(100 * (other_drug_vs_is & sel).sum() / max((any_is & sel).sum(), 1), 1)})
    immuno = pd.DataFrame(rows)

    (out / "cohort_summary.json").write_text(json.dumps(summary, indent=2))
    services.to_csv(out / "follow_up_admission_services.csv", index=False)
    by_type.to_csv(out / "workload_by_admission_type.csv", index=False)
    fam.to_csv(out / "concerns_by_rule_family_and_admission_type.csv", index=False)
    immuno.to_csv(out / "immunosuppressant_concerns.csv", index=False)
    print(json.dumps(summary, indent=2))
    print(by_type.to_string(index=False))
    print(services.head(12).to_string(index=False))
    print(immuno.to_string(index=False))
    print("Saved:", out)


if __name__ == "__main__":
    main()
