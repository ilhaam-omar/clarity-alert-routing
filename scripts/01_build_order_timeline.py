from __future__ import annotations

import pandas as pd

from common import atomic_csv, banner, parser, progress_printer
from clarity_v3.mimic import build_order_timeline
from clarity_v3.project import ensure_output_dirs, load_project_paths
from clarity_v3.reference import ReferenceStore


def main() -> None:
    args = parser("Build time-causal MIMIC medication-order timeline").parse_args()
    banner("STAGE 01 - TIME-CAUSAL ORDER AND TRANSPLANT-PHASE TIMELINE")
    paths = load_project_paths(args.paths)
    ensure_output_dirs(paths)
    reference = ReferenceStore(paths.reference_dir)
    frame = build_order_timeline(
        paths.cohort_file,
        paths.patients_file,
        paths.admissions_file,
        paths.prescriptions_file,
        paths.labevents_file,
        paths.procedures_icd_file,
        paths.transfers_file,
        paths.services_file,
        paths.procedureevents_file,
        reference,
        expected_patients=paths.expected_patients,
        progress_callback=progress_printer("Order timeline"),
        poe_file=paths.poe_file,
        decision_time_mode=paths.decision_time_mode,
    )
    if frame.empty:
        raise ValueError("No order events were constructed")
    atomic_csv(frame, paths.orders_file)
    print(f"Orders: {len(frame):,}")
    print(f"Patients represented: {frame['subject_id'].nunique():,}")
    print(f"Admissions represented: {frame['hadm_id'].nunique():,}")
    print(f"Order bundles: {frame['bundle_id'].nunique():,}")
    print(f"Decision-time mode: {paths.decision_time_mode}")
    print("Decision-time source distribution:")
    print(frame["decision_time_source"].value_counts(dropna=False).to_string())
    print("POE linkage status distribution:")
    print(frame["poe_link_status"].value_counts(dropna=False).to_string())
    print(f"Orders with preceding creatinine: {frame['current_creatinine'].notna().sum():,}")
    print(f"KDIGO-aligned acute deterioration flags: {frame['acute_renal_deterioration'].astype(bool).sum():,}")
    print(f"Orders during time-scoped CRRT: {frame['crrt_active'].astype(bool).sum():,}")
    if "transplant_episode_id" in frame.columns:
        print(
            "Transplant episodes represented: "
            f"{frame['transplant_episode_id'].dropna().nunique():,}"
        )
        post_allowed_column = (
            "post_transplant_phase_context_allowed"
            if "post_transplant_phase_context_allowed" in frame.columns
            else "phase_specific_context_allowed"
        )
        print(
            "Orders with post-transplant phase-specific protocol context allowed: "
            f"{frame[post_allowed_column].astype(bool).sum():,}"
        )
        print(
            "Index-admission pre-procedure orders retained: "
            f"{frame['transplant_phase'].eq('index_admission_preprocedure').sum():,}"
        )
        print(
            "Transplant-day orders with procedure time uncertain: "
            f"{frame['transplant_phase'].eq('transplant_day_time_uncertain').sum():,}"
        )
        print(
            "Peri-transplant orders used as decision epochs/history: "
            f"{frame.get('peri_transplant_order', pd.Series(False, index=frame.index)).astype(bool).sum():,}"
        )
        print(
            "Orders assigned to short-gap repeat-transplant candidates: "
            f"{frame['short_gap_repeat_transplant_candidate'].astype(bool).sum():,}"
        )
        print("Transplant anchor-quality distribution:")
        print(
            frame["transplant_anchor_quality_level"]
            .value_counts(dropna=False)
            .to_string()
        )
    print("Transplant phase distribution:")
    print(frame["transplant_phase"].value_counts(dropna=False).to_string())
    print(f"Saved private event output: {paths.orders_file}")
    print("\nSTAGE 01: COMPLETE - PASS")


if __name__ == "__main__":
    main()
