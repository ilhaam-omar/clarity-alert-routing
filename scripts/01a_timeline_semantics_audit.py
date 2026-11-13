from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clarity_v3.project import ensure_output_dirs, load_project_paths


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y", "t"})
    )


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 3) if denominator else 0.0


def _safe_min(series: pd.Series) -> Any:
    cleaned = series.dropna()
    return None if cleaned.empty else float(cleaned.min())


def _safe_median(series: pd.Series) -> Any:
    cleaned = series.dropna()
    return None if cleaned.empty else float(cleaned.median())


def _safe_max(series: pd.Series) -> Any:
    cleaned = series.dropna()
    return None if cleaned.empty else float(cleaned.max())


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the Stage-01 order timeline after the peri-transplant phase correction. "
            "Pre-procedure and transplant-day orders are retained as decision epochs and "
            "longitudinal history, but cannot inherit ordinary post-transplant phase narrowing."
        )
    )
    parser.add_argument("--paths", required=True, help="Primary Phase-1 configuration")
    args = parser.parse_args()

    paths = load_project_paths(Path(args.paths).resolve())
    ensure_output_dirs(paths)

    if not paths.orders_file.exists():
        raise FileNotFoundError(f"Stage-01 order timeline does not exist: {paths.orders_file}")

    orders = pd.read_csv(paths.orders_file, low_memory=False)
    cohort = pd.read_csv(paths.cohort_file, low_memory=False)

    required_order_columns = {
        "subject_id",
        "hadm_id",
        "order_id",
        "bundle_id",
        "timestamp",
        "transplant_date",
        "transplant_hadm_id",
        "transplant_episode_id",
        "transplant_phase",
        "transplant_protocol_phase",
        "days_since_transplant",
        "transplant_anchor_quality_level",
        "episode_phase_specific_context_allowed",
        "phase_specific_context_allowed",
        "post_transplant_phase_context_allowed",
        "peri_transplant_order",
        "short_gap_repeat_transplant_candidate",
        "current_creatinine",
        "acute_renal_deterioration",
        "crrt_active",
    }
    missing = required_order_columns - set(orders.columns)
    if missing:
        raise ValueError(f"Order timeline missing required columns: {sorted(missing)}")

    required_cohort_columns = {
        "subject_id",
        "hadm_id",
        "relation",
        "assigned_phase",
        "days_from_assigned_anchor",
        "anchor_quality_level",
        "phase_specific_context_allowed",
    }
    missing_cohort = required_cohort_columns - set(cohort.columns)
    if missing_cohort:
        raise ValueError(f"Primary cohort missing required columns: {sorted(missing_cohort)}")

    for frame, columns in (
        (orders, ["subject_id", "hadm_id", "transplant_hadm_id"]),
        (cohort, ["subject_id", "hadm_id"]),
    ):
        for column in columns:
            frame[column] = _num(frame[column])

    orders["timestamp"] = pd.to_datetime(orders["timestamp"], errors="coerce")
    orders["transplant_date"] = pd.to_datetime(orders["transplant_date"], errors="coerce")
    orders["days_since_transplant"] = _num(orders["days_since_transplant"])
    orders["phase_allowed_bool"] = _as_bool(orders["phase_specific_context_allowed"])
    orders["post_phase_allowed_bool"] = _as_bool(
        orders["post_transplant_phase_context_allowed"]
    )
    orders["episode_phase_allowed_bool"] = _as_bool(
        orders["episode_phase_specific_context_allowed"]
    )
    orders["peri_bool"] = _as_bool(orders["peri_transplant_order"])
    orders["short_gap_bool"] = _as_bool(
        orders["short_gap_repeat_transplant_candidate"]
    )
    orders["acute_bool"] = _as_bool(orders["acute_renal_deterioration"])
    orders["crrt_bool"] = _as_bool(orders["crrt_active"])

    cohort["days_from_assigned_anchor"] = _num(cohort["days_from_assigned_anchor"])
    cohort["cohort_phase_allowed_bool"] = _as_bool(
        cohort["phase_specific_context_allowed"]
    )

    cohort_unique = cohort.drop_duplicates(["subject_id", "hadm_id"]).copy()
    order_admissions = set(orders["hadm_id"].dropna().astype("int64"))
    cohort_admissions = set(cohort_unique["hadm_id"].dropna().astype("int64"))
    extra_order_admissions = sorted(order_admissions - cohort_admissions)
    admissions_without_orders = sorted(cohort_admissions - order_admissions)

    merge_columns = [
        "subject_id",
        "hadm_id",
        "relation",
        "assigned_phase",
        "days_from_assigned_anchor",
        "anchor_quality_level",
        "cohort_phase_allowed_bool",
    ]
    merged = orders.merge(
        cohort_unique[merge_columns],
        on=["subject_id", "hadm_id"],
        how="left",
        validate="many_to_one",
        suffixes=("_order", "_cohort"),
    )

    pre = merged[
        merged["transplant_phase"].eq("index_admission_preprocedure")
    ].copy()
    day_uncertain = merged[
        merged["transplant_phase"].eq("transplant_day_time_uncertain")
    ].copy()
    legacy_pre = merged[
        merged["transplant_phase"].eq("pre_transplant_observed")
    ].copy()
    maintenance = merged[
        merged["transplant_phase"].eq("maintenance_gt365d")
    ].copy()
    blocked = merged[~merged["post_phase_allowed_bool"]].copy()

    pre_all_index = bool(
        pre.empty
        or (
            pre["hadm_id"].astype("Int64")
            == pre["transplant_hadm_id"].astype("Int64")
        ).all()
    )
    pre_all_before_anchor = bool(
        pre.empty
        or (
            pre["timestamp"].dt.normalize()
            < pre["transplant_date"].dt.normalize()
        ).all()
    )
    pre_post_allowed_count = int(pre["post_phase_allowed_bool"].sum())
    pre_protocol_unknown_count = int(
        pre["transplant_protocol_phase"].eq("prevalent_phase_unknown").sum()
    )

    day_all_index = bool(
        day_uncertain.empty
        or (
            day_uncertain["hadm_id"].astype("Int64")
            == day_uncertain["transplant_hadm_id"].astype("Int64")
        ).all()
    )
    day_all_same_calendar_date = bool(
        day_uncertain.empty
        or (
            day_uncertain["timestamp"].dt.normalize()
            == day_uncertain["transplant_date"].dt.normalize()
        ).all()
    )
    day_post_allowed_count = int(day_uncertain["post_phase_allowed_bool"].sum())
    day_protocol_unknown_count = int(
        day_uncertain["transplant_protocol_phase"].eq("prevalent_phase_unknown").sum()
    )

    alias_mismatch_count = int(
        (orders["phase_allowed_bool"] != orders["post_phase_allowed_bool"]).sum()
    )
    peri_flag_mismatch_count = int(
        (
            orders["peri_bool"]
            != orders["transplant_phase"].isin(
                {"index_admission_preprocedure", "transplant_day_time_uncertain"}
            )
        ).sum()
    )

    maintenance_days = maintenance["days_since_transplant"]
    maintenance_started_within_365 = int(
        maintenance["days_from_assigned_anchor"].between(
            0, 365, inclusive="both"
        ).sum()
    )
    maintenance_all_full_admission_spillover = bool(
        maintenance.empty
        or maintenance_started_within_365 == len(maintenance)
    )

    orders_with_creatinine = int(orders["current_creatinine"].notna().sum())
    acute_orders = int(orders["acute_bool"].sum())
    acute_bundles = int(orders.loc[orders["acute_bool"], "bundle_id"].nunique())
    acute_admissions = int(orders.loc[orders["acute_bool"], "hadm_id"].nunique())
    acute_patients = int(orders.loc[orders["acute_bool"], "subject_id"].nunique())
    crrt_orders = int(orders["crrt_bool"].sum())

    missing_admission_frame = cohort_unique[
        cohort_unique["hadm_id"].isin(admissions_without_orders)
    ].copy()
    missing_admission_summary = (
        missing_admission_frame.groupby(
            ["relation", "assigned_phase", "anchor_quality_level"],
            dropna=False,
        )
        .size()
        .reset_index(name="admissions_without_usable_orders")
        .sort_values(
            ["relation", "assigned_phase", "anchor_quality_level"],
            kind="stable",
        )
    )

    phase_quality = (
        merged.groupby(
            ["transplant_phase", "transplant_anchor_quality_level"],
            dropna=False,
        )
        .size()
        .reset_index(name="order_count")
        .sort_values(
            ["transplant_phase", "transplant_anchor_quality_level"],
            kind="stable",
        )
    )
    phase_gate = (
        merged.groupby(
            [
                "transplant_phase",
                "post_phase_allowed_bool",
                "transplant_protocol_phase",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="order_count")
        .sort_values(
            [
                "transplant_phase",
                "post_phase_allowed_bool",
                "transplant_protocol_phase",
            ],
            kind="stable",
        )
    )
    maintenance_context = (
        maintenance.groupby(
            [
                "relation",
                "assigned_phase",
                "transplant_anchor_quality_level",
                "post_phase_allowed_bool",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="order_count")
        .sort_values(
            [
                "relation",
                "assigned_phase",
                "transplant_anchor_quality_level",
                "post_phase_allowed_bool",
            ],
            kind="stable",
        )
    )
    blocked_context = (
        blocked.groupby(
            [
                "transplant_phase",
                "transplant_anchor_quality_level",
                "short_gap_bool",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="order_count")
        .sort_values(
            [
                "transplant_phase",
                "transplant_anchor_quality_level",
                "short_gap_bool",
            ],
            kind="stable",
        )
    )

    summary = {
        "orders": int(len(orders)),
        "patients_represented": int(orders["subject_id"].nunique()),
        "cohort_patients": int(cohort_unique["subject_id"].nunique()),
        "admissions_represented": int(orders["hadm_id"].nunique()),
        "cohort_admissions": int(cohort_unique["hadm_id"].nunique()),
        "admission_coverage_percent": _pct(
            int(orders["hadm_id"].nunique()),
            int(cohort_unique["hadm_id"].nunique()),
        ),
        "admissions_without_usable_orders": int(len(admissions_without_orders)),
        "order_admissions_outside_cohort": int(len(extra_order_admissions)),
        "order_bundles": int(orders["bundle_id"].nunique()),
        "orders_with_preceding_creatinine": orders_with_creatinine,
        "creatinine_coverage_percent": _pct(orders_with_creatinine, len(orders)),
        "orders_without_preceding_creatinine": int(
            len(orders) - orders_with_creatinine
        ),
        "acute_renal_deterioration_order_rows": acute_orders,
        "acute_renal_deterioration_order_percent": _pct(
            acute_orders, len(orders)
        ),
        "acute_renal_deterioration_bundles": acute_bundles,
        "acute_renal_deterioration_admissions": acute_admissions,
        "acute_renal_deterioration_patients": acute_patients,
        "crrt_order_rows": crrt_orders,
        "crrt_order_percent": _pct(crrt_orders, len(orders)),
        "transplant_episodes_represented": int(
            orders["transplant_episode_id"].dropna().nunique()
        ),
        "orders_missing_episode_assignment": int(
            orders["transplant_episode_id"].isna().sum()
        ),
        "index_admission_preprocedure_order_rows": int(len(pre)),
        "index_admission_preprocedure_patients": int(pre["subject_id"].nunique()),
        "index_admission_preprocedure_admissions": int(pre["hadm_id"].nunique()),
        "index_admission_preprocedure_bundles": int(pre["bundle_id"].nunique()),
        "preprocedure_all_in_transplant_episode_admission": pre_all_index,
        "preprocedure_all_before_anchor_date": pre_all_before_anchor,
        "preprocedure_orders_with_posttransplant_phase_permission": pre_post_allowed_count,
        "preprocedure_orders_using_unknown_protocol_phase": pre_protocol_unknown_count,
        "transplant_day_time_uncertain_order_rows": int(len(day_uncertain)),
        "transplant_day_time_uncertain_patients": int(
            day_uncertain["subject_id"].nunique()
        ),
        "transplant_day_time_uncertain_admissions": int(
            day_uncertain["hadm_id"].nunique()
        ),
        "transplant_day_all_in_transplant_episode_admission": day_all_index,
        "transplant_day_all_on_anchor_calendar_date": day_all_same_calendar_date,
        "transplant_day_orders_with_posttransplant_phase_permission": day_post_allowed_count,
        "transplant_day_orders_using_unknown_protocol_phase": day_protocol_unknown_count,
        "legacy_pre_transplant_observed_order_rows": int(len(legacy_pre)),
        "phase_permission_alias_mismatches": alias_mismatch_count,
        "peri_transplant_flag_mismatches": peri_flag_mismatch_count,
        "maintenance_gt365_order_rows": int(len(maintenance)),
        "maintenance_gt365_patients": int(maintenance["subject_id"].nunique()),
        "maintenance_gt365_admissions": int(maintenance["hadm_id"].nunique()),
        "maintenance_gt365_bundles": int(maintenance["bundle_id"].nunique()),
        "maintenance_gt365_min_days": _safe_min(maintenance_days),
        "maintenance_gt365_median_days": _safe_median(maintenance_days),
        "maintenance_gt365_max_days": _safe_max(maintenance_days),
        "maintenance_gt365_orders_in_admissions_starting_within_365d": maintenance_started_within_365,
        "maintenance_gt365_all_valid_full_admission_spillover": maintenance_all_full_admission_spillover,
        "post_transplant_phase_context_allowed_orders": int(
            orders["post_phase_allowed_bool"].sum()
        ),
        "post_transplant_phase_context_blocked_orders": int(
            (~orders["post_phase_allowed_bool"]).sum()
        ),
        "peri_transplant_order_rows": int(orders["peri_bool"].sum()),
        "short_gap_candidate_order_rows": int(orders["short_gap_bool"].sum()),
    }

    fatal: list[str] = []
    documented_limitations: list[str] = []

    if summary["patients_represented"] != summary["cohort_patients"]:
        fatal.append("not all cohort patients are represented in the order timeline")
    if summary["order_admissions_outside_cohort"]:
        fatal.append("the order timeline contains admissions outside the frozen cohort")
    if summary["orders_missing_episode_assignment"]:
        fatal.append("some order rows have no transplant-episode assignment")
    if not pre_all_index:
        fatal.append("some pre-procedure orders occur outside the assigned transplant admission")
    if not pre_all_before_anchor:
        fatal.append("some pre-procedure-labelled orders are not before the anchor date")
    if pre_post_allowed_count:
        fatal.append(
            "pre-procedure orders retain post-transplant phase-specific protocol permission"
        )
    if pre_protocol_unknown_count != len(pre):
        fatal.append(
            "not all pre-procedure orders are mapped to the conservative unknown protocol phase"
        )
    if not day_all_index:
        fatal.append("some transplant-day time-uncertain orders occur outside the index admission")
    if not day_all_same_calendar_date:
        fatal.append("some transplant-day time-uncertain orders are not on the anchor date")
    if day_post_allowed_count:
        fatal.append(
            "transplant-day time-uncertain orders retain post-transplant phase permission"
        )
    if day_protocol_unknown_count != len(day_uncertain):
        fatal.append(
            "not all transplant-day time-uncertain orders use the conservative unknown protocol phase"
        )
    if len(legacy_pre):
        fatal.append(
            "legacy pre_transplant_observed rows remain in the primary post-transplant cohort"
        )
    if alias_mismatch_count:
        fatal.append(
            "phase_specific_context_allowed is not an exact backward-compatible alias of post_transplant_phase_context_allowed"
        )
    if peri_flag_mismatch_count:
        fatal.append("peri_transplant_order does not match the two peri-transplant phases")
    if not maintenance_all_full_admission_spillover:
        fatal.append(
            "some >365-day order rows do not arise from admissions that began within the primary 365-day horizon"
        )

    if admissions_without_orders:
        documented_limitations.append(
            f"{len(admissions_without_orders)} eligible admissions contain no usable timestamped prescription rows and are excluded from order-level denominators"
        )
    if len(maintenance):
        documented_limitations.append(
            f"{len(maintenance)} order rows occur after day 365 because complete admissions beginning within the first-year horizon are retained; strict order-time truncation remains a sensitivity analysis"
        )
    if len(pre):
        documented_limitations.append(
            f"{len(pre)} pre-procedure order rows are retained as peri-transplant decision epochs and longitudinal history, but cannot use ordinary post-transplant phase narrowing"
        )
    if len(day_uncertain):
        documented_limitations.append(
            f"{len(day_uncertain)} transplant-day orders lack a reliable procedure time and are handled conservatively"
        )

    summary["fatal_findings"] = fatal
    summary["documented_limitations"] = documented_limitations

    evidence_json = paths.evidence_dir / "01a_timeline_semantics_audit.json"
    evidence_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(phase_quality, paths.evidence_dir / "01a_phase_by_anchor_quality.csv")
    _write_csv(phase_gate, paths.evidence_dir / "01a_phase_by_context_permission.csv")
    _write_csv(maintenance_context, paths.evidence_dir / "01a_gt365_context.csv")
    _write_csv(blocked_context, paths.evidence_dir / "01a_blocked_context.csv")
    _write_csv(
        missing_admission_summary,
        paths.evidence_dir / "01a_admissions_without_orders.csv",
    )

    print("=" * 78)
    print("STAGE 01A - CORRECTED TIMELINE SEMANTICS AND HORIZON AUDIT")
    print("=" * 78)
    ordered_keys = [
        "orders",
        "patients_represented",
        "cohort_patients",
        "admissions_represented",
        "cohort_admissions",
        "admission_coverage_percent",
        "admissions_without_usable_orders",
        "order_admissions_outside_cohort",
        "order_bundles",
        "orders_with_preceding_creatinine",
        "creatinine_coverage_percent",
        "orders_without_preceding_creatinine",
        "acute_renal_deterioration_order_rows",
        "acute_renal_deterioration_order_percent",
        "acute_renal_deterioration_bundles",
        "acute_renal_deterioration_admissions",
        "acute_renal_deterioration_patients",
        "crrt_order_rows",
        "crrt_order_percent",
        "transplant_episodes_represented",
        "orders_missing_episode_assignment",
        "index_admission_preprocedure_order_rows",
        "index_admission_preprocedure_patients",
        "index_admission_preprocedure_admissions",
        "index_admission_preprocedure_bundles",
        "preprocedure_all_in_transplant_episode_admission",
        "preprocedure_all_before_anchor_date",
        "preprocedure_orders_with_posttransplant_phase_permission",
        "preprocedure_orders_using_unknown_protocol_phase",
        "transplant_day_time_uncertain_order_rows",
        "transplant_day_time_uncertain_patients",
        "transplant_day_time_uncertain_admissions",
        "transplant_day_all_in_transplant_episode_admission",
        "transplant_day_all_on_anchor_calendar_date",
        "transplant_day_orders_with_posttransplant_phase_permission",
        "transplant_day_orders_using_unknown_protocol_phase",
        "legacy_pre_transplant_observed_order_rows",
        "phase_permission_alias_mismatches",
        "peri_transplant_flag_mismatches",
        "maintenance_gt365_order_rows",
        "maintenance_gt365_patients",
        "maintenance_gt365_admissions",
        "maintenance_gt365_bundles",
        "maintenance_gt365_min_days",
        "maintenance_gt365_median_days",
        "maintenance_gt365_max_days",
        "maintenance_gt365_orders_in_admissions_starting_within_365d",
        "maintenance_gt365_all_valid_full_admission_spillover",
        "post_transplant_phase_context_allowed_orders",
        "post_transplant_phase_context_blocked_orders",
        "peri_transplant_order_rows",
        "short_gap_candidate_order_rows",
    ]
    for key in ordered_keys:
        print(f"{key}: {summary[key]}")

    print("\nPhase by anchor quality:")
    print(phase_quality.to_string(index=False))
    print("\nProtocol phase and permission by actual phase:")
    print(phase_gate.to_string(index=False))
    if not maintenance_context.empty:
        print("\nOrders beyond 365 days retained as complete-admission spillover:")
        print(maintenance_context.to_string(index=False))
    if not missing_admission_summary.empty:
        print("\nCohort admissions without usable prescription orders:")
        print(missing_admission_summary.to_string(index=False))

    print(f"\nSaved aggregate audit: {evidence_json}")
    print(f"Saved supporting aggregate tables under: {paths.evidence_dir}")

    if fatal:
        print("\nSTAGE 01A: FAIL")
        for item in fatal:
            print(f" - {item}")
        raise SystemExit(1)

    if documented_limitations:
        print("\nDocumented limitations (do not block progression):")
        for item in documented_limitations:
            print(f" - {item}")

    print("\nSTAGE 01A: COMPLETE - PASS")


if __name__ == "__main__":
    main()
