from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clarity_v3.project import ensure_output_dirs, load_project_paths


def _bool(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().isin({"1","true","yes","y","t"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit provider-order-time linkage and order-bundle integrity")
    parser.add_argument("--paths", required=True)
    args = parser.parse_args()
    paths = load_project_paths(Path(args.paths).resolve())
    ensure_output_dirs(paths)
    orders = pd.read_csv(paths.orders_file, low_memory=False)
    required = {
        "subject_id","hadm_id","order_id","bundle_id","timestamp",
        "exposure_starttime","decision_time_source","poe_link_status",
        "source_poe_id","source_rx_row_index"
    }
    missing = required - set(orders.columns)
    if missing:
        raise ValueError(f"Timeline missing V3.2 columns: {sorted(missing)}")
    orders["timestamp"] = pd.to_datetime(orders["timestamp"], errors="coerce")
    orders["exposure_starttime"] = pd.to_datetime(orders["exposure_starttime"], errors="coerce")
    if "provider_ordertime" in orders:
        orders["provider_ordertime"] = pd.to_datetime(orders["provider_ordertime"], errors="coerce")
    grouped = orders.groupby("bundle_id", dropna=False)
    exact = orders["poe_link_status"].eq("exact_subject_admission_order_match")
    fallback = ~exact
    summary = {
        "order_rows": int(len(orders)),
        "order_bundles": int(orders["bundle_id"].nunique()),
        "bundles_spanning_multiple_subjects": int((grouped["subject_id"].nunique() > 1).sum()),
        "bundles_spanning_multiple_admissions": int((grouped["hadm_id"].nunique() > 1).sum()),
        "exact_poe_link_rows": int(exact.sum()),
        "fallback_rows": int(fallback.sum()),
        "poe_id_found_other_admission_rows": int(orders["poe_link_status"].eq("poe_id_found_other_admission").sum()),
        "poe_id_not_found_rows": int(orders["poe_link_status"].eq("poe_id_not_found").sum()),
        "missing_poe_id_rows": int(orders["poe_link_status"].eq("missing_poe_id").sum()),
        "exact_rows_not_using_ordertime": int((exact & ~orders["decision_time_source"].eq("poe_ordertime")).sum()),
        "fallback_rows_not_using_starttime": int((fallback & ~orders["decision_time_source"].eq("prescription_starttime_fallback")).sum()),
        "exact_timestamp_mismatches": int((exact & orders["provider_ordertime"].notna() & orders["timestamp"].ne(orders["provider_ordertime"])).sum()),
        "fallback_timestamp_mismatches": int((fallback & orders["timestamp"].ne(orders["exposure_starttime"])).sum()),
        "invalid_poe_rows_grouped_with_another_row": int(
            orders.loc[fallback, "bundle_id"].map(orders["bundle_id"].value_counts()).gt(1).sum()
        ),
        "poe_bundles_with_multiple_starttimes": int(
            (orders.loc[exact].groupby("bundle_id")["exposure_starttime"].nunique() > 1).sum()
        ),
    }
    delta = (orders.loc[exact, "exposure_starttime"] - orders.loc[exact, "timestamp"]).dt.total_seconds()/3600
    if not delta.empty:
        summary.update({
            "start_minus_order_hours_median": float(delta.median()),
            "start_minus_order_hours_p95": float(delta.quantile(0.95)),
            "start_minus_order_hours_max": float(delta.max()),
            "start_after_ordertime_rows": int((delta > 0).sum()),
            "start_before_ordertime_rows": int((delta < 0).sum()),
        })
    fatal = []
    for key in (
        "bundles_spanning_multiple_subjects",
        "bundles_spanning_multiple_admissions",
        "exact_rows_not_using_ordertime",
        "fallback_rows_not_using_starttime",
        "exact_timestamp_mismatches",
        "fallback_timestamp_mismatches",
        "invalid_poe_rows_grouped_with_another_row",
    ):
        if summary[key]:
            fatal.append(f"{key}={summary[key]}")
    output = paths.evidence_dir / "05a_order_bundle_integrity_audit_v32.json"
    output.write_text(json.dumps({**summary, "fatal_findings": fatal}, indent=2), encoding="utf-8")
    print("="*78)
    print("STAGE 05A - PROVIDER-ORDER-TIME AND BUNDLE INTEGRITY AUDIT")
    print("="*78)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("Saved aggregate audit:", output)
    if fatal:
        print("\nSTAGE 05A: FAIL")
        for item in fatal:
            print(" -", item)
        raise SystemExit(1)
    print("\nDocumented fallback limitation:")
    print(" - Rows without an exact subject+admission+poe_id link use prescription starttime and remain standalone bundles.")
    print("\nSTAGE 05A: COMPLETE - PASS")


if __name__ == "__main__":
    main()
