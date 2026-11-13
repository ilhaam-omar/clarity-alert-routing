from __future__ import annotations

import pandas as pd

from common import atomic_csv, banner, parser
from clarity_v3.capacity import capacity_summary, run_capacity_scenarios
from clarity_v3.config import load_capacity_scenarios
from clarity_v3.project import ensure_output_dirs, load_project_paths


def main() -> None:
    args = parser(
        "Apply cohort-normalized frontline and expert capacity-response scenarios"
    ).parse_args()
    banner("STAGE 06 - NORMALIZED DUAL-CAPACITY RESPONSE ANALYSIS")
    paths = load_project_paths(args.paths)
    ensure_output_dirs(paths)
    if not paths.bundles_file.exists() or not paths.orders_file.exists():
        raise FileNotFoundError("Run Stages 01-05 first")

    bundles = pd.read_csv(paths.bundles_file, low_memory=False)
    orders = pd.read_csv(paths.orders_file, usecols=["bundle_id"], low_memory=False)
    total_order_bundles = int(orders["bundle_id"].astype(str).nunique())
    scenarios = load_capacity_scenarios(
        paths.root / "config" / "capacity_scenarios.json"
    )

    print("[DESIGN] MIMIC-IV patient dates are not pooled into hospital shifts.")
    print(
        "[DESIGN] Frontline capacity is a nominal alert density per 100 prescribing bundles."
    )
    print(
        "[DESIGN] Expert capacity is the fraction of pre-capacity review demand that can be served."
    )
    print(
        "[DESIGN] Protected immediate alerts and unserved protected reviews may exceed the nominal frontline threshold; overruns are reported, never hidden."
    )
    print(f"All prescribing bundles (capacity denominator): {total_order_bundles:,}")
    print(f"Affected bundles entering capacity allocation: {len(bundles):,}")

    final = run_capacity_scenarios(
        bundles,
        scenarios,
        total_order_bundles=total_order_bundles,
    )
    atomic_csv(final, paths.final_bundles_file)
    summary = capacity_summary(final)
    atomic_csv(summary, paths.evidence_dir / "capacity_scenario_summary.csv")

    expected_rows = len(bundles) * len(scenarios)
    if len(final) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows:,} scenario rows, observed {len(final):,}"
        )
    print(f"Bundles per scenario: {len(bundles):,}")
    print(f"Scenarios: {len(scenarios):,}")
    print(f"Scenario rows: {len(final):,}")
    display_columns = [
        "scenario",
        "nominal_frontline_slots",
        "nominal_expert_slots",
        "protected_must_interrupt_floor",
        "protected_review_alert_floor",
        "protected_immediate_alert_floor",
        "suppress",
        "passive",
        "alert",
        "escalate",
        "critical_spillover_to_frontline",
        "frontline_overrun_bundles",
        "final_alerts_per_100_order_bundles",
        "final_expert_reviews_per_100_order_bundles",
        "expert_demand_served_percent",
        "capacity_binding",
    ]
    print("\nCapacity-response summary:")
    print(summary[display_columns].to_string(index=False))
    print(f"\nSaved private final bundle output: {paths.final_bundles_file}")
    print(
        f"Saved aggregate capacity summary: {paths.evidence_dir / 'capacity_scenario_summary.csv'}"
    )
    print("\nSTAGE 06: COMPLETE - PASS")


if __name__ == "__main__":
    main()
