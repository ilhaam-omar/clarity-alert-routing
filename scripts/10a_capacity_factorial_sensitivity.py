from __future__ import annotations

import pandas as pd

from common import atomic_csv, banner, parser
from clarity_v3.capacity import apply_dual_capacity, capacity_summary
from clarity_v3.models import CapacityScenario
from clarity_v3.project import ensure_output_dirs, load_project_paths


FRONTLINE_RATES = (0.5, 1.0, 2.0)
EXPERT_COVERAGE = (0.25, 0.50, 0.75)


def _scenario_name(frontline_rate: float, expert_fraction: float) -> str:
    front = str(frontline_rate).replace(".", "p")
    expert = int(round(100 * expert_fraction))
    return f"frontline_{front}_expert_{expert}pct"


def _matrix(
    summary: pd.DataFrame,
    value_column: str,
    *,
    index_name: str = "expert_review_coverage_percent",
    column_name: str = "frontline_alerts_per_100_order_bundles",
) -> pd.DataFrame:
    matrix = summary.pivot(
        index=index_name,
        columns=column_name,
        values=value_column,
    ).sort_index().sort_index(axis=1)
    matrix.index.name = index_name
    matrix.columns.name = column_name
    return matrix.reset_index()


def main() -> None:
    args = parser(
        "Evaluate physician-alert and enhanced-pharmacist-review capacity independently"
    ).parse_args()
    banner("STAGE 10A - FACTORIAL DUAL-CAPACITY SENSITIVITY")

    paths = load_project_paths(args.paths)
    ensure_output_dirs(paths)

    if not paths.bundles_file.exists() or not paths.orders_file.exists():
        raise FileNotFoundError("Run Stages 01-05 before Stage 10A")

    bundles = pd.read_csv(paths.bundles_file, low_memory=False)
    orders = pd.read_csv(paths.orders_file, usecols=["bundle_id"], low_memory=False)
    total_order_bundles = int(orders["bundle_id"].astype(str).nunique())

    scenarios: list[CapacityScenario] = []
    scenario_inputs: list[dict] = []
    for frontline_rate in FRONTLINE_RATES:
        for expert_fraction in EXPERT_COVERAGE:
            name = _scenario_name(frontline_rate, expert_fraction)
            scenarios.append(
                CapacityScenario(
                    name=name,
                    allocation_scope="cohort_normalized",
                    frontline_alerts_per_100_order_bundles=frontline_rate,
                    expert_review_coverage_fraction=expert_fraction,
                    unconstrained=False,
                    critical_reviews_spill_to_frontline=True,
                    description=(
                        "Factorial Phase-1 stress test. Values are normalized scenario "
                        "assumptions, not estimates of observed staffing."
                    ),
                )
            )
            scenario_inputs.append(
                {
                    "scenario": name,
                    "frontline_alerts_per_100_order_bundles": frontline_rate,
                    "expert_review_coverage_fraction": expert_fraction,
                    "expert_review_coverage_percent": 100.0 * expert_fraction,
                }
            )

    print("[DESIGN] The two capacity dimensions are varied independently.")
    print("[DESIGN] Frontline values are nominal direct-alert densities per 100 prescribing bundles.")
    print("[DESIGN] Expert values are the fraction of pre-capacity enhanced-review demand served.")
    print("[DESIGN] These are MIMIC Phase-1 stress tests, not observed staffing estimates.")
    print(f"All prescribing bundles (denominator): {total_order_bundles:,}")
    print(f"Affected bundles entering allocation: {len(bundles):,}")
    print(f"Factorial scenarios: {len(scenarios)}")

    frames: list[pd.DataFrame] = []
    safety_rows: list[dict] = []
    for index, scenario in enumerate(scenarios, start=1):
        print(
            f"[{index}/{len(scenarios)}] {scenario.name}: "
            f"frontline={scenario.frontline_alerts_per_100_order_bundles}/100; "
            f"expert={100 * float(scenario.expert_review_coverage_fraction or 0):.0f}%"
        )
        result = apply_dual_capacity(
            bundles,
            scenario,
            total_order_bundles=total_order_bundles,
        )
        if len(result) != len(bundles):
            raise AssertionError(
                f"{scenario.name}: expected {len(bundles):,} decisions, observed {len(result):,}"
            )
        protected_alert = result["protected_alert"].astype(str).str.lower().isin(
            {"1", "true", "yes", "y", "t"}
        )
        protected_review = result["protected_review"].astype(str).str.lower().isin(
            {"1", "true", "yes", "y", "t"}
        )
        alert_violations = int(
            (protected_alert & ~result["final_action"].eq("alert")).sum()
        )
        review_violations = int(
            (
                protected_review
                & ~result["final_action"].isin(["alert", "escalate"])
            ).sum()
        )
        safety_rows.append(
            {
                "scenario": scenario.name,
                "protected_alert_violations": alert_violations,
                "protected_review_violations": review_violations,
            }
        )
        frames.append(result)

    final = pd.concat(frames, ignore_index=True)
    summary = capacity_summary(final).merge(
        pd.DataFrame(scenario_inputs),
        on="scenario",
        how="left",
        validate="one_to_one",
    )
    safety = pd.DataFrame(safety_rows)
    summary = summary.merge(safety, on="scenario", how="left", validate="one_to_one")
    summary["protected_safety_violations"] = (
        summary["protected_alert_violations"]
        + summary["protected_review_violations"]
    )

    summary = summary.sort_values(
        [
            "frontline_alerts_per_100_order_bundles",
            "expert_review_coverage_fraction",
        ],
        kind="stable",
    ).reset_index(drop=True)

    # Monotonicity checks expected from this deterministic allocation design.
    for frontline_rate, group in summary.groupby(
        "frontline_alerts_per_100_order_bundles", sort=True
    ):
        ordered = group.sort_values("expert_review_coverage_fraction")
        if not ordered["escalate"].is_monotonic_increasing:
            raise AssertionError(
                f"Expert reviews should not decrease as expert coverage rises at frontline={frontline_rate}"
            )
        if not ordered["alert"].is_monotonic_decreasing:
            raise AssertionError(
                f"Direct alerts should not increase as expert coverage rises at frontline={frontline_rate}"
            )

    if int(summary["protected_safety_violations"].sum()) != 0:
        raise AssertionError("A factorial capacity scenario violated a protected safety pathway")

    output = paths.evidence_dir / "10a_capacity_factorial_summary.csv"
    atomic_csv(summary, output)

    matrices = {
        "10a_direct_alert_rate_matrix.csv": _matrix(
            summary, "final_alerts_per_100_order_bundles"
        ),
        "10a_expert_review_rate_matrix.csv": _matrix(
            summary, "final_expert_reviews_per_100_order_bundles"
        ),
        "10a_critical_spillover_matrix.csv": _matrix(
            summary, "critical_spillover_to_frontline"
        ),
        "10a_frontline_overrun_matrix.csv": _matrix(
            summary, "frontline_overrun_bundles"
        ),
        "10a_interruption_reduction_matrix.csv": _matrix(
            summary, "immediate_interruption_reduction_vs_B1_percent"
        ),
    }
    for filename, frame in matrices.items():
        atomic_csv(frame, paths.evidence_dir / filename)

    display_columns = [
        "frontline_alerts_per_100_order_bundles",
        "expert_review_coverage_percent",
        "nominal_frontline_slots",
        "nominal_expert_slots",
        "alert",
        "escalate",
        "critical_spillover_to_frontline",
        "frontline_overrun_bundles",
        "final_alerts_per_100_order_bundles",
        "final_expert_reviews_per_100_order_bundles",
        "immediate_interruption_reduction_vs_B1_percent",
        "protected_safety_violations",
    ]
    print("\nFactorial capacity-response summary:")
    print(summary[display_columns].to_string(index=False))
    print(f"\nSaved aggregate summary: {output}")
    for filename in matrices:
        print(f"Saved matrix: {paths.evidence_dir / filename}")
    print("\nSTAGE 10A: COMPLETE - PASS")


if __name__ == "__main__":
    main()
