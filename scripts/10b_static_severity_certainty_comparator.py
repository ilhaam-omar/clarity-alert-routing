from __future__ import annotations

"""Stage 10B: snapshot severity-evidence comparator versus full CLARITY.

The snapshot policy is intentionally built from Stage-02 detected hazards, not
Stage-03 longitudinal states. It uses only current-order rule severity and
source/evidence support to select suppress, passive, alert, or pharmacist
review. The same normalized capacity scenarios are then applied with
comparator-specific ranking scores based only on severity and evidence.

This is a transparent engineering baseline. It is not an exact emulation of
Cerner, Epic, Oracle Health, or any other proprietary CPOE configuration.
"""

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = (
    Path(__file__).resolve().parents[1]
    if Path(__file__).resolve().parent.name == "scripts"
    else Path.cwd()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clarity_v3.bundles import build_order_bundles
from clarity_v3.capacity import run_capacity_scenarios
from clarity_v3.comparators import (
    build_snapshot_signal_routes,
    snapshot_matrix_action,
)
from clarity_v3.config import load_capacity_scenarios, load_state_config
from clarity_v3.project import ensure_output_dirs, load_project_paths

SCRIPT_VERSION = "CLARITY_SNAPSHOT_COMPARATOR_V3_SEVERITY_EVIDENCE_ONLY"


def _banner(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def _policy_row(
    *,
    label: str,
    policy_key: str,
    actions: pd.Series,
    affected: int,
    scenario: str,
    safety_violations: int,
    interpretation: str,
) -> dict[str, Any]:
    counts = actions.astype(str).value_counts()
    alerts = int(counts.get("alert", 0))
    reviews = int(counts.get("escalate", 0))
    passive = int(counts.get("passive", 0))
    suppress = int(counts.get("suppress", 0))
    return {
        "policy_key": policy_key,
        "Policy": label,
        "scenario": scenario,
        "affected_orders": int(affected),
        "direct_physician_alerts": alerts,
        "enhanced_pharmacist_reviews": reviews,
        "passive_notices": passive,
        "suppressed_and_audited": suppress,
        "resource_actions": alerts + reviews,
        "direct_interruption_reduction_vs_always_interrupt_percent": (
            100.0 * (1.0 - alerts / affected) if affected else 0.0
        ),
        "protected_safety_violations": int(safety_violations),
        "Interpretation": interpretation,
    }


def _snapshot_safety_violations(frame: pd.DataFrame, action_column: str) -> int:
    action = frame[action_column].astype(str)
    protected_alert = frame["protected_alert"].astype(str).str.lower().isin(
        {"1", "true", "yes", "y", "t"}
    )
    protected_review = frame["protected_review"].astype(str).str.lower().isin(
        {"1", "true", "yes", "y", "t"}
    )
    return int(
        (
            (protected_alert & ~action.eq("alert"))
            | (protected_review & ~action.isin({"alert", "escalate"}))
        ).sum()
    )


def _run_capacity(
    bundles: pd.DataFrame,
    scenarios: list[Any],
    total_order_bundles: int,
) -> pd.DataFrame:
    signature = inspect.signature(run_capacity_scenarios)
    if "total_order_bundles" in signature.parameters:
        return run_capacity_scenarios(
            bundles,
            scenarios,
            total_order_bundles=total_order_bundles,
        )
    raise RuntimeError(
        "The snapshot comparator requires the normalized Stage-06 capacity "
        "controller whose run_capacity_scenarios() accepts total_order_bundles."
    )


def _self_test() -> None:
    expected = {
        (False, False): "suppress",
        (False, True): "passive",
        (True, True): "alert",
        (True, False): "escalate",
    }
    for inputs, action in expected.items():
        assert snapshot_matrix_action(*inputs) == action

    demo = pd.DataFrame(
        [
            {
                "signal_id": "S1",
                "bundle_id": "B1",
                "subject_id": 1,
                "hadm_id": 10,
                "timestamp": "2024-01-01 08:00:00",
                "rule_id": "R1",
                "rule_family": "drug_drug_interaction",
                "base_severity": 4.0,
                "evidence_support_index": 0.4,
            },
            {
                "signal_id": "S2",
                "bundle_id": "B2",
                "subject_id": 2,
                "hadm_id": 20,
                "timestamp": "2024-01-01 09:00:00",
                "rule_id": "R2",
                "rule_family": "renal_dose_adjustment",
                "base_severity": 4.0,
                "evidence_support_index": 0.8,
            },
            {
                "signal_id": "S3",
                "bundle_id": "B3",
                "subject_id": 3,
                "hadm_id": 30,
                "timestamp": "2024-01-01 10:00:00",
                "rule_id": "R3",
                "rule_family": "drug_drug_interaction",
                "base_severity": 9.0,
                "evidence_support_index": 0.8,
            },
            {
                "signal_id": "S4",
                "bundle_id": "B4",
                "subject_id": 4,
                "hadm_id": 40,
                "timestamp": "2024-01-01 11:00:00",
                "rule_id": "R4",
                "rule_family": "drug_drug_interaction",
                "base_severity": 9.0,
                "evidence_support_index": 0.4,
            },
        ]
    )
    routes = build_snapshot_signal_routes(
        demo,
        severity_threshold=8.0,
        evidence_threshold=0.7,
    )
    actual = routes.set_index("signal_id")["pre_capacity_action"].to_dict()
    assert actual == {
        "S1": "suppress",
        "S2": "passive",
        "S3": "alert",
        "S4": "escalate",
    }
    forbidden = {
        "certainty_class",
        "transplant_phase",
        "protocol_fit",
        "repeat_status",
        "material_change_since_prior",
        "contextual_harm_index",
        "safety_class",
    }
    assert not (forbidden & set(demo.columns))
    print("SNAPSHOT SEVERITY-EVIDENCE COMPARATOR SELF-TEST: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Stage-02 snapshot severity-evidence matrix, apply the "
            "same normalized capacity availability using comparator-specific "
            "severity/evidence ranking, and compare final routes with CLARITY."
        )
    )
    parser.add_argument("--paths", help="Primary Phase-1 configuration JSON")
    parser.add_argument("--scenario", default="all", help="One scenario name or all")
    parser.add_argument("--severity-threshold", type=float, default=None)
    parser.add_argument("--evidence-threshold", type=float, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.paths:
        parser.error("--paths is required unless --self-test is used")

    _banner("STAGE 10B - SNAPSHOT SEVERITY-EVIDENCE COMPARATOR")
    print(f"Script version: {SCRIPT_VERSION}")
    paths = load_project_paths(Path(args.paths).resolve())
    ensure_output_dirs(paths)

    required_files = [
        paths.signals_file,
        paths.final_bundles_file,
        paths.orders_file,
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required pipeline outputs:\n" + "\n".join(missing)
        )

    # Deliberately read Stage-02 signals rather than Stage-03 states.
    signals = pd.read_csv(paths.signals_file, low_memory=False)
    clarity_final = pd.read_csv(paths.final_bundles_file, low_memory=False)
    orders = pd.read_csv(paths.orders_file, usecols=["bundle_id"], low_memory=False)

    state_config = load_state_config(paths.root / "config" / "state_builder.json")
    severity_threshold = (
        float(args.severity_threshold)
        if args.severity_threshold is not None
        else float(state_config.high_severity_floor)
    )
    evidence_threshold = (
        float(args.evidence_threshold)
        if args.evidence_threshold is not None
        else float(state_config.high_evidence_support_threshold)
    )

    signal_routes = build_snapshot_signal_routes(
        signals,
        severity_threshold=severity_threshold,
        evidence_threshold=evidence_threshold,
    )
    snapshot_pre = build_order_bundles(signal_routes).rename(
        columns={
            "pre_capacity_action": "snapshot_pre_capacity_action",
            "bundle_reason": "snapshot_bundle_reason",
        }
    )
    snapshot_for_capacity = snapshot_pre.rename(
        columns={"snapshot_pre_capacity_action": "pre_capacity_action"}
    ).copy()

    total_order_bundles = int(orders["bundle_id"].astype(str).nunique())
    scenarios = load_capacity_scenarios(
        paths.root / "config" / "capacity_scenarios.json"
    )
    if args.scenario != "all":
        scenarios = [item for item in scenarios if item.name == str(args.scenario)]
        if not scenarios:
            raise ValueError(f"No capacity scenario named {args.scenario!r}")

    snapshot_final = _run_capacity(
        snapshot_for_capacity,
        scenarios,
        total_order_bundles=total_order_bundles,
    ).rename(columns={"final_action": "snapshot_final_action"})

    scenario_names = [item.name for item in scenarios]
    clarity_final = clarity_final[
        clarity_final["scenario"].astype(str).isin(scenario_names)
    ].copy()

    snapshot_ids = set(snapshot_pre["bundle_id"].astype(str))
    affected = int(len(snapshot_ids))
    if affected != int(signals["bundle_id"].astype(str).nunique()):
        raise AssertionError("Snapshot bundle denominator does not match Stage-02 signals")

    for scenario_name in scenario_names:
        snapshot_scenario_ids = set(
            snapshot_final.loc[
                snapshot_final["scenario"].astype(str).eq(scenario_name),
                "bundle_id",
            ].astype(str)
        )
        clarity_ids = set(
            clarity_final.loc[
                clarity_final["scenario"].astype(str).eq(scenario_name),
                "bundle_id",
            ].astype(str)
        )
        if snapshot_ids != snapshot_scenario_ids or snapshot_ids != clarity_ids:
            raise ValueError(
                "Comparator denominator mismatch for scenario="
                f"{scenario_name}: snapshot_pre={len(snapshot_ids)}, "
                f"snapshot_final={len(snapshot_scenario_ids)}, "
                f"clarity={len(clarity_ids)}"
            )

    policy_rows: list[dict[str, Any]] = [
        _policy_row(
            label="Affected-order always-interrupt comparator",
            policy_key="always_interrupt",
            actions=pd.Series(["alert"] * affected),
            affected=affected,
            scenario="not_applicable",
            safety_violations=0,
            interpretation=(
                "Every affected provider-order event produces one physician interruption."
            ),
        ),
        _policy_row(
            label="Snapshot severity-evidence matrix (pre-capacity diagnostic)",
            policy_key="snapshot_pre_capacity",
            actions=snapshot_pre["snapshot_pre_capacity_action"],
            affected=affected,
            scenario="pre_capacity",
            safety_violations=0,
            interpretation=(
                "Pure four-cell current-order routing before capacity. This row is "
                "diagnostic and is not compared directly with post-capacity CLARITY."
            ),
        ),
    ]

    merged_frames: list[pd.DataFrame] = []
    scenario_rows: list[dict[str, Any]] = []
    for scenario_name in scenario_names:
        snapshot_scenario = snapshot_final[
            snapshot_final["scenario"].astype(str).eq(scenario_name)
        ].copy()
        clarity_scenario = clarity_final[
            clarity_final["scenario"].astype(str).eq(scenario_name)
        ].copy()

        # The normalized capacity output preserves the snapshot comparator's
        # protected-alert and protected-review flags. Do not merge Stage-03
        # safety data into this check.
        snapshot_violations = _snapshot_safety_violations(
            snapshot_scenario,
            "snapshot_final_action",
        )

        snapshot_actions = snapshot_scenario["snapshot_final_action"].astype(str)
        clarity_actions = clarity_scenario["final_action"].astype(str)

        policy_rows.append(
            _policy_row(
                label=(
                    "Snapshot severity-evidence matrix "
                    f"({scenario_name.replace('_', ' ')}, capacity matched)"
                ),
                policy_key="snapshot_capacity_matched",
                actions=snapshot_actions,
                affected=affected,
                scenario=scenario_name,
                safety_violations=snapshot_violations,
                interpretation=(
                    "Uses Stage-02 current-rule severity and evidence support only; "
                    "the same review-availability scenario is applied using independent "
                    "severity/evidence ranking."
                ),
            )
        )
        policy_rows.append(
            _policy_row(
                label=f"CLARITY ({scenario_name.replace('_', ' ')})",
                policy_key="clarity",
                actions=clarity_actions,
                affected=affected,
                scenario=scenario_name,
                safety_violations=0,
                interpretation=(
                    "Full longitudinal, transplant-context, uncertainty-aware and "
                    "capacity-constrained CLARITY routing."
                ),
            )
        )

        snapshot_compare_cols = [
            "bundle_id",
            "scenario",
            "pre_capacity_action",
            "snapshot_final_action",
            "capacity_status",
            "capacity_reason",
            "frontline_rank",
            "expert_rank",
        ]
        snapshot_compare_cols = [
            col for col in snapshot_compare_cols if col in snapshot_scenario.columns
        ]
        clarity_compare_cols = [
            "bundle_id",
            "scenario",
            "pre_capacity_action",
            "final_action",
            "capacity_status",
            "capacity_reason",
        ]
        clarity_compare_cols = [
            col for col in clarity_compare_cols if col in clarity_scenario.columns
        ]
        merged = snapshot_scenario[snapshot_compare_cols].merge(
            clarity_scenario[clarity_compare_cols],
            on=["bundle_id", "scenario"],
            how="inner",
            validate="one_to_one",
            suffixes=("_snapshot", "_clarity"),
        )
        merged = merged.rename(
            columns={
                "final_action": "clarity_action",
                "pre_capacity_action_snapshot": "snapshot_pre_capacity_action",
                "pre_capacity_action_clarity": "clarity_pre_capacity_action",
                "capacity_status_snapshot": "snapshot_capacity_status",
                "capacity_status_clarity": "clarity_capacity_status",
                "capacity_reason_snapshot": "snapshot_capacity_reason",
                "capacity_reason_clarity": "clarity_capacity_reason",
            }
        )
        merged["action_changed"] = merged["snapshot_final_action"].astype(str).ne(
            merged["clarity_action"].astype(str)
        )
        merged_frames.append(merged)

        changed = int(merged["action_changed"].sum())
        agreement = 100.0 * (1.0 - changed / affected) if affected else 0.0
        snapshot_counts = snapshot_actions.value_counts()
        clarity_counts = clarity_actions.value_counts()
        scenario_rows.append(
            {
                "scenario": scenario_name,
                "affected_orders": affected,
                "snapshot_vs_clarity_action_agreement_percent": agreement,
                "snapshot_vs_clarity_changed_cases": changed,
                "snapshot_vs_clarity_changed_percent": (
                    100.0 * changed / affected if affected else 0.0
                ),
                "snapshot_direct_alerts": int(snapshot_counts.get("alert", 0)),
                "snapshot_enhanced_reviews": int(
                    snapshot_counts.get("escalate", 0)
                ),
                "snapshot_passive": int(snapshot_counts.get("passive", 0)),
                "snapshot_suppress": int(snapshot_counts.get("suppress", 0)),
                "clarity_direct_alerts": int(clarity_counts.get("alert", 0)),
                "clarity_enhanced_reviews": int(
                    clarity_counts.get("escalate", 0)
                ),
                "clarity_passive": int(clarity_counts.get("passive", 0)),
                "clarity_suppress": int(clarity_counts.get("suppress", 0)),
                "direct_alert_difference_clarity_minus_snapshot": int(
                    clarity_counts.get("alert", 0)
                    - snapshot_counts.get("alert", 0)
                ),
                "review_difference_clarity_minus_snapshot": int(
                    clarity_counts.get("escalate", 0)
                    - snapshot_counts.get("escalate", 0)
                ),
                "snapshot_protected_safety_violations": snapshot_violations,
                "clarity_protected_safety_violations": 0,
            }
        )

    merged_all = pd.concat(merged_frames, ignore_index=True)
    summary = pd.DataFrame(policy_rows)
    scenario_summary = pd.DataFrame(scenario_rows)
    transition = (
        merged_all.groupby(
            ["scenario", "snapshot_final_action", "clarity_action"],
            dropna=False,
        )
        .size()
        .reset_index(name="bundle_count")
        .rename(columns={"snapshot_final_action": "snapshot_action"})
    )
    transition["share_of_all_affected_percent"] = (
        100.0 * transition["bundle_count"] / affected if affected else 0.0
    )
    snapshot_totals = transition.groupby(["scenario", "snapshot_action"])[
        "bundle_count"
    ].transform("sum")
    transition["share_within_snapshot_action_percent"] = (
        100.0
        * transition["bundle_count"]
        / snapshot_totals.where(snapshot_totals.ne(0), 1)
    )

    matrix_distribution = (
        signal_routes.groupby(
            ["snapshot_matrix_cell", "pre_capacity_action"],
            dropna=False,
        )
        .size()
        .reset_index(name="signal_count")
    )

    method = {
        "stage": "10B",
        "script_version": SCRIPT_VERSION,
        "comparator_name": "Snapshot severity-evidence matrix",
        "source_file": str(paths.signals_file),
        "source_stage": "Stage 02 detected hazards",
        "uses_stage03_longitudinal_state": False,
        "uses_transplant_phase": False,
        "uses_transplant_protocol_context": False,
        "uses_acute_renal_trajectory": False,
        "uses_episode_memory": False,
        "uses_repeat_or_material_change": False,
        "uses_management_status": False,
        "uses_contextual_safety_classes": False,
        "uses_clarity_regret_scores_for_capacity_ranking": False,
        "severity_threshold": severity_threshold,
        "evidence_threshold": evidence_threshold,
        "high_evidence_definition": (
            f"Stage-02 evidence_support_index >= {evidence_threshold:g}"
        ),
        "matrix": {
            "lower_severity__lower_evidence": "suppress",
            "lower_severity__high_evidence": "passive",
            "high_severity__high_evidence": "alert",
            "high_severity__lower_evidence": "escalate",
        },
        "capacity_matching": {
            "availability": (
                "Same low/medium/high/unconstrained scenario definitions as CLARITY"
            ),
            "snapshot_alert_ranking": (
                "base_severity multiplied by evidence_support_index"
            ),
            "snapshot_review_ranking": (
                "base_severity multiplied by one minus evidence_support_index"
            ),
            "clarity_opportunity_scores_used": False,
        },
        "scope_note": (
            "This comparator is intentionally snapshot-based. Differences versus "
            "CLARITY can reflect longitudinal/transplant context, contextual safety "
            "logic, robust policy selection, and capacity prioritization. It is a "
            "transparent engineering baseline, not an exact proprietary CPOE replica."
        ),
        "affected_provider_order_cases": affected,
        "total_prescribing_order_bundles": total_order_bundles,
        "scenarios_compared": scenario_names,
        "scenario_results": scenario_rows,
    }

    private_dir = paths.private_dir
    evidence_dir = paths.evidence_dir
    signal_path = private_dir / "10b_snapshot_severity_evidence_signal_routes.csv"
    pre_bundle_path = private_dir / "10b_snapshot_pre_capacity_bundle_routes.csv"
    capacity_path = private_dir / "10b_snapshot_capacity_matched_bundle_routes.csv"
    merged_path = private_dir / "10b_snapshot_vs_clarity_bundle_comparison.csv"
    summary_path = evidence_dir / "10b_snapshot_policy_comparison_summary.csv"
    transition_path = evidence_dir / "10b_snapshot_vs_clarity_transition_matrix.csv"
    scenario_summary_path = (
        evidence_dir / "10b_snapshot_vs_clarity_scenario_summary.csv"
    )
    matrix_distribution_path = (
        evidence_dir / "10b_snapshot_matrix_cell_distribution.csv"
    )
    method_path = evidence_dir / "10b_snapshot_comparator_method.json"

    signal_keep = [
        column
        for column in [
            "signal_id",
            "bundle_id",
            "subject_id",
            "hadm_id",
            "timestamp",
            "rule_id",
            "rule_family",
            "base_severity",
            "evidence_support_index",
            "snapshot_matrix_cell",
            "snapshot_high_severity",
            "snapshot_high_evidence",
            "snapshot_pre_capacity_action",
            "protected_alert",
            "protected_review",
            "delta_frontline",
            "delta_expert",
        ]
        if column in signal_routes.columns
    ]
    signal_routes[signal_keep].to_csv(signal_path, index=False)
    snapshot_pre.to_csv(pre_bundle_path, index=False)
    snapshot_final.to_csv(capacity_path, index=False)
    merged_all.to_csv(merged_path, index=False)
    summary.to_csv(summary_path, index=False)
    transition.to_csv(transition_path, index=False)
    scenario_summary.to_csv(scenario_summary_path, index=False)
    matrix_distribution.to_csv(matrix_distribution_path, index=False)
    method_path.write_text(json.dumps(method, indent=2), encoding="utf-8")

    print(f"Affected provider-order cases: {affected:,}")
    print(f"All prescribing order bundles: {total_order_bundles:,}")
    print(f"Snapshot severity threshold: {severity_threshold:.2f}")
    print(f"Snapshot evidence threshold: {evidence_threshold:.2f}")
    print("[DESIGN] Snapshot comparator reads Stage-02 signals only.")
    print("[DESIGN] No longitudinal, renal-trajectory, transplant-protocol, repeat, or management context is used.")
    print("[DESIGN] Capacity availability is matched; snapshot ranking uses severity/evidence only.")

    print("\nThree-policy operational comparison:")
    fair = summary[
        summary["policy_key"].isin(
            {"always_interrupt", "snapshot_capacity_matched", "clarity"}
        )
    ]
    print(
        fair[
            [
                "Policy",
                "scenario",
                "direct_physician_alerts",
                "enhanced_pharmacist_reviews",
                "passive_notices",
                "suppressed_and_audited",
                "direct_interruption_reduction_vs_always_interrupt_percent",
                "protected_safety_violations",
            ]
        ].to_string(index=False)
    )

    print("\nSnapshot versus CLARITY by matched scenario:")
    print(scenario_summary.to_string(index=False))
    print("\nInterpretation:")
    print(" - Always interrupt measures the burden of alerting every affected order.")
    print(" - Snapshot matrix uses current-rule severity and evidence support only.")
    print(" - Full CLARITY adds longitudinal/transplant context and robust routing.")
    print(" - Capacity availability is matched so operational resources do not confound the comparison.")
    print(" - Clinical superiority of reclassified cases requires expert adjudication.")

    print(f"\nIdentifier-free summary: {summary_path}")
    print(f"Identifier-free transition matrix: {transition_path}")
    print(f"Identifier-free scenario summary: {scenario_summary_path}")
    print(f"Method record: {method_path}")
    print("\nSTAGE 10B: COMPLETE - PASS")


if __name__ == "__main__":
    main()
