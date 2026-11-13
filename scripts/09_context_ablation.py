from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from common import atomic_csv, atomic_json, banner, parser, progress_printer, sha256
from clarity_v3.bundles import build_order_bundles
from clarity_v3.component_ablation import (
    ABLATIONS,
    AblationSpec,
    build_component_variant,
    build_reusable_active_snapshots,
)
from clarity_v3.capacity import apply_dual_capacity
from clarity_v3.config import (
    load_capacity_scenarios,
    load_state_config,
    load_uncertainty_config,
)
from clarity_v3.project import ensure_output_dirs, load_project_paths
from clarity_v3.reference import ReferenceStore
from clarity_v3.uncertainty import MultiEnvelopeActionPolicy, sample_design


def _bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y", "t"})
    )


def _string_changed(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.astype("string").fillna("<NA>").ne(
        right.astype("string").fillna("<NA>")
    )


def _bool_changed(left: pd.Series, right: pd.Series) -> pd.Series:
    return _bool(left).ne(_bool(right))


def _numeric_changed(
    left: pd.Series,
    right: pd.Series,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> pd.Series:
    lhs = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    rhs = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    equal = np.isclose(lhs, rhs, atol=atol, rtol=rtol, equal_nan=True)
    return pd.Series(~equal, index=left.index)


def _transition_table(
    frame: pd.DataFrame,
    primary_column: str,
    ablated_column: str,
    ablation: str,
    level: str,
) -> pd.DataFrame:
    result = (
        frame.groupby([primary_column, ablated_column], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .rename(
            columns={
                primary_column: "full_context_action",
                ablated_column: "ablated_action",
            }
        )
    )
    result.insert(0, "level", level)
    result.insert(0, "ablation", ablation)
    result["changed"] = result["full_context_action"].ne(result["ablated_action"])
    return result


def _eligible_mask(spec: AblationSpec, primary_states: pd.DataFrame) -> pd.Series:
    if spec.remove_episode_memory:
        return primary_states["repeat_status"].astype(str).ne("new_episode")
    if spec.remove_acute_renal_context:
        return _bool(primary_states["acute_renal_deterioration"])
    if spec.remove_transplant_protocol_context:
        return ~primary_states["protocol_fit"].astype(str).isin(
            {"no_matching_transplant_protocol", "removed_by_ablation"}
        )
    if spec.remove_transition_context:
        return ~primary_states["transition_status"].astype(str).isin(
            {"none", "removed_by_ablation"}
        )
    return pd.Series(False, index=primary_states.index)


def _assert_variant_safety(
    states: pd.DataFrame,
    decisions: pd.DataFrame,
    bundles: pd.DataFrame,
    final: pd.DataFrame,
) -> dict[str, bool]:
    checks: dict[str, bool] = {
        "signal_state_preservation": len(states) == len(decisions),
        "state_ids_unique": not states["signal_id"].duplicated().any(),
        "decision_ids_unique": not decisions["signal_id"].duplicated().any(),
        "bundle_ids_unique": not bundles["bundle_id"].duplicated().any(),
        "one_final_row_per_bundle": not final["bundle_id"].duplicated().any(),
        "final_bundle_preservation": len(final) == len(bundles),
    }
    protected_alert_bundles = set(
        decisions.loc[_bool(decisions["protected_alert"]), "bundle_id"].astype(str)
    )
    protected_review_bundles = set(
        decisions.loc[_bool(decisions["protected_review"]), "bundle_id"].astype(str)
    )
    checks["protected_alerts_remain_alert"] = final.loc[
        final["bundle_id"].astype(str).isin(protected_alert_bundles), "final_action"
    ].eq("alert").all()
    checks["protected_reviews_remain_resource_actions"] = final.loc[
        final["bundle_id"].astype(str).isin(protected_review_bundles), "final_action"
    ].isin({"alert", "escalate"}).all()
    suppressed = decisions[decisions["pre_capacity_action"].eq("suppress")]
    checks["suppression_only_standard_safety"] = suppressed["safety_class"].eq(
        "standard"
    ).all()
    checks["suppression_not_material_change"] = ~_bool(
        suppressed["material_change_since_prior"]
    ).any()
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise AssertionError(
            "Ablation safety validation failed: " + ", ".join(failures)
        )
    return checks


def _compare_state_metrics(
    spec: AblationSpec,
    primary_states: pd.DataFrame,
    ablated_states: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "signal_id",
        "bundle_id",
        "management_status",
        "contextual_harm_index",
        "material_change_since_prior",
        "safety_class",
        "admissible_actions",
        "protocol_fit",
        "monitoring_status",
        "transition_status",
        "repeat_status",
        "acute_renal_deterioration",
    ]
    primary = primary_states[columns].copy()
    ablated = ablated_states[columns].copy()
    compared = primary.merge(
        ablated,
        on=["signal_id", "bundle_id"],
        suffixes=("_full", "_ablated"),
        validate="one_to_one",
    )
    eligible_by_signal = dict(
        zip(
            primary_states["signal_id"].astype(str),
            _eligible_mask(spec, primary_states).astype(bool),
        )
    )
    eligible = compared["signal_id"].astype(str).map(eligible_by_signal).fillna(False)
    comparisons: dict[str, pd.Series] = {
        "management_status": _string_changed(
            compared["management_status_full"], compared["management_status_ablated"]
        ),
        "contextual_harm_index": _numeric_changed(
            compared["contextual_harm_index_full"],
            compared["contextual_harm_index_ablated"],
        ),
        "material_change_since_prior": _bool_changed(
            compared["material_change_since_prior_full"],
            compared["material_change_since_prior_ablated"],
        ),
        "safety_class": _string_changed(
            compared["safety_class_full"], compared["safety_class_ablated"]
        ),
        "admissible_actions": _string_changed(
            compared["admissible_actions_full"],
            compared["admissible_actions_ablated"],
        ),
        "protocol_fit": _string_changed(
            compared["protocol_fit_full"], compared["protocol_fit_ablated"]
        ),
        "monitoring_status": _string_changed(
            compared["monitoring_status_full"],
            compared["monitoring_status_ablated"],
        ),
        "transition_status": _string_changed(
            compared["transition_status_full"],
            compared["transition_status_ablated"],
        ),
        "repeat_status": _string_changed(
            compared["repeat_status_full"], compared["repeat_status_ablated"]
        ),
        "acute_renal_deterioration": _bool_changed(
            compared["acute_renal_deterioration_full"],
            compared["acute_renal_deterioration_ablated"],
        ),
    }
    rows: list[dict] = []
    any_change = pd.Series(False, index=compared.index)
    for metric, changed in comparisons.items():
        any_change |= changed
        rows.append(
            {
                "ablation": spec.name,
                "component": spec.label,
                "metric": metric,
                "signals": int(len(compared)),
                "changed_count": int(changed.sum()),
                "changed_percent": 100.0 * float(changed.mean()),
                "eligible_signal_count": int(eligible.sum()),
                "changed_among_eligible_count": int((changed & eligible).sum()),
                "changed_among_eligible_percent": (
                    100.0 * float((changed & eligible).sum()) / int(eligible.sum())
                    if int(eligible.sum())
                    else 0.0
                ),
            }
        )
    compared["ablation"] = spec.name
    compared["component_eligible"] = eligible
    compared["any_state_metric_changed"] = any_change
    return pd.DataFrame(rows), compared


def main() -> None:
    args_parser = parser(
        "Run rebuild-based component contribution through medium-capacity final routing"
    )
    args_parser.add_argument(
        "--scenario",
        default="medium_capacity",
        help="Capacity scenario for final-action contribution analysis",
    )
    args = args_parser.parse_args()

    banner("STAGE 09R - REBUILD-BASED COMPONENT CONTRIBUTION ANALYSIS")
    paths = load_project_paths(args.paths)
    ensure_output_dirs(paths)
    required_files = [
        paths.orders_file,
        paths.signals_file,
        paths.states_file,
        paths.decisions_file,
        paths.bundles_file,
        paths.final_bundles_file,
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run Stages 01-07 before Stage 09R. Missing: " + ", ".join(missing)
        )
    if paths.note_context_mode != "disabled":
        raise ValueError("MIMIC Phase-1 component analysis requires note context disabled")

    orders = pd.read_csv(paths.orders_file, low_memory=False)
    signals = pd.read_csv(paths.signals_file, low_memory=False)
    primary_states = pd.read_csv(paths.states_file, low_memory=False)
    primary_decisions = pd.read_csv(paths.decisions_file, low_memory=False)
    primary_bundles = pd.read_csv(paths.bundles_file, low_memory=False)
    primary_final_all = pd.read_csv(paths.final_bundles_file, low_memory=False)
    primary_final = primary_final_all[
        primary_final_all["scenario"].astype(str).eq(args.scenario)
    ].copy()
    if primary_final.empty:
        raise ValueError(f"Primary final decisions do not contain scenario={args.scenario!r}")

    reference = ReferenceStore(paths.reference_dir)
    state_config = load_state_config(paths.root / "config" / "state_builder.json")
    envelopes, action_config = load_uncertainty_config(
        paths.root / "config" / "uncertainty_design.json"
    )
    design = sample_design(envelopes)
    policy = MultiEnvelopeActionPolicy(
        design,
        {envelope.name: envelope.regret_quantile for envelope in envelopes},
        action_config,
    )
    capacity_scenarios = load_capacity_scenarios(
        paths.root / "config" / "capacity_scenarios.json"
    )
    matching_scenarios = [item for item in capacity_scenarios if item.name == args.scenario]
    if len(matching_scenarios) != 1:
        raise ValueError(f"Expected exactly one capacity scenario named {args.scenario!r}")
    capacity_scenario = matching_scenarios[0]
    total_order_bundles = int(orders["bundle_id"].astype(str).nunique())

    print("[DESIGN] Each ablation rebuilds Stage 03 before rerunning Stages 04-06.")
    print("[DESIGN] Current creatinine is retained when acute renal trajectory is removed.")
    print("[DESIGN] Transplant status and phase are retained when protocol interpretation is removed.")
    print("[DESIGN] Results are algorithmic component contributions, not causal clinical effects.")
    print(f"Signals: {len(signals):,}")
    print(f"Affected bundles: {len(primary_bundles):,}")
    print(f"Final-routing scenario: {args.scenario}")

    required_order_ids = set(signals["order_id"].astype(str))
    snapshots = build_reusable_active_snapshots(
        orders,
        signals,
        progress_callback=progress_printer("Reusable active-medication state"),
    )

    evidence_dir = paths.evidence_dir / "component_contribution"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    private_dir = paths.private_dir / "component_contribution"
    private_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    state_metric_frames: list[pd.DataFrame] = []
    signal_transition_frames: list[pd.DataFrame] = []
    bundle_transition_frames: list[pd.DataFrame] = []
    final_transition_frames: list[pd.DataFrame] = []
    signal_counterfactuals = primary_decisions[
        [
            "signal_id",
            "bundle_id",
            "safety_class",
            "calibration_sensitive",
            "pre_capacity_action",
        ]
    ].copy().rename(
        columns={
            "safety_class": "full_context_safety_class",
            "calibration_sensitive": "full_context_calibration_sensitive",
            "pre_capacity_action": "full_context_pre_capacity_action",
        }
    )
    bundle_counterfactuals = primary_bundles[
        ["bundle_id", "pre_capacity_action", "safety_class"]
    ].copy().rename(
        columns={
            "pre_capacity_action": "full_context_bundle_action",
            "safety_class": "full_context_bundle_safety_class",
        }
    )
    bundle_counterfactuals = bundle_counterfactuals.merge(
        primary_final[["bundle_id", "final_action"]].rename(
            columns={"final_action": "full_context_final_action"}
        ),
        on="bundle_id",
        validate="one_to_one",
    )

    primary_final_counts = primary_final["final_action"].value_counts()
    for spec in ABLATIONS:
        print("\n" + "-" * 78)
        print(f"Rebuilding: {spec.name}")
        print(spec.label)
        ablated_states = build_component_variant(
            spec,
            signals,
            orders,
            reference,
            state_config,
            active_snapshots=snapshots,
            progress_callback_signals=progress_printer(
                f"{spec.name} state rebuild"
            ),
        )
        ablated_decisions = policy.decide_dataframe(ablated_states)
        ablated_bundles = build_order_bundles(ablated_decisions)
        ablated_final = apply_dual_capacity(
            ablated_bundles,
            capacity_scenario,
            total_order_bundles=total_order_bundles,
        )
        if set(ablated_bundles["bundle_id"].astype(str)) != set(
            primary_bundles["bundle_id"].astype(str)
        ):
            raise AssertionError(
                f"{spec.name} changed the affected bundle denominator"
            )
        checks = _assert_variant_safety(
            ablated_states,
            ablated_decisions,
            ablated_bundles,
            ablated_final,
        )

        state_metrics, state_compared = _compare_state_metrics(
            spec, primary_states, ablated_states
        )
        state_metric_frames.append(state_metrics)

        signal_compared = primary_decisions[
            ["signal_id", "pre_capacity_action", "calibration_sensitive", "consensus_action"]
        ].merge(
            ablated_decisions[
                ["signal_id", "pre_capacity_action", "calibration_sensitive", "consensus_action"]
            ],
            on="signal_id",
            suffixes=("_full", "_ablated"),
            validate="one_to_one",
        )
        signal_action_changed = signal_compared["pre_capacity_action_full"].ne(
            signal_compared["pre_capacity_action_ablated"]
        )
        calibration_changed = _bool_changed(
            signal_compared["calibration_sensitive_full"],
            signal_compared["calibration_sensitive_ablated"],
        )
        consensus_changed = _string_changed(
            signal_compared["consensus_action_full"],
            signal_compared["consensus_action_ablated"],
        )
        signal_transition_frames.append(
            _transition_table(
                signal_compared,
                "pre_capacity_action_full",
                "pre_capacity_action_ablated",
                spec.name,
                "signal_pre_capacity",
            )
        )

        bundle_compared = primary_bundles[
            ["bundle_id", "pre_capacity_action", "calibration_sensitive", "safety_class"]
        ].merge(
            ablated_bundles[
                ["bundle_id", "pre_capacity_action", "calibration_sensitive", "safety_class"]
            ],
            on="bundle_id",
            suffixes=("_full", "_ablated"),
            validate="one_to_one",
        )
        bundle_action_changed = bundle_compared["pre_capacity_action_full"].ne(
            bundle_compared["pre_capacity_action_ablated"]
        )
        bundle_transition_frames.append(
            _transition_table(
                bundle_compared,
                "pre_capacity_action_full",
                "pre_capacity_action_ablated",
                spec.name,
                "bundle_pre_capacity",
            )
        )

        final_compared = primary_final[["bundle_id", "final_action"]].merge(
            ablated_final[["bundle_id", "final_action"]],
            on="bundle_id",
            suffixes=("_full", "_ablated"),
            validate="one_to_one",
        )
        final_action_changed = final_compared["final_action_full"].ne(
            final_compared["final_action_ablated"]
        )
        final_transition_frames.append(
            _transition_table(
                final_compared,
                "final_action_full",
                "final_action_ablated",
                spec.name,
                "medium_capacity_final",
            )
        )

        eligible = _eligible_mask(spec, primary_states)
        eligible_ids = set(primary_states.loc[eligible, "signal_id"].astype(str))
        eligible_changed_signal_actions = int(
            signal_compared.loc[
                signal_compared["signal_id"].astype(str).isin(eligible_ids),
                "pre_capacity_action_full",
            ].reset_index(drop=True).ne(
                signal_compared.loc[
                    signal_compared["signal_id"].astype(str).isin(eligible_ids),
                    "pre_capacity_action_ablated",
                ].reset_index(drop=True)
            ).sum()
        )
        ablated_final_counts = ablated_final["final_action"].value_counts()
        summary_rows.append(
            {
                "ablation": spec.name,
                "component": spec.label,
                "signals": int(len(ablated_states)),
                "eligible_signals": int(eligible.sum()),
                "affected_bundles": int(len(ablated_bundles)),
                "any_state_metric_changed": int(
                    state_compared["any_state_metric_changed"].sum()
                ),
                "safety_class_changed": int(
                    _string_changed(
                        state_compared["safety_class_full"],
                        state_compared["safety_class_ablated"],
                    ).sum()
                ),
                "admissible_actions_changed": int(
                    _string_changed(
                        state_compared["admissible_actions_full"],
                        state_compared["admissible_actions_ablated"],
                    ).sum()
                ),
                "calibration_sensitive_changed": int(calibration_changed.sum()),
                "consensus_action_changed": int(consensus_changed.sum()),
                "signal_pre_capacity_action_changed": int(signal_action_changed.sum()),
                "eligible_signal_action_changed": eligible_changed_signal_actions,
                "bundle_pre_capacity_action_changed": int(bundle_action_changed.sum()),
                "medium_capacity_final_action_changed": int(final_action_changed.sum()),
                "primary_suppress": int(primary_final_counts.get("suppress", 0)),
                "ablated_suppress": int(ablated_final_counts.get("suppress", 0)),
                "delta_suppress": int(
                    ablated_final_counts.get("suppress", 0)
                    - primary_final_counts.get("suppress", 0)
                ),
                "primary_passive": int(primary_final_counts.get("passive", 0)),
                "ablated_passive": int(ablated_final_counts.get("passive", 0)),
                "delta_passive": int(
                    ablated_final_counts.get("passive", 0)
                    - primary_final_counts.get("passive", 0)
                ),
                "primary_alert": int(primary_final_counts.get("alert", 0)),
                "ablated_alert": int(ablated_final_counts.get("alert", 0)),
                "delta_direct_physician_alert": int(
                    ablated_final_counts.get("alert", 0)
                    - primary_final_counts.get("alert", 0)
                ),
                "primary_escalate": int(primary_final_counts.get("escalate", 0)),
                "ablated_escalate": int(ablated_final_counts.get("escalate", 0)),
                "delta_enhanced_pharmacist_review": int(
                    ablated_final_counts.get("escalate", 0)
                    - primary_final_counts.get("escalate", 0)
                ),
                "all_safety_checks_pass": bool(all(checks.values())),
            }
        )

        signal_counterfactuals = signal_counterfactuals.merge(
            ablated_decisions[
                ["signal_id", "safety_class", "calibration_sensitive", "pre_capacity_action"]
            ].rename(
                columns={
                    "safety_class": f"{spec.name}_safety_class",
                    "calibration_sensitive": f"{spec.name}_calibration_sensitive",
                    "pre_capacity_action": f"{spec.name}_pre_capacity_action",
                }
            ),
            on="signal_id",
            validate="one_to_one",
        )
        bundle_counterfactuals = bundle_counterfactuals.merge(
            ablated_bundles[
                ["bundle_id", "pre_capacity_action", "safety_class"]
            ].rename(
                columns={
                    "pre_capacity_action": f"{spec.name}_bundle_action",
                    "safety_class": f"{spec.name}_bundle_safety_class",
                }
            ),
            on="bundle_id",
            validate="one_to_one",
        ).merge(
            ablated_final[["bundle_id", "final_action"]].rename(
                columns={"final_action": f"{spec.name}_final_action"}
            ),
            on="bundle_id",
            validate="one_to_one",
        )

        print(
            f"{spec.name}: state changes={int(state_compared['any_state_metric_changed'].sum()):,}; "
            f"signal actions changed={int(signal_action_changed.sum()):,}; "
            f"bundle actions changed={int(bundle_action_changed.sum()):,}; "
            f"medium-capacity final actions changed={int(final_action_changed.sum()):,}"
        )

    summary = pd.DataFrame(summary_rows)
    state_metrics = pd.concat(state_metric_frames, ignore_index=True)
    signal_transitions = pd.concat(signal_transition_frames, ignore_index=True)
    bundle_transitions = pd.concat(bundle_transition_frames, ignore_index=True)
    final_transitions = pd.concat(final_transition_frames, ignore_index=True)

    atomic_csv(summary, evidence_dir / "09r_component_contribution_summary.csv")
    atomic_csv(state_metrics, evidence_dir / "09r_state_metric_changes.csv")
    atomic_csv(signal_transitions, evidence_dir / "09r_signal_action_transitions.csv")
    atomic_csv(bundle_transitions, evidence_dir / "09r_bundle_action_transitions.csv")
    atomic_csv(final_transitions, evidence_dir / "09r_final_action_transitions.csv")
    atomic_csv(
        signal_counterfactuals,
        private_dir / "09r_signal_counterfactuals.csv",
    )
    atomic_csv(
        bundle_counterfactuals,
        private_dir / "09r_bundle_counterfactuals.csv",
    )

    manifest = {
        "stage": "09R",
        "method": "rebuild_based_component_contribution",
        "interpretation": (
            "Algorithmic component contribution under the frozen Phase-1 policy; "
            "not a causal estimate of clinical benefit."
        ),
        "capacity_scenario": args.scenario,
        "signals": int(len(signals)),
        "affected_bundles": int(len(primary_bundles)),
        "all_order_bundles": total_order_bundles,
        "uncertainty_design_points": int(len(design)),
        "ablations": [spec.__dict__ for spec in ABLATIONS],
        "source_hashes": {
            "orders": sha256(paths.orders_file),
            "signals": sha256(paths.signals_file),
            "primary_states": sha256(paths.states_file),
            "primary_decisions": sha256(paths.decisions_file),
            "primary_bundles": sha256(paths.bundles_file),
            "primary_final": sha256(paths.final_bundles_file),
        },
        "evidence_files": [
            "09r_component_contribution_summary.csv",
            "09r_state_metric_changes.csv",
            "09r_signal_action_transitions.csv",
            "09r_bundle_action_transitions.csv",
            "09r_final_action_transitions.csv",
        ],
        "private_counterfactual_files": [
            "09r_signal_counterfactuals.csv",
            "09r_bundle_counterfactuals.csv",
        ],
    }
    atomic_json(manifest, evidence_dir / "09r_component_contribution_manifest.json")

    print("\nComponent contribution summary:")
    display_columns = [
        "ablation",
        "eligible_signals",
        "any_state_metric_changed",
        "signal_pre_capacity_action_changed",
        "bundle_pre_capacity_action_changed",
        "medium_capacity_final_action_changed",
        "delta_direct_physician_alert",
        "delta_enhanced_pharmacist_review",
        "all_safety_checks_pass",
    ]
    print(summary[display_columns].to_string(index=False))
    print(f"\nIdentifier-free evidence: {evidence_dir}")
    print(f"Private counterfactual records: {private_dir}")
    print("\nSTAGE 09R: COMPLETE - PASS")


if __name__ == "__main__":
    main()
