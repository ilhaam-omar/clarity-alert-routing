from __future__ import annotations

from math import floor
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .models import CapacityScenario, FinalBundleDecision, parse_actions


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _rank_map(frame: pd.DataFrame, value_column: str) -> Dict[str, int]:
    if frame.empty:
        return {}
    ranked = frame.assign(
        _rank_value=pd.to_numeric(frame[value_column], errors="coerce").fillna(float("-inf")),
        _bundle_id=frame["bundle_id"].astype(str),
    ).sort_values(
        ["_rank_value", "_bundle_id"],
        ascending=[False, True],
        kind="stable",
    )
    return {
        bundle_id: index + 1
        for index, bundle_id in enumerate(ranked["_bundle_id"].tolist())
    }


def _fallback_action(row: pd.Series) -> str:
    admissible = set(parse_actions(row["admissible_actions"]))
    fallback = str(row.get("fallback_non_resource_action", "passive"))
    if fallback in admissible and fallback in {"suppress", "passive"}:
        return fallback
    if "passive" in admissible:
        return "passive"
    if "suppress" in admissible:
        return "suppress"
    # Protected review cases do not have a non-resource option and are handled
    # before this helper is called.
    raise ValueError(
        f"Bundle {row.get('bundle_id')} has no admissible non-resource fallback"
    )


def _scenario_slots(
    scenario: CapacityScenario,
    *,
    total_order_bundles: int,
    pre_capacity_alert_demand: int,
    pre_capacity_expert_demand: int,
) -> tuple[int, int]:
    if scenario.unconstrained:
        return pre_capacity_alert_demand, pre_capacity_expert_demand
    frontline = int(
        floor(
            total_order_bundles
            * float(scenario.frontline_alerts_per_100_order_bundles or 0.0)
            / 100.0
        )
    )
    expert = int(
        floor(
            pre_capacity_expert_demand
            * float(scenario.expert_review_coverage_fraction or 0.0)
        )
    )
    return frontline, expert


def apply_dual_capacity(
    bundles: pd.DataFrame,
    scenario: CapacityScenario,
    total_order_bundles: Optional[int] = None,
) -> pd.DataFrame:
    """Apply normalized frontline and expert capacity to bundle decisions.

    Phase-1 MIMIC capacity is a cohort-normalized stress test, not a replay of
    real hospital shifts. The allocator never pools shifted timestamps across
    patients. It rations only actions that were already selected before
    capacity:

    * protected immediate alerts always remain alerts;
    * pre-capacity escalations compete for expert-review capacity by
      ``delta_expert``;
    * an unserved protected review spills to an immediate alert;
    * discretionary pre-capacity alerts compete for remaining nominal
      frontline capacity by ``delta_frontline``;
    * passive and suppress actions remain non-resource actions.

    The algorithm is deterministic, O(n log n), and preserves the theoretical
    escalation rule: when expert capacity binds, cases with the largest positive
    benefit of escalation are served first.
    """
    scenario.validate()
    if bundles.empty:
        return pd.DataFrame()
    required = {
        "bundle_id",
        "pre_capacity_action",
        "fallback_non_resource_action",
        "admissible_actions",
        "protected_alert",
        "protected_review",
        "delta_frontline",
        "delta_expert",
    }
    missing = required - set(bundles.columns)
    if missing:
        raise ValueError(f"Bundle table missing columns: {sorted(missing)}")
    working = bundles.copy()
    working["bundle_id"] = working["bundle_id"].astype(str)
    if working["bundle_id"].duplicated().any():
        raise ValueError("Bundle table contains duplicate bundle_id values")
    working["protected_alert_bool"] = working["protected_alert"].map(_as_bool)
    working["protected_review_bool"] = working["protected_review"].map(_as_bool)
    working["pre_capacity_action"] = working["pre_capacity_action"].astype(str)
    working["delta_frontline"] = pd.to_numeric(
        working["delta_frontline"], errors="coerce"
    ).fillna(0.0)
    working["delta_expert"] = pd.to_numeric(
        working["delta_expert"], errors="coerce"
    ).fillna(0.0)

    if total_order_bundles is None:
        total_order_bundles = int(working["bundle_id"].nunique())
    total_order_bundles = int(total_order_bundles)
    if total_order_bundles <= 0:
        raise ValueError("total_order_bundles must be positive")

    alert_demand_mask = working["pre_capacity_action"].eq("alert")
    # Expert demand is defined by the pre-capacity escalation recommendation.
    # A bundle that is already an alert but also contains a protected-review
    # signal must not be silently converted to escalation in the unconstrained
    # reference. It remains an immediate protected resource action.
    expert_demand_mask = (
        working["pre_capacity_action"].eq("escalate")
        & ~working["protected_alert_bool"]
    )
    protected_review_alert_mask = (
        alert_demand_mask
        & working["protected_review_bool"]
        & ~working["protected_alert_bool"]
    )
    protected_immediate_mask = (
        working["protected_alert_bool"] | protected_review_alert_mask
    )
    pre_capacity_alert_demand = int(alert_demand_mask.sum())
    pre_capacity_expert_demand = int(expert_demand_mask.sum())
    frontline_slots, expert_slots = _scenario_slots(
        scenario,
        total_order_bundles=total_order_bundles,
        pre_capacity_alert_demand=pre_capacity_alert_demand,
        pre_capacity_expert_demand=pre_capacity_expert_demand,
    )

    expert_candidates = working[expert_demand_mask].copy()
    alert_candidates = working[alert_demand_mask].copy()
    expert_rank = _rank_map(expert_candidates, "delta_expert")
    frontline_rank = _rank_map(alert_candidates, "delta_frontline")

    assigned: Dict[str, str] = {}
    status: Dict[str, str] = {}
    reason: Dict[str, str] = {}

    # 1. Non-negotiable immediate alerts establish the safety floor.
    # This includes both must-interrupt bundles and bundles whose consolidated
    # pre-capacity action is alert while another component requires at least
    # alert-or-review. Neither may be downgraded to a non-resource action.
    protected_alerts = working[protected_immediate_mask].sort_values(
        ["delta_frontline", "bundle_id"],
        ascending=[False, True],
        kind="stable",
    )
    immediate_alerts_used = 0
    for _, row in protected_alerts.iterrows():
        bundle_id = str(row["bundle_id"])
        immediate_alerts_used += 1
        assigned[bundle_id] = "alert"
        within_nominal = immediate_alerts_used <= frontline_slots or scenario.unconstrained
        if _as_bool(row["protected_alert_bool"]):
            prefix = "protected_immediate_alert"
            reason_text = (
                "must_interrupt safety constraint; immediate alert cannot be rationed"
            )
        else:
            prefix = "protected_review_alert"
            reason_text = (
                "bundle pre-capacity action is alert and contains a protected review signal; "
                "it cannot fall back to passive or suppression"
            )
        status[bundle_id] = (
            f"{prefix}_within_nominal"
            if within_nominal
            else f"{prefix}_overrun"
        )
        reason[bundle_id] = reason_text

    # 2. Allocate expert capacity only among cases already requiring escalation.
    expert_selected_ids: set[str] = set()
    if scenario.unconstrained:
        expert_selected_ids = set(expert_candidates["bundle_id"].astype(str))
    elif expert_slots > 0 and not expert_candidates.empty:
        expert_selected_ids = set(
            expert_candidates.assign(
                _rank=expert_candidates["bundle_id"].astype(str).map(expert_rank)
            )
            .nsmallest(expert_slots, "_rank")["bundle_id"]
            .astype(str)
        )

    for _, row in expert_candidates.sort_values(
        ["delta_expert", "bundle_id"],
        ascending=[False, True],
        kind="stable",
    ).iterrows():
        bundle_id = str(row["bundle_id"])
        if bundle_id in assigned:
            continue
        if bundle_id in expert_selected_ids:
            assigned[bundle_id] = "escalate"
            status[bundle_id] = "expert_capacity_selected"
            reason[bundle_id] = (
                "selected for asynchronous expert review by descending escalation benefit"
            )
            continue
        if _as_bool(row["protected_review_bool"]):
            if not scenario.critical_reviews_spill_to_frontline:
                raise ValueError(
                    "A protected review could not be served and spillover is disabled"
                )
            assigned[bundle_id] = "alert"
            immediate_alerts_used += 1
            within_nominal = (
                immediate_alerts_used <= frontline_slots or scenario.unconstrained
            )
            status[bundle_id] = (
                "protected_review_spillover_within_nominal"
                if within_nominal
                else "protected_review_spillover_overrun"
            )
            reason[bundle_id] = (
                "expert capacity unavailable; protected high-severity uncertainty spills to frontline alert"
            )
        else:
            fallback = _fallback_action(row)
            assigned[bundle_id] = fallback
            status[bundle_id] = "expert_capacity_not_selected_fallback"
            reason[bundle_id] = (
                "expert capacity unavailable; bundle receives its best admissible non-resource action"
            )

    # 3. Allocate remaining nominal frontline capacity to discretionary alerts.
    discretionary_alerts = working[
        alert_demand_mask
        & ~protected_immediate_mask
        & ~working["bundle_id"].isin(assigned)
    ].copy()
    available_frontline = (
        len(discretionary_alerts)
        if scenario.unconstrained
        else max(0, frontline_slots - immediate_alerts_used)
    )
    selected_discretionary: set[str] = set()
    if available_frontline > 0 and not discretionary_alerts.empty:
        selected_discretionary = set(
            discretionary_alerts.sort_values(
                ["delta_frontline", "bundle_id"],
                ascending=[False, True],
                kind="stable",
            )
            .head(available_frontline)["bundle_id"]
            .astype(str)
        )
    for _, row in discretionary_alerts.iterrows():
        bundle_id = str(row["bundle_id"])
        if bundle_id in selected_discretionary:
            assigned[bundle_id] = "alert"
            immediate_alerts_used += 1
            status[bundle_id] = "frontline_capacity_selected"
            reason[bundle_id] = (
                "discretionary immediate alert selected by descending frontline benefit"
            )
        else:
            fallback = _fallback_action(row)
            assigned[bundle_id] = fallback
            status[bundle_id] = "frontline_capacity_not_selected_fallback"
            reason[bundle_id] = (
                "nominal frontline attention threshold reached; bundle receives non-resource fallback"
            )

    # 4. Preserve all pre-capacity non-resource actions.
    for _, row in working.iterrows():
        bundle_id = str(row["bundle_id"])
        if bundle_id in assigned:
            continue
        action = str(row["pre_capacity_action"])
        if action not in {"suppress", "passive"}:
            # This should only occur for malformed inputs not covered above.
            fallback = _fallback_action(row)
            assigned[bundle_id] = fallback
            status[bundle_id] = "unhandled_resource_action_fallback"
            reason[bundle_id] = "resource action was not eligible under the normalized allocator"
        else:
            assigned[bundle_id] = action
            status[bundle_id] = "non_resource_pre_capacity_action"
            reason[bundle_id] = "pre-capacity non-resource action retained"

    if scenario.unconstrained:
        mismatches = [
            bundle_id
            for bundle_id, expected in zip(
                working["bundle_id"].astype(str),
                working["pre_capacity_action"].astype(str),
            )
            if assigned.get(bundle_id) != expected
        ]
        if mismatches:
            raise AssertionError(
                "Unconstrained capacity must preserve every pre-capacity action; "
                f"observed {len(mismatches)} mismatch(es)"
            )

    records: list[dict] = []
    for _, row in working.iterrows():
        bundle_id = str(row["bundle_id"])
        record = FinalBundleDecision(
            bundle_id=bundle_id,
            scenario=scenario.name,
            allocation_scope=scenario.allocation_scope,
            epoch_start="not_applicable",
            pre_capacity_action=str(row["pre_capacity_action"]),
            final_action=assigned[bundle_id],
            capacity_status=status[bundle_id],
            frontline_rank=frontline_rank.get(bundle_id),
            expert_rank=expert_rank.get(bundle_id),
            frontline_slots=frontline_slots,
            expert_slots=expert_slots,
            total_order_bundles=total_order_bundles,
            pre_capacity_alert_demand=pre_capacity_alert_demand,
            pre_capacity_expert_demand=pre_capacity_expert_demand,
            frontline_alerts_per_100_order_bundles=(
                None
                if scenario.unconstrained
                else scenario.frontline_alerts_per_100_order_bundles
            ),
            expert_review_coverage_fraction=(
                1.0
                if scenario.unconstrained
                else scenario.expert_review_coverage_fraction
            ),
            protected_alert=_as_bool(row["protected_alert_bool"]),
            protected_review=_as_bool(row["protected_review_bool"]),
            capacity_reason=reason[bundle_id],
        )
        records.append(record.as_dict())
    result = pd.DataFrame(records)
    if len(result) != len(working):
        raise AssertionError("Capacity allocation did not preserve the bundle denominator")
    return result


def run_capacity_scenarios(
    bundles: pd.DataFrame,
    scenarios: Iterable[CapacityScenario],
    total_order_bundles: int,
) -> pd.DataFrame:
    frames = [
        apply_dual_capacity(
            bundles,
            scenario,
            total_order_bundles=total_order_bundles,
        )
        for scenario in scenarios
    ]
    if not frames:
        return pd.DataFrame()
    records: list[dict] = []
    for frame in frames:
        records.extend(frame.to_dict("records"))
    return pd.DataFrame.from_records(records)


def capacity_summary(final: pd.DataFrame) -> pd.DataFrame:
    if final.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for scenario, group in final.groupby("scenario", sort=False):
        counts = group["final_action"].value_counts()
        total_order_bundles = int(group["total_order_bundles"].iloc[0])
        affected_bundles = int(group["bundle_id"].nunique())
        alert_count = int(counts.get("alert", 0))
        expert_count = int(counts.get("escalate", 0))
        expert_demand = int(group["pre_capacity_expert_demand"].iloc[0])
        protected_alert_bool = group["protected_alert"].map(_as_bool)
        protected_review_bool = group["protected_review"].map(_as_bool)
        protected_review_alert_bool = (
            protected_review_bool
            & ~protected_alert_bool
            & group["pre_capacity_action"].astype(str).eq("alert")
        )
        protected_must_interrupt_floor = int(protected_alert_bool.sum())
        protected_review_alert_floor = int(protected_review_alert_bool.sum())
        protected_floor = (
            protected_must_interrupt_floor + protected_review_alert_floor
        )
        spillover = int(
            group["capacity_status"].astype(str).str.startswith(
                "protected_review_spillover", na=False
            ).sum()
        )
        frontline_overrun = int(
            group["capacity_status"].isin(
                {
                    "protected_immediate_alert_overrun",
                    "protected_review_alert_overrun",
                    "protected_review_spillover_overrun",
                }
            ).sum()
        )
        rows.append(
            {
                "scenario": scenario,
                "allocation_scope": str(group["allocation_scope"].iloc[0]),
                "total_order_bundles": total_order_bundles,
                "affected_bundles": affected_bundles,
                "nominal_frontline_slots": int(group["frontline_slots"].iloc[0]),
                "nominal_expert_slots": int(group["expert_slots"].iloc[0]),
                "pre_capacity_alert_demand": int(
                    group["pre_capacity_alert_demand"].iloc[0]
                ),
                "pre_capacity_expert_demand": expert_demand,
                "protected_must_interrupt_floor": protected_must_interrupt_floor,
                "protected_review_alert_floor": protected_review_alert_floor,
                "protected_immediate_alert_floor": protected_floor,
                "suppress": int(counts.get("suppress", 0)),
                "passive": int(counts.get("passive", 0)),
                "alert": alert_count,
                "escalate": expert_count,
                "expert_demand_served_percent": (
                    100.0 * expert_count / expert_demand if expert_demand else 100.0
                ),
                "critical_spillover_to_frontline": spillover,
                "frontline_overrun_bundles": frontline_overrun,
                "final_alerts_per_100_order_bundles": (
                    100.0 * alert_count / total_order_bundles
                    if total_order_bundles
                    else 0.0
                ),
                "final_expert_reviews_per_100_order_bundles": (
                    100.0 * expert_count / total_order_bundles
                    if total_order_bundles
                    else 0.0
                ),
                "immediate_interruption_reduction_vs_B1_percent": (
                    100.0 * (1.0 - alert_count / affected_bundles)
                    if affected_bundles
                    else 0.0
                ),
                "capacity_binding": bool(
                    spillover
                    or frontline_overrun
                    or expert_count < expert_demand
                    or group["capacity_status"].isin(
                        {
                            "frontline_capacity_not_selected_fallback",
                            "expert_capacity_not_selected_fallback",
                        }
                    ).any()
                ),
            }
        )
    return pd.DataFrame(rows)
