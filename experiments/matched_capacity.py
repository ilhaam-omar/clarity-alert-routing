"""Absolute-budget allocator for the matched Phase-1 comparison.

This module deliberately does not import the CLARITY engine. It preserves the
original controller's route precedence and safety spillover while accepting the
same integer budgets for every policy. Severity ranking reads no optimisation
scores. Suppression eligibility and a common non-resource fallback are supplied
by the caller; the allocator does not compute clinical eligibility.
"""
from __future__ import annotations

from numbers import Integral

import pandas as pd


ACTIONS = {"suppress", "passive", "alert", "escalate"}
OUTPUT_COLUMNS = [
    "bundle_id", "scenario", "pre_capacity_action", "final_action",
    "capacity_status", "frontline_slots", "expert_slots", "protected_alert",
    "protected_review", "review_rank", "frontline_rank", "total_order_bundles",
]


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _allowed(value: object) -> set[str]:
    if isinstance(value, (tuple, list, set)):
        return {str(item) for item in value if str(item) in ACTIONS}
    if pd.isna(value):
        return set()
    return {item.strip() for item in str(value).split("|") if item.strip() in ACTIONS}


def _budget(value: int, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    value = int(value)
    if value < (1 if positive else 0):
        raise ValueError(f"{label} must be {'positive' if positive else 'non-negative'}")
    return value


def allocate_absolute(
    bundles: pd.DataFrame,
    frontline_slots: int,
    expert_slots: int,
    ranking: str,
    scenario: str,
    total_order_bundles: int,
) -> pd.DataFrame:
    """Allocate fixed F/K budgets without changing initial policy choices.

    Required alerts are never rationed. An unserved protected review becomes an
    immediate alert, even if this exceeds F. Only an initial escalation competes
    for K. Other resource actions use the supplied admissible fallback when
    unserved. Every final suppression must pass the caller's common gate.

    ``severity`` sorts highest_base_severity descending then bundle_id ascending.
    ``regret`` uses delta_expert for reviews and delta_frontline for alerts, with
    the same identifier tie-break. Ties are analytical, not hospital queue order.
    """
    if ranking not in {"severity", "regret"}:
        raise ValueError("ranking must be 'severity' or 'regret'")
    frontline_slots = _budget(frontline_slots, "frontline_slots")
    expert_slots = _budget(expert_slots, "expert_slots")
    total_order_bundles = _budget(total_order_bundles, "total_order_bundles", positive=True)
    required = {
        "bundle_id", "pre_capacity_action", "fallback_non_resource_action",
        "admissible_actions", "protected_alert", "protected_review",
        "bundle_suppression_eligible",
    }
    required |= {"highest_base_severity"} if ranking == "severity" else {
        "delta_frontline", "delta_expert"
    }
    missing = required - set(bundles.columns)
    if missing:
        raise ValueError(f"Bundle table missing columns: {sorted(missing)}")
    if bundles.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    working = bundles.copy()
    if working["bundle_id"].isna().any():
        raise ValueError("Bundle identifiers cannot be missing")
    working["bundle_id"] = working["bundle_id"].astype(str)
    if working["bundle_id"].duplicated().any():
        raise ValueError("Bundle table contains duplicate bundle_id values")
    if len(working) > total_order_bundles:
        raise ValueError("Affected bundles exceed total_order_bundles")
    working["pre_capacity_action"] = working["pre_capacity_action"].astype(str)
    if not working["pre_capacity_action"].isin(ACTIONS).all():
        raise ValueError("Initial route is outside the four defined actions")
    for name in ("protected_alert", "protected_review", "bundle_suppression_eligible"):
        working[name] = working[name].map(_as_bool)
    working["_allowed"] = working["admissible_actions"].map(_allowed)
    if working["_allowed"].map(len).eq(0).any():
        raise ValueError("Every bundle requires a nonempty admissible action set")
    if ranking == "regret":
        # Match the original allocator's handling of absent numerical gains.
        for name in ("delta_frontline", "delta_expert"):
            working[name] = pd.to_numeric(working[name], errors="coerce").fillna(0.0)
    else:
        # No read or conversion of delta_frontline/delta_expert in this arm.
        working["highest_base_severity"] = pd.to_numeric(
            working["highest_base_severity"], errors="raise"
        )
        if working["highest_base_severity"].isna().any():
            raise ValueError("Severity ranking requires nonmissing severity")

    def ordered(frame: pd.DataFrame, resource: str) -> pd.DataFrame:
        column = "highest_base_severity" if ranking == "severity" else resource
        return frame.sort_values([column, "bundle_id"], ascending=[False, True], kind="stable")

    def fallback(row: pd.Series) -> str:
        allowed = row["_allowed"]
        selected = str(row["fallback_non_resource_action"])
        if selected in allowed and selected in {"suppress", "passive"}:
            result = selected
        elif "passive" in allowed:
            result = "passive"
        elif "suppress" in allowed:
            result = "suppress"
        else:
            raise ValueError(f"Bundle {row['bundle_id']} has no admissible non-resource fallback")
        if result == "suppress" and not row["bundle_suppression_eligible"]:
            raise ValueError(f"Suppression gate failed for fallback bundle {row['bundle_id']}")
        return result

    alert_mask = working["pre_capacity_action"].eq("alert")
    review_mask = working["pre_capacity_action"].eq("escalate") & ~working["protected_alert"]
    protected_review_alert_mask = alert_mask & working["protected_review"] & ~working["protected_alert"]
    protected_immediate_mask = working["protected_alert"] | protected_review_alert_mask
    review_candidates = ordered(working[review_mask], "delta_expert")
    alert_candidates = ordered(working[alert_mask], "delta_frontline")
    review_rank = {key: i + 1 for i, key in enumerate(review_candidates["bundle_id"])}
    frontline_rank = {key: i + 1 for i, key in enumerate(alert_candidates["bundle_id"])}
    assigned: dict[str, str] = {}
    status: dict[str, str] = {}

    # 1. Required immediate responses establish the safety floor.
    immediate_used = 0
    for _, row in ordered(working[protected_immediate_mask], "delta_frontline").iterrows():
        key = row["bundle_id"]
        immediate_used += 1
        assigned[key] = "alert"
        prefix = "protected_immediate_alert" if row["protected_alert"] else "protected_review_alert"
        status[key] = prefix + ("_within_nominal" if immediate_used <= frontline_slots else "_overrun")

    # 2. Only the preselected review candidate pool consumes expert slots.
    served_review = set(review_candidates.head(expert_slots)["bundle_id"])
    for _, row in review_candidates.iterrows():
        key = row["bundle_id"]
        if key in assigned:
            continue
        if key in served_review:
            assigned[key] = "escalate"
            status[key] = "expert_capacity_selected"
        elif row["protected_review"]:
            assigned[key] = "alert"
            immediate_used += 1
            status[key] = "protected_review_spillover" + (
                "_within_nominal" if immediate_used <= frontline_slots else "_overrun"
            )
        else:
            assigned[key] = fallback(row)
            status[key] = "expert_capacity_not_selected_fallback"

    # 3. Discretionary initial alerts compete for the remaining nominal F slots.
    discretionary = working[
        alert_mask & ~protected_immediate_mask & ~working["bundle_id"].isin(assigned)
    ]
    remaining_frontline = max(0, frontline_slots - immediate_used)
    served_frontline = set(ordered(discretionary, "delta_frontline").head(remaining_frontline)["bundle_id"])
    for _, row in discretionary.iterrows():
        key = row["bundle_id"]
        if key in served_frontline:
            assigned[key] = "alert"
            status[key] = "frontline_capacity_selected"
            immediate_used += 1
        else:
            assigned[key] = fallback(row)
            status[key] = "frontline_capacity_not_selected_fallback"

    # 4. Preserve initial non-resource actions; allocation cannot create demand.
    for _, row in working.iterrows():
        key = row["bundle_id"]
        if key in assigned:
            continue
        action = row["pre_capacity_action"]
        if action in {"suppress", "passive"}:
            assigned[key] = action
            status[key] = "non_resource_pre_capacity_action"
        else:
            assigned[key] = fallback(row)
            status[key] = "unhandled_resource_action_fallback"

    records = []
    for _, row in working.iterrows():
        key = row["bundle_id"]
        action = assigned[key]
        if action not in row["_allowed"]:
            raise ValueError(f"Final action {action} is inadmissible for bundle {key}")
        if action == "suppress" and not row["bundle_suppression_eligible"]:
            raise ValueError(f"Suppression gate failed for final bundle {key}")
        if row["protected_alert"] and action != "alert":
            raise AssertionError(f"Protected alert lost for {key}")
        if row["protected_review"] and action not in {"alert", "escalate"}:
            raise AssertionError(f"Protected review lost for {key}")
        records.append({
            "bundle_id": key, "scenario": str(scenario),
            "pre_capacity_action": row["pre_capacity_action"], "final_action": action,
            "capacity_status": status[key], "frontline_slots": frontline_slots,
            "expert_slots": expert_slots, "protected_alert": row["protected_alert"],
            "protected_review": row["protected_review"],
            "review_rank": review_rank.get(key), "frontline_rank": frontline_rank.get(key),
            "total_order_bundles": total_order_bundles,
        })
    result = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    if len(result) != len(working) or result["bundle_id"].duplicated().any():
        raise AssertionError("Allocation did not preserve one final row per bundle")
    return result
