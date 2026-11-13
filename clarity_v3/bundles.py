from __future__ import annotations

from typing import Iterable

import pandas as pd

from .models import ACTIONS, BundleDecision

_SAFETY_RANK = {
    "standard": 0,
    "no_suppress": 1,
    "must_review_or_interrupt": 2,
    "must_interrupt": 3,
}


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _highest_safety(values: Iterable[object]) -> str:
    names = [str(value) for value in values]
    return max(names, key=lambda value: _SAFETY_RANK.get(value, -1))


def build_order_bundles(decisions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate related signal rows into one prescribing-workflow bundle.

    A bundle is the unit of frontline interruption and expert review. Signal
    rows remain available for clinical content and audit; capacity is not
    consumed once per signal when several signals arise from the same order.
    """
    if decisions.empty:
        return pd.DataFrame()
    required = {
        "bundle_id", "subject_id", "hadm_id", "timestamp", "signal_id",
        "base_severity", "safety_class", "pre_capacity_action",
        "fallback_non_resource_action", "protected_alert", "protected_review",
        "calibration_sensitive", "delta_frontline", "delta_expert",
    }
    missing = required - set(decisions.columns)
    if missing:
        raise ValueError(f"Action-set table missing columns: {sorted(missing)}")
    results: list[dict] = []
    for bundle_id, group in decisions.groupby("bundle_id", sort=False):
        group = group.sort_values(["timestamp", "signal_id"], kind="stable")
        safety_class = _highest_safety(group["safety_class"])
        protected_alert = bool(group["protected_alert"].map(_bool).any())
        protected_review = bool(group["protected_review"].map(_bool).any())
        actions = list(group["pre_capacity_action"].astype(str))
        if protected_alert or "alert" in actions:
            pre_capacity = "alert"
        elif protected_review or "escalate" in actions:
            pre_capacity = "escalate"
        elif "passive" in actions:
            pre_capacity = "passive"
        else:
            pre_capacity = "suppress"
        fallback = (
            "passive"
            if "passive" in set(group["fallback_non_resource_action"].astype(str))
            else "suppress"
        )
        if safety_class == "must_interrupt":
            admissible = ("alert",)
        elif safety_class == "must_review_or_interrupt":
            admissible = ("alert", "escalate")
        elif safety_class == "no_suppress":
            admissible = ("passive", "alert", "escalate")
        else:
            admissible = ACTIONS
        reason = (
            f"{len(group)} signal(s) consolidated into one order bundle; "
            f"highest safety class={safety_class}; signal actions="
            + "|".join(sorted(set(actions)))
        )
        record = BundleDecision(
            bundle_id=str(bundle_id),
            subject_id=int(group.iloc[0]["subject_id"]),
            hadm_id=int(group.iloc[0]["hadm_id"]),
            timestamp=pd.Timestamp(group.iloc[0]["timestamp"]),
            signal_count=int(len(group)),
            signal_ids="|".join(group["signal_id"].astype(str)),
            highest_base_severity=float(group["base_severity"].max()),
            safety_class=safety_class,
            admissible_actions="|".join(admissible),
            pre_capacity_action=pre_capacity,
            fallback_non_resource_action=fallback,
            protected_alert=protected_alert,
            protected_review=protected_review,
            calibration_sensitive=bool(group["calibration_sensitive"].map(_bool).any()),
            delta_frontline=float(
                pd.to_numeric(group["delta_frontline"], errors="coerce")
                .replace([float("inf"), float("-inf")], pd.NA)
                .dropna()
                .clip(lower=0)
                .sum()
            ),
            delta_expert=float(
                pd.to_numeric(group["delta_expert"], errors="coerce")
                .replace([float("inf"), float("-inf")], pd.NA)
                .dropna()
                .clip(lower=0)
                .sum()
            ),
            bundle_reason=reason,
        )
        results.append(record.as_dict())
    frame = pd.DataFrame(results)
    if frame["bundle_id"].duplicated().any():
        raise ValueError("Bundle aggregation produced duplicate bundle_id values")
    return frame.sort_values(
        ["timestamp", "hadm_id", "bundle_id"], kind="stable"
    ).reset_index(drop=True)
