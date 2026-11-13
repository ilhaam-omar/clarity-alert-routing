from __future__ import annotations

"""Transparent comparator policies for CLARITY Phase 1.

The primary comparator in this module is a *snapshot severity-evidence matrix*.
It deliberately reads only Stage-02 detected-signal fields that exist at the
current medication-order event:

* base rule severity;
* source/evidence support for that rule;
* provider-order identifiers and timestamp.

It does not read Stage-03 longitudinal state, renal trajectory, transplant
phase, protocol fit, previous occurrences, material change, management status,
contextual safety classes, admissible actions, or robust-regret outputs.

This makes it a transparent engineering baseline for testing the incremental
value of the full CLARITY architecture. It is not claimed to reproduce any
specific proprietary CPOE implementation.
"""

from typing import Any, Iterable

import numpy as np
import pandas as pd

ACTIONS = ("suppress", "passive", "alert", "escalate")


# ---------------------------------------------------------------------------
# Primary snapshot comparator
# ---------------------------------------------------------------------------
def snapshot_matrix_action(high_severity: bool, high_evidence: bool) -> str:
    """Map current-order severity and rule evidence to one of four actions."""
    if high_severity and high_evidence:
        return "alert"
    if high_severity and not high_evidence:
        return "escalate"
    if not high_severity and high_evidence:
        return "passive"
    return "suppress"


def snapshot_matrix_cell(high_severity: bool, high_evidence: bool) -> str:
    return (
        ("high_severity" if high_severity else "lower_severity")
        + "__"
        + ("high_evidence" if high_evidence else "lower_evidence")
    )


def _snapshot_safety(action: str) -> tuple[str, str, bool, bool, str]:
    """Return safety/admissibility fields derived only from the matrix cell.

    These are comparator-internal safeguards, not CLARITY Stage-03 safety
    classes. High-severity/high-evidence signals remain immediate alerts;
    high-severity/lower-evidence signals remain alert-or-review cases.
    """
    if action == "alert":
        return "must_interrupt", "alert", True, False, "passive"
    if action == "escalate":
        return "must_review_or_interrupt", "alert|escalate", False, True, "passive"
    if action == "passive":
        return "standard", "suppress|passive|alert|escalate", False, False, "passive"
    return "standard", "suppress|passive|alert|escalate", False, False, "suppress"


def build_snapshot_signal_routes(
    signals: pd.DataFrame,
    *,
    severity_threshold: float,
    evidence_threshold: float,
) -> pd.DataFrame:
    """Create signal-level snapshot routes from Stage-02 output only.

    Parameters are prespecified threshold values, not fitted probabilities.
    ``evidence_support_index`` is the detector's source/evidence score and is
    intentionally not replaced by Stage-03 completeness or certainty classes.
    """
    required = {
        "signal_id",
        "bundle_id",
        "subject_id",
        "hadm_id",
        "timestamp",
        "base_severity",
        "evidence_support_index",
    }
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(
            "Stage-02 signal file is missing snapshot-comparator columns: "
            f"{sorted(missing)}"
        )
    if signals.empty:
        raise ValueError("Stage-02 signal file is empty")

    working = signals.copy()
    working["base_severity"] = pd.to_numeric(
        working["base_severity"], errors="coerce"
    )
    working["evidence_support_index"] = pd.to_numeric(
        working["evidence_support_index"], errors="coerce"
    )
    if working["base_severity"].isna().any():
        bad = working.loc[working["base_severity"].isna(), "signal_id"].astype(str).head(5)
        raise ValueError(f"Missing base_severity for signal(s): {bad.tolist()}")
    if working["evidence_support_index"].isna().any():
        bad = working.loc[
            working["evidence_support_index"].isna(), "signal_id"
        ].astype(str).head(5)
        raise ValueError(f"Missing evidence_support_index for signal(s): {bad.tolist()}")
    if not np.isfinite(working["base_severity"]).all():
        raise ValueError("base_severity contains non-finite values")
    if not np.isfinite(working["evidence_support_index"]).all():
        raise ValueError("evidence_support_index contains non-finite values")

    records: list[dict[str, Any]] = []
    for _, row in working.iterrows():
        severity = float(row["base_severity"])
        evidence = float(row["evidence_support_index"])
        high_severity = severity >= float(severity_threshold)
        high_evidence = evidence >= float(evidence_threshold)
        action = snapshot_matrix_action(high_severity, high_evidence)
        safety_class, admissible, protected_alert, protected_review, fallback = (
            _snapshot_safety(action)
        )

        # Independent, transparent capacity-ranking scores. They use only the
        # current rule severity and evidence support—never CLARITY context or
        # regret-derived opportunity gains.
        delta_frontline = severity * max(evidence, 0.0) if action == "alert" else 0.0
        delta_expert = (
            severity * max(1.0 - min(max(evidence, 0.0), 1.0), 0.0)
            if action == "escalate"
            else 0.0
        )

        record = row.to_dict()
        record.update(
            {
                "snapshot_matrix_cell": snapshot_matrix_cell(
                    high_severity, high_evidence
                ),
                "snapshot_high_severity": bool(high_severity),
                "snapshot_high_evidence": bool(high_evidence),
                "snapshot_severity_threshold": float(severity_threshold),
                "snapshot_evidence_threshold": float(evidence_threshold),
                "snapshot_pre_capacity_action": action,
                "pre_capacity_action": action,
                "snapshot_action_reason": (
                    "four_cell_current_order_severity_and_evidence_matrix"
                ),
                "safety_class": safety_class,
                "admissible_actions": admissible,
                "fallback_non_resource_action": fallback,
                "protected_alert": protected_alert,
                "protected_review": protected_review,
                "calibration_sensitive": False,
                "delta_frontline": float(delta_frontline),
                "delta_expert": float(delta_expert),
                "snapshot_input_scope": (
                    "stage02_current_rule_severity_and_evidence_only"
                ),
            }
        )
        records.append(record)

    result = pd.DataFrame(records)
    if len(result) != len(signals):
        raise AssertionError("Snapshot routing did not preserve the signal denominator")
    if result["signal_id"].duplicated().any():
        raise ValueError("Snapshot routing produced duplicate signal identifiers")
    return result


# ---------------------------------------------------------------------------
# Backwards-compatible names retained so older regression tests/imports do not
# fail after installation. The final Stage-10B script uses the snapshot API.
# ---------------------------------------------------------------------------
def static_matrix_action(high_severity: bool, high_certainty: bool) -> str:
    return snapshot_matrix_action(high_severity, high_certainty)


def static_matrix_cell(high_severity: bool, high_certainty: bool) -> str:
    return snapshot_matrix_cell(high_severity, high_certainty)


def parse_actions(value: Any) -> tuple[str, ...]:
    if value is None or (not isinstance(value, (list, tuple, set)) and pd.isna(value)):
        return ACTIONS
    values: Iterable[Any] = value.split("|") if isinstance(value, str) else value
    actions = tuple(str(item).strip() for item in values if str(item).strip() in ACTIONS)
    return actions or ACTIONS


_ACTION_ORDER = {
    "suppress": ("suppress", "passive", "escalate", "alert"),
    "passive": ("passive", "suppress", "escalate", "alert"),
    "alert": ("alert", "escalate", "passive", "suppress"),
    "escalate": ("escalate", "alert", "passive", "suppress"),
}


def apply_static_guardrails(
    preferred: str,
    *,
    safety_class: str,
    admissible_actions: Iterable[str],
    high_certainty: bool,
) -> tuple[str, str]:
    """Legacy helper retained for old tests; not used by snapshot Stage 10B."""
    allowed = tuple(action for action in admissible_actions if action in ACTIONS) or ACTIONS
    if safety_class == "must_interrupt":
        return "alert", "hard_safety_must_interrupt"
    if safety_class == "must_review_or_interrupt":
        preferred_resource = "alert" if high_certainty else "escalate"
        for action in (preferred_resource, "alert", "escalate"):
            if action in allowed:
                return action, "hard_safety_review_or_interrupt"
    if safety_class == "no_suppress" and preferred == "suppress":
        preferred = "passive"
    for action in _ACTION_ORDER.get(preferred, ACTIONS):
        if action in allowed:
            return action, (
                "matrix_cell" if action == preferred else "admissibility_adjustment"
            )
    raise ValueError(f"No admissible action remained for safety_class={safety_class!r}")


def build_static_signal_routes(
    frame: pd.DataFrame,
    severity_threshold: float,
    evidence_threshold: float = 0.7,
) -> pd.DataFrame:
    """Compatibility wrapper.

    New Stage-10B inputs should be Stage-02 signals and include
    ``evidence_support_index``. Older context-informed inputs with only
    ``certainty_class`` remain supported for historical tests, but are not used
    in the final comparator.
    """
    if "evidence_support_index" in frame.columns:
        return build_snapshot_signal_routes(
            frame,
            severity_threshold=severity_threshold,
            evidence_threshold=evidence_threshold,
        )

    # Historical compatibility path only.
    required = {
        "signal_id",
        "bundle_id",
        "subject_id",
        "hadm_id",
        "timestamp",
        "base_severity",
        "certainty_class",
        "safety_class",
        "admissible_actions",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Legacy static input is missing columns: {sorted(missing)}")
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        severity = float(pd.to_numeric(pd.Series([row["base_severity"]]), errors="raise").iloc[0])
        high_severity = severity >= float(severity_threshold)
        high_certainty = str(row.get("certainty_class", "")).strip().lower() == "high"
        unconstrained = static_matrix_action(high_severity, high_certainty)
        action, reason = apply_static_guardrails(
            unconstrained,
            safety_class=str(row.get("safety_class", "standard")),
            admissible_actions=parse_actions(row.get("admissible_actions")),
            high_certainty=high_certainty,
        )
        record = row.to_dict()
        record.update(
            {
                "static_matrix_cell": static_matrix_cell(high_severity, high_certainty),
                "static_unconstrained_action": unconstrained,
                "pre_capacity_action": action,
                "static_adjustment_reason": reason,
                "fallback_non_resource_action": "passive",
                "protected_alert": str(row.get("safety_class")) == "must_interrupt",
                "protected_review": str(row.get("safety_class")) == "must_review_or_interrupt",
                "calibration_sensitive": False,
                "delta_frontline": 0.0,
                "delta_expert": 0.0,
            }
        )
        records.append(record)
    return pd.DataFrame(records)
