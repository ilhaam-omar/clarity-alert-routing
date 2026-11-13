from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd

ACTIONS: Tuple[str, ...] = ("suppress", "passive", "alert", "escalate")
NON_RESOURCE_ACTIONS: Tuple[str, ...] = ("suppress", "passive")
RESOURCE_ACTIONS: Tuple[str, ...] = ("alert", "escalate")


@dataclass(frozen=True)
class DrugClassification:
    raw_name: str
    ingredient_key: Optional[str]
    canonical_ingredient: Optional[str]
    category: str

    @property
    def is_drug(self) -> bool:
        return self.category == "DRUG"


@dataclass(frozen=True)
class ActiveMedication:
    raw_name: str
    ingredient_key: str
    start_time: pd.Timestamp
    stop_time: Optional[pd.Timestamp]
    route: Optional[str] = None
    dose: Optional[float] = None
    dose_unit: Optional[str] = None


@dataclass(frozen=True)
class OrderEvent:
    subject_id: int
    hadm_id: int
    order_id: str
    bundle_id: str
    raw_name: str
    start_time: pd.Timestamp
    stop_time: Optional[pd.Timestamp] = None
    route: Optional[str] = None
    dose: Optional[float] = None
    dose_unit: Optional[str] = None
    frequency: Optional[str] = None
    source_row_index: Optional[int] = None


@dataclass(frozen=True)
class DetectedSignal:
    signal_id: str
    subject_id: int
    hadm_id: int
    order_id: str
    bundle_id: str
    timestamp: pd.Timestamp
    new_drug: str
    new_ingredient_key: str
    rule_id: str
    rule_family: str
    message: str
    recommended_action: str
    source_id: str
    base_severity: float
    evidence_support_index: float
    severity_category: str
    context: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        context = row.pop("context", {})
        row.update(context)
        return row


@dataclass(frozen=True)
class ContextState:
    """Time-causal state available before action selection.

    Numeric values are transparent indices used for prototype stress testing;
    they are not interpreted as calibrated clinical probabilities.
    """

    signal_id: str
    subject_id: int
    hadm_id: int
    order_id: str
    bundle_id: str
    timestamp: pd.Timestamp
    rule_id: str
    rule_family: str
    base_severity: float
    evidence_support_index: float
    contextual_harm_index: float
    evidence_completeness_index: float
    certainty_class: str
    current_creatinine: Optional[float]
    hours_since_current_creatinine: Optional[float]
    acute_renal_deterioration: bool
    acute_renal_deterioration_basis: str
    crrt_active: bool
    transplant_patient: bool
    transplant_date_source: str
    transplant_phase: str
    days_since_transplant: Optional[int]
    protocol_fit: str
    monitoring_status: str
    transition_status: str
    episode_id: str
    episode_occurrence_index: int
    prior_lifetime_occurrences: int
    prior_admission_occurrences: int
    hours_since_prior_same_signal: Optional[float]
    material_change_since_prior: bool
    repeat_status: str
    note_context_usable: bool
    management_status: str
    safety_class: str
    admissible_actions: Tuple[str, ...]
    missingness_flags: Tuple[str, ...]
    context_reason: str
    note_context_mode: str = "disabled"
    note_context_available: bool = False
    note_context_types: str = ""
    note_context_source_count: int = 0
    current_clinical_intent_documented: bool = False
    current_monitoring_plan_documented: bool = False
    prior_adverse_reaction_documented: bool = False
    prior_interaction_management_documented: bool = False

    def as_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["admissible_actions"] = "|".join(self.admissible_actions)
        row["missingness_flags"] = "|".join(self.missingness_flags)
        return row


@dataclass(frozen=True)
class ActionSetDecision:
    signal_id: str
    raw_winner_safety_priority: str
    raw_winner_workflow_balanced: str
    raw_winner_burden_priority: str
    robust_action_set: str
    consensus_action: Optional[str]
    consensus_support_floor: float
    calibration_sensitive: bool
    pre_capacity_action: str
    fallback_non_resource_action: str
    admissible_actions: str
    safety_class: str
    protected_alert: bool
    protected_review: bool
    delta_frontline: float
    delta_expert: float
    action_support_suppress_min: float
    action_support_passive_min: float
    action_support_alert_min: float
    action_support_escalate_min: float
    robust_loss_suppress_max: float
    robust_loss_passive_max: float
    robust_loss_alert_max: float
    robust_loss_escalate_max: float
    decision_reason: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BundleDecision:
    bundle_id: str
    subject_id: int
    hadm_id: int
    timestamp: pd.Timestamp
    signal_count: int
    signal_ids: str
    highest_base_severity: float
    safety_class: str
    admissible_actions: str
    pre_capacity_action: str
    fallback_non_resource_action: str
    protected_alert: bool
    protected_review: bool
    calibration_sensitive: bool
    delta_frontline: float
    delta_expert: float
    bundle_reason: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CapacityScenario:
    """Normalized Phase-1 capacity-response scenario.

    MIMIC-IV preserves chronology within a patient but not concurrency across
    patients. Consequently, Phase 1 does not reconstruct hospital shifts.
    Frontline capacity is expressed as a nominal interruptive-alert density per
    100 prescribing bundles. Expert capacity is expressed as the fraction of
    pre-capacity expert-review demand that can be served. These values are
    stress-test assumptions, not estimates of observed staffing.
    """

    name: str
    allocation_scope: str
    frontline_alerts_per_100_order_bundles: Optional[float]
    expert_review_coverage_fraction: Optional[float]
    unconstrained: bool = False
    critical_reviews_spill_to_frontline: bool = True
    description: str = ""

    def validate(self) -> None:
        if self.allocation_scope != "cohort_normalized":
            raise ValueError(
                "allocation_scope must be 'cohort_normalized' for MIMIC Phase 1"
            )
        if self.unconstrained:
            if self.expert_review_coverage_fraction not in (None, 1.0):
                raise ValueError(
                    "An unconstrained scenario must use expert coverage None or 1.0"
                )
            if (
                self.frontline_alerts_per_100_order_bundles is not None
                and self.frontline_alerts_per_100_order_bundles < 0
            ):
                raise ValueError("Frontline alert density cannot be negative")
            return
        if self.frontline_alerts_per_100_order_bundles is None:
            raise ValueError("A frontline alert-density threshold is required")
        if self.frontline_alerts_per_100_order_bundles < 0:
            raise ValueError("Frontline alert density cannot be negative")
        if self.expert_review_coverage_fraction is None:
            raise ValueError("An expert-review coverage fraction is required")
        if not 0.0 <= self.expert_review_coverage_fraction <= 1.0:
            raise ValueError("expert_review_coverage_fraction must be between 0 and 1")


@dataclass(frozen=True)
class FinalBundleDecision:
    bundle_id: str
    scenario: str
    allocation_scope: str
    epoch_start: str
    pre_capacity_action: str
    final_action: str
    capacity_status: str
    frontline_rank: Optional[int]
    expert_rank: Optional[int]
    frontline_slots: int
    expert_slots: int
    total_order_bundles: int
    pre_capacity_alert_demand: int
    pre_capacity_expert_demand: int
    frontline_alerts_per_100_order_bundles: Optional[float]
    expert_review_coverage_fraction: Optional[float]
    protected_alert: bool
    protected_review: bool
    capacity_reason: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_actions(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ACTIONS
    if isinstance(value, str):
        values: Iterable[str] = value.split("|")
    else:
        values = value
    result = tuple(str(v).strip() for v in values if str(v).strip())
    invalid = set(result) - set(ACTIONS)
    if invalid:
        raise ValueError(f"Unknown action(s): {sorted(invalid)}")
    if not result:
        raise ValueError("At least one admissible action is required")
    return result