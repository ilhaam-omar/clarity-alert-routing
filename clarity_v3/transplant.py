from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import pandas as pd

from .models import ACTIONS, ActiveMedication, DetectedSignal, OrderEvent
from .reference import ReferenceStore


@dataclass(frozen=True)
class TransplantAssessment:
    transplant_patient: bool
    protocol_fit: str
    monitoring_status: str
    transition_status: str
    management_status: str
    admissible_actions: Tuple[str, ...]
    reason: str


_STEROID_EQUIVALENCE_TO_PREDNISONE = {
    "prednisone": 1.0,
    "prednisolone": 1.0,
    "methylprednisolone": 1.25,
    "dexamethasone": 6.67,
    "hydrocortisone": 0.25,
}


def _ingredient_steroid_factor(value: str) -> Optional[float]:
    lowered = str(value).lower()
    for ingredient, factor in _STEROID_EQUIVALENCE_TO_PREDNISONE.items():
        if ingredient in lowered:
            return factor
    return None


def _prednisone_equivalent(dose: object, unit: object, ingredient: str) -> Optional[float]:
    parsed = pd.to_numeric(pd.Series([dose]), errors="coerce").iloc[0]
    if pd.isna(parsed) or str(unit).lower().strip() not in {"mg", "milligram", "milligrams"}:
        return None
    factor = _ingredient_steroid_factor(ingredient)
    return None if factor is None else float(parsed) * factor


def confirmed_steroid_decline(
    order: OrderEvent,
    active_meds: Sequence[ActiveMedication],
) -> bool:
    current = _prednisone_equivalent(order.dose, order.dose_unit, order.raw_name)
    if current is None:
        return False
    prior: list[float] = []
    for med in active_meds:
        value = _prednisone_equivalent(med.dose, med.dose_unit, med.ingredient_key)
        if value is not None:
            prior.append(value)
    return bool(prior and max(prior) > current)


def route_transition(
    signal: DetectedSignal,
    order: OrderEvent,
    active_meds: Sequence[ActiveMedication],
) -> bool:
    if signal.rule_family != "duplicate_therapy" or not order.route:
        return False
    ingredient_set = {
        value for value in str(signal.context.get("overlapping_ingredient_keys", "")).split("|")
        if value
    }
    previous_routes = {
        str(med.route).upper()
        for med in active_meds
        if med.route and med.ingredient_key in ingredient_set
    }
    return bool(previous_routes and str(order.route).upper() not in previous_routes)


class TransplantContextEngine:
    """Apply phase-aware transplant context before action routing.

    A protocol match never removes the underlying hazard. It can establish that
    the combination is expected or that an observed transition is plausible,
    but phase-unknown or monitoring-incomplete cases remain conservatively
    visible.
    """

    def __init__(self, reference: ReferenceStore):
        self.reference = reference
        self._protocol_by_id = {
            str(row["rule_id"]): row
            for _, row in reference.transplant_protocol_rules.iterrows()
        }
        self._phase_by_protocol = {}
        if not reference.transplant_phase_scope.empty:
            for _, row in reference.transplant_phase_scope.iterrows():
                self._phase_by_protocol[str(row["transplant_rule_id"])] = row

    def _phase_fit(self, protocol_id: str, phase: str) -> str:
        row = self._phase_by_protocol.get(protocol_id)
        if row is None:
            return "phase_scope_not_registered"
        scopes = {
            value.strip()
            for value in str(row.get("phase_scope_pipe", "")).split("|")
            if value.strip()
        }
        required = str(row.get("phase_required", "0")).strip().lower() in {"1", "true", "yes"}
        if phase in scopes:
            return "phase_specific_match" if required else "phase_independent_match"
        if phase == "prevalent_phase_unknown":
            return "phase_unknown_no_narrowing" if required else "phase_independent_match"
        return "phase_mismatch_no_narrowing" if required else "phase_independent_match"

    @staticmethod
    def _monitoring_status(
        protocol_row: pd.Series,
        hours_since_creatinine: Optional[float],
    ) -> str:
        required = str(protocol_row.get("required_monitoring", "")).lower()
        availability = str(protocol_row.get("monitoring_availability_flag", "")).lower()
        if not required or required in {"nan", "none_additional", "none_structured"}:
            return "not_required_or_not_structured"
        has_creatinine = "creatinine" in required
        unavailable_components = "unavailable" in availability or "partial" in availability
        if has_creatinine:
            if hours_since_creatinine is None or pd.isna(hours_since_creatinine):
                return "required_monitoring_missing"
            if float(hours_since_creatinine) > 72:
                return "required_monitoring_stale"
            return "partial_available" if unavailable_components else "available_within_72h"
        return "not_observable_in_primary_mimic_state"

    def assess(
        self,
        signal: DetectedSignal,
        order: OrderEvent,
        active_meds: Sequence[ActiveMedication],
        transplant_phase: str,
        hours_since_creatinine: Optional[float],
        acute_renal_deterioration: bool,
        documented_intent: bool = False,
        documented_monitoring_plan: bool = False,
    ) -> TransplantAssessment:
        actions = set(ACTIONS)
        reasons: list[str] = []
        if signal.base_severity >= 8:
            actions.discard("suppress")
            reasons.append("high-severity floor removes suppress")
        if acute_renal_deterioration:
            actions.discard("suppress")
            reasons.append("acute renal deterioration removes suppress")

        decline = confirmed_steroid_decline(order, active_meds)
        route_change = route_transition(signal, order, active_meds)
        transition_status = (
            "confirmed_steroid_decline"
            if decline
            else ("route_transition" if route_change else "none")
        )
        if transition_status != "none" and signal.base_severity < 8:
            actions &= {"suppress", "passive"}
            reasons.append("observed time-causal transition limits interruptive actions")

        protocol_fit = "no_matching_transplant_protocol"
        monitoring_status = "not_assessed"
        management_status = "unmanaged"
        override = self.reference.transplant_override(signal.rule_id)
        if override is not None:
            protocol_id = str(override.get("transplant_rule_id", ""))
            protocol_row = self._protocol_by_id.get(protocol_id)
            if protocol_row is not None:
                protocol_fit = self._phase_fit(protocol_id, transplant_phase)
                monitoring_status = self._monitoring_status(
                    protocol_row, hours_since_creatinine
                )
                override_type = str(override.get("override_type", ""))
                phase_allows = protocol_fit in {
                    "phase_specific_match",
                    "phase_independent_match",
                }
                if override_type == "requires_temporal_pattern_trigger":
                    if transition_status != "none" and phase_allows:
                        management_status = "transition_confirmed"
                    else:
                        reasons.append("temporal transition or phase support not established")
                elif override_type == "require_documented_intent_before_narrowing":
                    if documented_intent:
                        management_status = "intent_documented"
                        if signal.base_severity < 8:
                            actions.discard("alert")
                    else:
                        reasons.append("current, rule-specific documented intent required; phase alone is insufficient")
                elif override_type in {
                    "narrow_ambiguity_toward_passive",
                    "narrow_ambiguity_toward_escalate",
                }:
                    if phase_allows and monitoring_status == "available_within_72h":
                        management_status = "protocol_supported_with_observed_monitoring"
                        if signal.base_severity < 8:
                            actions.discard("alert")
                    elif phase_allows and monitoring_status in {
                        "partial_available",
                        "not_required_or_not_structured",
                    }:
                        management_status = "protocol_possible_incomplete_monitoring"
                        actions.discard("suppress")
                    elif phase_allows and documented_monitoring_plan:
                        management_status = "protocol_possible_documented_monitoring_plan"
                        monitoring_status = "documented_plan_structured_result_unverified"
                        actions.discard("suppress")
                    elif phase_allows:
                        management_status = "protocol_possible_monitoring_unresolved"
                        actions.discard("suppress")
                    else:
                        reasons.append("phase scope does not support protocol narrowing")

        if transition_status != "none":
            management_status = "transition_confirmed"
        if not actions:
            actions = {"passive"}
        return TransplantAssessment(
            transplant_patient=True,
            protocol_fit=protocol_fit,
            monitoring_status=monitoring_status,
            transition_status=transition_status,
            management_status=management_status,
            admissible_actions=tuple(action for action in ACTIONS if action in actions),
            reason="; ".join(reasons) if reasons else "no transplant-specific narrowing applied",
        )
