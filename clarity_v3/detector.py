from __future__ import annotations

import hashlib
from typing import Iterable, List, Optional

import pandas as pd

from .models import ActiveMedication, DetectedSignal, OrderEvent
from .reference import ReferenceStore


def _signal_id(order: OrderEvent, rule_id: str) -> str:
    raw = f"{order.subject_id}|{order.hadm_id}|{order.order_id}|{rule_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _number(value: object) -> Optional[float]:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else float(parsed)


class DeterministicDetector:
    """Shared deterministic hazard detector.

    This module deliberately preserves repeated event-level rule firings. It
    answers only whether a rule fires; later stages determine episode novelty,
    presentation intensity, resource allocation, and explanation.
    """

    def __init__(self, reference: ReferenceStore):
        self.reference = reference
        self._ddi_by_ingredient: dict[str, list[pd.Series]] = {}
        for _, row in self.reference.ddi_rules.iterrows():
            for key in (str(row["drug_a_key"]), str(row["drug_b_key"])):
                self._ddi_by_ingredient.setdefault(key, []).append(row)
        self._duplicate_by_ingredient: dict[str, list[pd.Series]] = {}
        for _, row in self.reference.duplicate_rules.iterrows():
            members = [
                value.strip()
                for value in str(row["member_ingredient_keys_pipe"]).split("|")
                if value.strip()
            ]
            for key in members:
                self._duplicate_by_ingredient.setdefault(key, []).append(row)

    def detect(
        self,
        order: OrderEvent,
        active_meds: Iterable[ActiveMedication],
        patient_age: int,
        renal_metric: Optional[float],
    ) -> List[DetectedSignal]:
        classification = self.reference.classify(order.raw_name)
        if not classification.is_drug or not classification.ingredient_key:
            return []
        active = list(active_meds)
        new_key = classification.ingredient_key
        signals: List[DetectedSignal] = []
        signals.extend(self._ddi(order, new_key, active))
        signals.extend(self._renal(order, new_key, renal_metric))
        signals.extend(self._duplicates(order, new_key, active))
        signals.extend(self._beers(order, new_key, patient_age))
        return signals

    def _build(
        self,
        order: OrderEvent,
        rule: pd.Series,
        family: str,
        message: str,
        context: dict,
    ) -> DetectedSignal:
        classification = self.reference.classify(order.raw_name)
        return DetectedSignal(
            signal_id=_signal_id(order, str(rule["rule_id"])),
            subject_id=int(order.subject_id),
            hadm_id=int(order.hadm_id),
            order_id=str(order.order_id),
            bundle_id=str(order.bundle_id),
            timestamp=pd.Timestamp(order.start_time),
            new_drug=str(order.raw_name),
            new_ingredient_key=str(classification.ingredient_key),
            rule_id=str(rule["rule_id"]),
            rule_family=family,
            message=message,
            recommended_action=str(rule.get("recommended_action", "")),
            source_id=str(rule["source_id"]),
            base_severity=float(rule["severity_score_nccmerp"]),
            evidence_support_index=float(rule["signal_confidence_numeric"]),
            severity_category=str(rule["nccmerp_category"]),
            context=context,
        )

    def _ddi(
        self,
        order: OrderEvent,
        new_key: str,
        active: List[ActiveMedication],
    ) -> List[DetectedSignal]:
        active_by_key = {med.ingredient_key: med for med in active}
        active_keys = set(active_by_key)
        results: List[DetectedSignal] = []
        for rule in self._ddi_by_ingredient.get(new_key, []):
            pair = {str(rule["drug_a_key"]), str(rule["drug_b_key"])}
            partner_key = next(iter(pair - {new_key}), new_key)
            if partner_key not in active_keys:
                continue
            partner = active_by_key[partner_key]
            overlap_end_candidates = [
                value for value in (order.stop_time, partner.stop_time) if value is not None
            ]
            overlap_end = min(overlap_end_candidates) if overlap_end_candidates else None
            results.append(
                self._build(
                    order,
                    rule,
                    "drug_drug_interaction",
                    f"{rule['drug_a']} + {rule['drug_b']}: {rule['mechanism']}",
                    {
                        "partner_ingredient_key": partner_key,
                        "partner_raw_name": partner.raw_name,
                        "partner_start_time": partner.start_time,
                        "partner_stop_time": partner.stop_time,
                        "partner_route": partner.route,
                        "drug_pair": "|".join(sorted(pair)),
                        "new_drug_stop_time": order.stop_time,
                        "episode_effective_end": overlap_end,
                    },
                )
            )
        return results

    def _renal(
        self,
        order: OrderEvent,
        new_key: str,
        renal_metric: Optional[float],
    ) -> List[DetectedSignal]:
        if renal_metric is None or pd.isna(renal_metric):
            return []
        results: List[DetectedSignal] = []
        matches = self.reference.renal_rules[
            self.reference.renal_rules["ingredient_key"].astype(str) == new_key
        ]
        for _, rule in matches.iterrows():
            threshold = float(rule["threshold_value"])
            op = str(rule["threshold_operator"])
            fires = {
                "<": renal_metric < threshold,
                "<=": renal_metric <= threshold,
                ">": renal_metric > threshold,
                ">=": renal_metric >= threshold,
            }.get(op)
            if fires is None:
                raise ValueError(
                    f"Unsupported threshold_operator {op!r} for rule {rule['rule_id']}"
                )
            if not fires:
                continue
            results.append(
                self._build(
                    order,
                    rule,
                    "renal_dose_adjustment",
                    (
                        f"{rule['ingredient']}: renal screening metric {renal_metric:.1f} "
                        f"{op} {threshold:g} {rule.get('threshold_units', '')}. "
                        f"{rule['recommended_action']}"
                    ),
                    {
                        "renal_metric": float(renal_metric),
                        "threshold_operator": op,
                        "threshold_value": threshold,
                        "order_dose": order.dose,
                        "order_dose_unit": order.dose_unit,
                        "order_route": order.route,
                        "order_frequency": order.frequency,
                        "new_drug_stop_time": order.stop_time,
                        "episode_effective_end": order.stop_time,
                    },
                )
            )
        return results

    def _duplicates(
        self,
        order: OrderEvent,
        new_key: str,
        active: List[ActiveMedication],
    ) -> List[DetectedSignal]:
        active_by_key = {med.ingredient_key: med for med in active}
        results: List[DetectedSignal] = []
        for rule in self._duplicate_by_ingredient.get(new_key, []):
            members = {
                value.strip()
                for value in str(rule["member_ingredient_keys_pipe"]).split("|")
                if value.strip()
            }
            overlaps = sorted((members & set(active_by_key)) - {new_key})
            if not overlaps:
                continue
            active_overlap = [active_by_key[key] for key in overlaps]
            stop_candidates = [
                value for value in [order.stop_time] + [med.stop_time for med in active_overlap]
                if value is not None
            ]
            overlap_end = min(stop_candidates) if stop_candidates else None
            ingredients = sorted(set(overlaps + [new_key]))
            results.append(
                self._build(
                    order,
                    rule,
                    "duplicate_therapy",
                    f"Duplicate {rule['therapy_class']} therapy: {', '.join(ingredients)}",
                    {
                        "therapy_class": str(rule["therapy_class"]),
                        "overlapping_ingredient_keys": "|".join(ingredients),
                        "drug_pair": "|".join(ingredients),
                        "overlapping_raw_names": "|".join(
                            sorted(str(active_by_key[key].raw_name) for key in overlaps)
                        ),
                        "overlap_routes": "|".join(
                            sorted(
                                str(active_by_key[key].route or "unknown")
                                for key in overlaps
                            )
                        ),
                        "overlap_start_times": "|".join(
                            str(active_by_key[key].start_time) for key in overlaps
                        ),
                        "allowed_overlap_hours": _number(rule.get("allowed_overlap_hours")),
                        "context_exception_defined": str(rule.get("context_exception", "")),
                        "new_drug_stop_time": order.stop_time,
                        "episode_effective_end": overlap_end,
                    },
                )
            )
        return results

    def _beers(
        self,
        order: OrderEvent,
        new_key: str,
        patient_age: int,
    ) -> List[DetectedSignal]:
        results: List[DetectedSignal] = []
        matches = self.reference.beers_rules[
            self.reference.beers_rules["ingredient_key"].astype(str) == new_key
        ]
        for _, rule in matches.iterrows():
            minimum = int(rule["age_min_years"])
            if patient_age < minimum:
                continue
            results.append(
                self._build(
                    order,
                    rule,
                    "beers_criteria",
                    f"{rule['ingredient']}: {rule['rationale_short_paraphrased']}",
                    {
                        "patient_age": int(patient_age),
                        "age_min_years": minimum,
                        "new_drug_stop_time": order.stop_time,
                        "episode_effective_end": order.stop_time,
                    },
                )
            )
        return results
