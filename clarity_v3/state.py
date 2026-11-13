from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import pandas as pd

from .models import ACTIONS, ActiveMedication, ContextState, DetectedSignal, OrderEvent
from .mimic import build_order_active_snapshots
from .notes import NoteFactStore
from .reference import ReferenceStore
from .temporal import EpisodeMemory, evidence_recency_index
from .transplant import TransplantContextEngine
from .transplant_phase import protocol_phase_for_context

ProgressCallback = Callable[[int, int, int], None]


@dataclass(frozen=True)
class StateBuilderConfig:
    creatinine_recency_half_life_hours: float = 48.0
    high_severity_floor: float = 8.0
    high_evidence_support_threshold: float = 0.70
    high_completeness_threshold: float = 0.70
    moderate_certainty_product_threshold: float = 0.35
    minimum_note_context_confidence: float = 0.80
    acute_renal_harm_multiplier: float = 1.25
    maximum_contextual_harm: float = 12.0

    def __post_init__(self) -> None:
        if self.creatinine_recency_half_life_hours <= 0:
            raise ValueError("creatinine_recency_half_life_hours must be positive")
        if self.high_severity_floor <= 0:
            raise ValueError("high_severity_floor must be positive")
        for name in (
            "high_evidence_support_threshold",
            "high_completeness_threshold",
            "moderate_certainty_product_threshold",
            "minimum_note_context_confidence",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _optional_float(value: object) -> Optional[float]:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else float(parsed)


def _optional_timestamp(value: object) -> Optional[pd.Timestamp]:
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _episode_key(row: pd.Series) -> str:
    family = str(row["rule_family"])
    if family == "drug_drug_interaction":
        return f"DDI:{row.get('drug_pair', row['rule_id'])}"
    if family == "duplicate_therapy":
        return (
            f"DUP:{row.get('therapy_class', '')}:"
            f"{row.get('overlapping_ingredient_keys', row['rule_id'])}"
        )
    if family == "renal_dose_adjustment":
        return f"RENAL:{row['rule_id']}:{row.get('new_ingredient_key', '')}"
    if family == "beers_criteria":
        return f"BEERS:{row['rule_id']}:{row.get('new_ingredient_key', '')}"
    return f"{family}:{row['rule_id']}"


def _build_active_snapshots(
    orders: pd.DataFrame,
    relevant_order_ids: set[str],
    progress_callback: Optional[ProgressCallback] = None,
    progress_every: int = 5000,
) -> Dict[str, tuple[ActiveMedication, ...]]:
    return build_order_active_snapshots(
        orders,
        relevant_order_ids=relevant_order_ids,
        progress_callback=progress_callback,
        progress_every=progress_every,
    )


class ContextStateBuilder:
    def __init__(
        self,
        reference: ReferenceStore,
        note_store: Optional[NoteFactStore] = None,
        config: Optional[StateBuilderConfig] = None,
        note_context_mode: str = "disabled",
    ) -> None:
        if note_context_mode not in {"disabled", "conservative", "secondary_analysis"}:
            raise ValueError(
                "note_context_mode must be disabled, conservative, or secondary_analysis"
            )
        self.reference = reference
        self.note_store = note_store or NoteFactStore([])
        self.config = config or StateBuilderConfig()
        self.note_context_mode = note_context_mode
        self.transplant = TransplantContextEngine(reference)

    def _completeness(self, signal: pd.Series) -> tuple[float, tuple[str, ...]]:
        missing: list[str] = []
        family = str(signal["rule_family"])
        if family == "renal_dose_adjustment":
            recency = evidence_recency_index(
                _optional_float(signal.get("hours_since_current_creatinine")),
                self.config.creatinine_recency_half_life_hours,
            )
            for column in ("order_dose", "order_dose_unit", "order_frequency"):
                if pd.isna(signal.get(column)) or str(signal.get(column)).strip() == "":
                    missing.append(column)
            dose_factor = 1.0 if not missing else 0.75
            if _as_bool(signal.get("crrt_active")):
                missing.append("renal_metric_confounded_by_crrt")
                dose_factor *= 0.6
            return max(0.0, min(1.0, recency * dose_factor)), tuple(sorted(set(missing)))
        if family in {"drug_drug_interaction", "duplicate_therapy"}:
            if pd.isna(signal.get("new_ingredient_key")):
                missing.append("new_ingredient_key")
            return (0.85 if missing else 1.0), tuple(missing)
        return 1.0, ()

    def _certainty_class(
        self,
        evidence_support: float,
        completeness: float,
    ) -> str:
        if (
            evidence_support >= self.config.high_evidence_support_threshold
            and completeness >= self.config.high_completeness_threshold
        ):
            return "high"
        if evidence_support * completeness >= self.config.moderate_certainty_product_threshold:
            return "moderate"
        return "low"

    def build(
        self,
        signals: pd.DataFrame,
        orders: pd.DataFrame,
        progress_callback_orders: Optional[ProgressCallback] = None,
        progress_callback_signals: Optional[ProgressCallback] = None,
        progress_every: int = 5000,
    ) -> pd.DataFrame:
        if signals.empty:
            return pd.DataFrame()
        required_orders = set(signals["order_id"].astype(str))
        snapshots = _build_active_snapshots(
            orders,
            required_orders,
            progress_callback=progress_callback_orders,
            progress_every=progress_every,
        )
        order_lookup = orders.set_index("order_id", drop=False)
        if len(order_lookup) != order_lookup.index.nunique():
            raise ValueError("order_id is not unique in the order timeline")
        memory = EpisodeMemory()
        results: list[dict] = []
        ordered_signals = signals.sort_values(
            ["subject_id", "timestamp", "hadm_id", "order_id", "rule_id"], kind="stable"
        )
        total = len(ordered_signals)
        for position, (_, signal_row) in enumerate(ordered_signals.iterrows(), start=1):
            order_id = str(signal_row["order_id"])
            if order_id not in order_lookup.index:
                raise ValueError(f"Signal references missing order_id: {order_id}")
            order_row = order_lookup.loc[order_id]
            active_meds = snapshots.get(order_id, ())
            order = OrderEvent(
                subject_id=int(order_row["subject_id"]),
                hadm_id=int(order_row["hadm_id"]),
                order_id=order_id,
                bundle_id=str(order_row["bundle_id"]),
                raw_name=str(order_row["drug"]),
                start_time=pd.Timestamp(order_row["timestamp"]),
                stop_time=_optional_timestamp(order_row.get("stoptime")),
                route=(str(order_row["route"]) if pd.notna(order_row.get("route")) else None),
                dose=_optional_float(order_row.get("dose_val_rx")),
                dose_unit=(
                    str(order_row["dose_unit_rx"])
                    if pd.notna(order_row.get("dose_unit_rx"))
                    else None
                ),
                frequency=(
                    str(order_row["frequency"])
                    if pd.notna(order_row.get("frequency"))
                    else None
                ),
                source_row_index=int(order_row["source_rx_row_index"]),
            )
            context = {
                key: signal_row.get(key)
                for key in signal_row.index
                if key not in {
                    "signal_id", "subject_id", "hadm_id", "order_id", "bundle_id",
                    "timestamp", "new_drug", "new_ingredient_key", "rule_id",
                    "rule_family", "message", "recommended_action", "source_id",
                    "base_severity", "evidence_support_index", "severity_category",
                }
            }
            signal = DetectedSignal(
                signal_id=str(signal_row["signal_id"]),
                subject_id=int(signal_row["subject_id"]),
                hadm_id=int(signal_row["hadm_id"]),
                order_id=order_id,
                bundle_id=str(signal_row["bundle_id"]),
                timestamp=pd.Timestamp(signal_row["timestamp"]),
                new_drug=str(signal_row["new_drug"]),
                new_ingredient_key=str(signal_row["new_ingredient_key"]),
                rule_id=str(signal_row["rule_id"]),
                rule_family=str(signal_row["rule_family"]),
                message=str(signal_row["message"]),
                recommended_action=str(signal_row["recommended_action"]),
                source_id=str(signal_row["source_id"]),
                base_severity=float(signal_row["base_severity"]),
                evidence_support_index=float(signal_row["evidence_support_index"]),
                severity_category=str(signal_row["severity_category"]),
                context=context,
            )
            drug_pair = None if pd.isna(signal_row.get("drug_pair")) else str(signal_row.get("drug_pair"))
            ingredient_keys = [signal.new_ingredient_key]
            partner_key = signal_row.get("partner_ingredient_key")
            if pd.notna(partner_key) and str(partner_key).strip():
                ingredient_keys.append(str(partner_key).strip())
            overlap_keys = signal_row.get("overlapping_ingredient_keys")
            if pd.notna(overlap_keys):
                ingredient_keys.extend(
                    value.strip() for value in str(overlap_keys).split("|") if value.strip()
                )
            note_facts = self.note_store.context_for_signal(
                signal.subject_id,
                signal.hadm_id,
                signal.timestamp,
                signal.rule_id,
                drug_pair,
                ingredient_keys,
            )
            intent_fact = note_facts["current_clinical_intent"]
            monitoring_fact = note_facts["current_monitoring_plan"]
            prior_management_fact = note_facts["prior_management_history"]
            adverse_fact = note_facts["prior_adverse_reaction"]
            intent_available = bool(
                intent_fact is not None
                and intent_fact.final_confidence >= self.config.minimum_note_context_confidence
            )
            monitoring_available = bool(
                monitoring_fact is not None
                and monitoring_fact.final_confidence >= self.config.minimum_note_context_confidence
            )
            prior_management_available = bool(
                prior_management_fact is not None
                and prior_management_fact.final_confidence >= self.config.minimum_note_context_confidence
            )
            adverse_available = bool(
                adverse_fact is not None
                and adverse_fact.final_confidence >= self.config.minimum_note_context_confidence
            )
            note_available = bool(
                intent_available or monitoring_available
                or prior_management_available or adverse_available
            )
            intent_usable = bool(
                self.note_context_mode == "secondary_analysis" and intent_available
            )
            monitoring_usable = bool(
                self.note_context_mode == "secondary_analysis" and monitoring_available
            )
            adverse_usable = bool(
                self.note_context_mode in {"conservative", "secondary_analysis"}
                and adverse_available
            )
            # Prior management is preserved as longitudinal audit context only.
            # It never establishes current tolerance, current monitoring, or a
            # lower-intensity action.
            prior_management_usable = False
            note_usable = bool(intent_usable or monitoring_usable or adverse_usable)
            note_context_types = "|".join(
                name for name, value in (
                    ("current_clinical_intent", intent_available),
                    ("current_monitoring_plan", monitoring_available),
                    ("prior_interaction_management_history", prior_management_available),
                    ("prior_adverse_reaction", adverse_available),
                ) if value
            )
            used_note_context_types = "|".join(
                name for name, value in (
                    ("current_clinical_intent", intent_usable),
                    ("current_monitoring_plan", monitoring_usable),
                    ("prior_adverse_reaction", adverse_usable),
                ) if value
            )
            actual_transplant_phase = str(
                order_row.get("transplant_phase", "prevalent_phase_unknown")
            )
            post_transplant_phase_context_allowed = _as_bool(
                order_row.get(
                    "post_transplant_phase_context_allowed",
                    order_row.get("phase_specific_context_allowed", True),
                )
            )
            # Use the persisted protocol phase when available; otherwise map
            # the fine-grained order phase through the central helper.
            persisted_protocol_phase = order_row.get("transplant_protocol_phase")
            if pd.notna(persisted_protocol_phase) and str(persisted_protocol_phase).strip():
                protocol_phase = str(persisted_protocol_phase)
            else:
                protocol_phase = protocol_phase_for_context(
                    actual_transplant_phase,
                    post_transplant_phase_context_allowed,
                )
            peri_transplant_order = _as_bool(
                order_row.get("peri_transplant_order", False)
            )
            assessment = self.transplant.assess(
                signal,
                order,
                active_meds,
                transplant_phase=protocol_phase,
                hours_since_creatinine=_optional_float(order_row.get("hours_since_current_creatinine")),
                acute_renal_deterioration=_as_bool(order_row.get("acute_renal_deterioration")),
                documented_intent=intent_usable,
                documented_monitoring_plan=monitoring_usable,
            )
            completeness, missingness = self._completeness(signal_row)
            certainty = self._certainty_class(signal.evidence_support_index, completeness)
            acute = _as_bool(order_row.get("acute_renal_deterioration"))
            contextual_harm = float(signal.base_severity)
            if acute:
                contextual_harm *= self.config.acute_renal_harm_multiplier
            contextual_harm = min(self.config.maximum_contextual_harm, contextual_harm)
            material_signature = (
                str(acute),
                actual_transplant_phase,
                str(post_transplant_phase_context_allowed),
                str(peri_transplant_order),
                str(order_row.get("transplant_episode_id", "unknown_episode")),
                str(order_row.get("careunit", "unknown")),
                str(order.route or "unknown"),
                str(order.dose if order.dose is not None else "unknown"),
                assessment.protocol_fit,
                assessment.monitoring_status,
                assessment.management_status,
                certainty,
                used_note_context_types or "no_note_context_used",
            )
            episode = memory.observe(
                signal.subject_id,
                signal.hadm_id,
                _episode_key(signal_row),
                signal.timestamp,
                _optional_timestamp(signal_row.get("episode_effective_end")),
                material_signature,
            )
            actions = set(assessment.admissible_actions)
            if signal.base_severity >= self.config.high_severity_floor:
                if certainty == "high" and assessment.management_status == "unmanaged":
                    safety_class = "must_interrupt"
                    actions = {"alert"}
                else:
                    safety_class = "must_review_or_interrupt"
                    actions &= {"alert", "escalate"}
                    if not actions:
                        actions = {"alert", "escalate"}
            elif acute or episode["material_change_since_prior"]:
                safety_class = "no_suppress"
                actions.discard("suppress")
            elif adverse_usable:
                safety_class = "no_suppress"
                actions.discard("suppress")
            elif signal.base_severity >= 5 or certainty == "high":
                safety_class = "no_suppress"
                actions.discard("suppress")
            else:
                safety_class = "standard"
            if not actions:
                actions = {"passive"}
            context_reason = "; ".join(
                value for value in (
                    assessment.reason,
                    f"certainty={certainty}",
                    f"episode={episode['repeat_status']}",
                    f"acute_renal={acute}",
                    f"transplant_phase={actual_transplant_phase}",
                    f"post_transplant_phase_context_allowed={post_transplant_phase_context_allowed}",
                    f"peri_transplant_order={peri_transplant_order}",
                    f"protocol_phase_used={protocol_phase}",
                    f"transplant_episode={order_row.get('transplant_episode_id', 'unknown')}",
                    f"note_context_mode={self.note_context_mode}",
                    f"note_context_available={note_context_types or 'none'}",
                    f"note_context_used={used_note_context_types or 'none'}",
                ) if value
            )
            state = ContextState(
                signal_id=signal.signal_id,
                subject_id=signal.subject_id,
                hadm_id=signal.hadm_id,
                order_id=signal.order_id,
                bundle_id=signal.bundle_id,
                timestamp=signal.timestamp,
                rule_id=signal.rule_id,
                rule_family=signal.rule_family,
                base_severity=signal.base_severity,
                evidence_support_index=signal.evidence_support_index,
                contextual_harm_index=contextual_harm,
                evidence_completeness_index=completeness,
                certainty_class=certainty,
                current_creatinine=_optional_float(order_row.get("current_creatinine")),
                hours_since_current_creatinine=_optional_float(order_row.get("hours_since_current_creatinine")),
                acute_renal_deterioration=acute,
                acute_renal_deterioration_basis=str(order_row.get("acute_deterioration_basis", "")),
                crrt_active=_as_bool(order_row.get("crrt_active")),
                transplant_patient=True,
                transplant_date_source=str(order_row.get("transplant_date_source", "")),
                transplant_phase=actual_transplant_phase,
                days_since_transplant=(
                    None if pd.isna(order_row.get("days_since_transplant"))
                    else int(order_row.get("days_since_transplant"))
                ),
                protocol_fit=assessment.protocol_fit,
                monitoring_status=assessment.monitoring_status,
                transition_status=assessment.transition_status,
                episode_id=str(episode["episode_id"]),
                episode_occurrence_index=int(episode["episode_occurrence_index"]),
                prior_lifetime_occurrences=int(episode["prior_lifetime_occurrences"]),
                prior_admission_occurrences=int(episode["prior_admission_occurrences"]),
                hours_since_prior_same_signal=episode["hours_since_prior_same_signal"],
                material_change_since_prior=bool(episode["material_change_since_prior"]),
                repeat_status=str(episode["repeat_status"]),
                note_context_mode=self.note_context_mode,
                note_context_available=note_available,
                note_context_usable=note_usable,
                management_status=assessment.management_status,
                safety_class=safety_class,
                admissible_actions=tuple(action for action in ACTIONS if action in actions),
                missingness_flags=missingness,
                context_reason=context_reason,
                note_context_types=note_context_types,
                note_context_source_count=(
                    int(intent_available) + int(monitoring_available)
                    + int(prior_management_available) + int(adverse_available)
                ),
                current_clinical_intent_documented=intent_available,
                current_monitoring_plan_documented=monitoring_available,
                prior_adverse_reaction_documented=adverse_available,
                prior_interaction_management_documented=prior_management_available,
            )
            state_record = state.as_dict()
            state_record.update(
                {
                    "transplant_episode_id": order_row.get("transplant_episode_id"),
                    "transplant_episode_number": order_row.get(
                        "transplant_episode_number"
                    ),
                    "transplant_hadm_id": order_row.get("transplant_hadm_id"),
                    "transplant_anchor_quality": order_row.get(
                        "transplant_anchor_quality"
                    ),
                    "transplant_anchor_quality_level": order_row.get(
                        "transplant_anchor_quality_level"
                    ),
                    "episode_phase_specific_context_allowed": _as_bool(
                        order_row.get("episode_phase_specific_context_allowed", False)
                    ),
                    "phase_specific_context_allowed": post_transplant_phase_context_allowed,
                    "post_transplant_phase_context_allowed": post_transplant_phase_context_allowed,
                    "peri_transplant_order": peri_transplant_order,
                    "short_gap_repeat_transplant_candidate": _as_bool(
                        order_row.get("short_gap_repeat_transplant_candidate")
                    ),
                    "protocol_phase_used": protocol_phase,
                }
            )
            results.append(state_record)
            if progress_callback_signals and (position % progress_every == 0 or position == total):
                progress_callback_signals(position, total, len(results))
        frame = pd.DataFrame(results)
        if len(frame) != len(signals):
            raise ValueError(
                f"Context-state preservation failed: {len(frame)} states for {len(signals)} signals"
            )
        if frame["signal_id"].duplicated().any():
            raise ValueError("Context-state signal_id values are not unique")
        return frame
