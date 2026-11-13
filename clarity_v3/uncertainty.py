from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd

from .models import ACTIONS, ActionSetDecision, ContextState, parse_actions

_ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}
_TIE_ORDER = ("escalate", "alert", "passive", "suppress")


@dataclass(frozen=True)
class EnvelopeConfig:
    name: str
    n_points: int
    seed: int
    regret_quantile: float
    ranges: Mapping[str, tuple[float, float]]
    # Grounded (judgement-defined) design: when non-empty, the ranges are the definitional
    # domain and only weightings that satisfy every listed (reference case, judgement) pair
    # are kept. Empty means the declared-range design used before.
    anchor_rules: tuple = ()


@dataclass(frozen=True)
class ActionSetConfig:
    consensus_support_threshold: float = 0.70
    suppress_support_threshold: float = 0.80
    high_severity_floor: float = 8.0

    def validate(self) -> None:
        for value in (self.consensus_support_threshold, self.suppress_support_threshold):
            if not 0 < value <= 1:
                raise ValueError("Support thresholds must be in (0, 1]")


_REQUIRED_PARAMETERS = (
    "rho_passive",
    "rho_alert",
    "rho_escalate",
    "burden_passive",
    "burden_alert",
    "burden_escalate",
    "uncertainty_penalty",
    "uncertainty_alert_fraction",
    "uncertainty_escalate_fraction",
    "repeat_alert_penalty",
    "managed_residual_fraction",
    "partly_managed_residual_fraction",
    "escalation_delay_penalty",
)


def _lhs(n: int, d: int, seed: int) -> np.ndarray:
    if n < 32:
        raise ValueError("Each uncertainty envelope requires at least 32 points")
    rng = np.random.default_rng(seed)
    values = np.empty((n, d), dtype=float)
    for column in range(d):
        values[:, column] = (rng.permutation(n) + rng.random(n)) / n
    return values


def _scale(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    low, high = map(float, bounds)
    if high < low:
        raise ValueError(f"Invalid range {bounds}")
    return low + (high - low) * values


# Reference concerns for the grounded design: adjusted severity h, certainty q and the
# repeat burden used by the loss (an unchanged third repeat gives log(1+3)/log(10)).
# Each case lists the responses the safety rules admit for it: suppression is excluded for a new
# concern of moderate or high severity and allowed for an unchanged low-severity repeat.
ANCHOR_CASES = {
    "A1_serious_new": {"h": 0.8, "q": 0.9, "repeat": 0.0, "admissible": ("passive", "alert", "escalate")},
    "A2_low_unchanged_third_repeat": {"h": 0.3, "q": 0.9, "repeat": float(np.log1p(3) / np.log(10.0)),
                                      "admissible": ("suppress", "passive", "alert", "escalate")},
    "A3_moderate_new": {"h": 0.5, "q": 0.9, "repeat": 0.0, "admissible": ("passive", "alert", "escalate")},
    # serious concern whose evidence is doubtful (certainty 0.4): a pharmacist can resolve the doubt
    # that an interruption cannot, so review is the intended response (as in the worked example)
    "A4_serious_doubtful_new": {"h": 0.8, "q": 0.4, "repeat": 0.0, "admissible": ("passive", "alert", "escalate")},
}
ACT_RESPONSES = ("alert", "escalate")
QUIET_RESPONSES = ("suppress", "passive")


def _best(losses: dict, responses) -> np.ndarray:
    return np.minimum.reduce([losses[r] for r in responses])


def _judge(losses: dict, judgement: str, admissible) -> np.ndarray:
    if judgement == "review":   # pharmacist review preferred to every other admissible response
        others = [r for r in admissible if r != "escalate"]
        return losses["escalate"] < _best(losses, others)
    act = _best(losses, [r for r in ACT_RESPONSES if r in admissible])
    quiet = _best(losses, [r for r in QUIET_RESPONSES if r in admissible])
    if judgement == "act":      # interruption or review preferred to every quiet response
        return act < quiet
    if judgement == "quiet":    # a quiet response preferred to interruption and review
        return quiet < act
    if judgement == "review":   # pharmacist review preferred to every other admissible response
        others = [r for r in admissible if r != "escalate"]
        return losses["escalate"] < _best(losses, others)
    raise ValueError(f"Unknown judgement: {judgement}")


def anchor_losses(columns: Mapping[str, np.ndarray], h: float, q: float, repeat: float) -> dict:
    """Loss of each response for one reference concern, using the cost function of
    MultiEnvelopeActionPolicy._costs with no management credit."""
    u = (1.0 - q) * h
    lam = columns["uncertainty_penalty"]
    return {
        "suppress": q * h + lam * u,
        "passive": q * columns["rho_passive"] * h + lam * u + columns["burden_passive"],
        "alert": q * columns["rho_alert"] * h + lam * columns["uncertainty_alert_fraction"] * u
        + columns["burden_alert"] + columns["repeat_alert_penalty"] * repeat,
        "escalate": q * columns["rho_escalate"] * h + lam * columns["uncertainty_escalate_fraction"] * u
        + columns["burden_escalate"] + columns["escalation_delay_penalty"] * q * h,
    }


def anchor_mask(columns: Mapping[str, np.ndarray], rules) -> np.ndarray:
    n = len(next(iter(columns.values())))
    keep = np.ones(n, dtype=bool)
    for case, judgement in rules:
        if case not in ANCHOR_CASES:
            raise ValueError(f"Unknown reference case: {case}")
        spec = ANCHOR_CASES[case]
        losses = anchor_losses(columns, spec["h"], spec["q"], spec["repeat"])
        keep &= _judge(losses, judgement, spec["admissible"])
    return keep


def _enforce_orders(columns: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    # Enforce structural partial orders without claiming exact calibration.
    columns["rho_alert"] = np.minimum(columns["rho_alert"], columns["rho_passive"] - 1e-6)
    columns["rho_escalate"] = np.minimum(
        columns["rho_escalate"], columns["rho_passive"] - 1e-6
    )
    columns["burden_alert"] = np.maximum(
        columns["burden_alert"], columns["burden_passive"] + 1e-6
    )
    columns["burden_escalate"] = np.maximum(
        columns["burden_escalate"], columns["burden_passive"] + 1e-6
    )
    columns["uncertainty_escalate_fraction"] = np.minimum(
        columns["uncertainty_escalate_fraction"],
        columns["uncertainty_alert_fraction"],
    )
    columns["partly_managed_residual_fraction"] = np.maximum(
        columns["partly_managed_residual_fraction"],
        columns["managed_residual_fraction"],
    )
    return columns


def _draw(config: EnvelopeConfig, n: int, seed: int) -> Dict[str, np.ndarray]:
    u = _lhs(n, len(_REQUIRED_PARAMETERS), seed)
    columns: Dict[str, np.ndarray] = {}
    for index, parameter in enumerate(_REQUIRED_PARAMETERS):
        columns[parameter] = _scale(u[:, index], config.ranges[parameter])
    return _enforce_orders(columns)


def sample_envelope(config: EnvelopeConfig) -> pd.DataFrame:
    missing = set(_REQUIRED_PARAMETERS) - set(config.ranges)
    if missing:
        raise ValueError(f"Envelope {config.name} is missing ranges: {sorted(missing)}")
    if not config.anchor_rules:
        columns = _draw(config, config.n_points, config.seed)
    else:
        # Grounded design: draw candidate weightings over the definitional domain in
        # reproducible batches and keep the first n_points that satisfy every judgement.
        kept: Dict[str, list] = {p: [] for p in _REQUIRED_PARAMETERS}
        accepted, batch, drawn = 0, 0, 0
        while accepted < config.n_points:
            if batch >= 2000:
                raise ValueError(f"Envelope {config.name}: too few weightings satisfy its judgements")
            cand = _draw(config, 50_000, config.seed * 1000 + batch)
            mask = anchor_mask(cand, config.anchor_rules)
            for p in _REQUIRED_PARAMETERS:
                kept[p].append(cand[p][mask])
            accepted += int(mask.sum()); drawn += 50_000; batch += 1
        columns = {p: np.concatenate(v)[: config.n_points] for p, v in kept.items()}
    frame = pd.DataFrame(columns)
    frame.insert(0, "parameter_id", np.arange(config.n_points, dtype=int))
    frame.insert(0, "envelope", config.name)
    return frame


def sample_design(envelopes: Iterable[EnvelopeConfig]) -> pd.DataFrame:
    frames = [sample_envelope(envelope) for envelope in envelopes]
    if not frames:
        raise ValueError("At least one uncertainty envelope is required")
    return pd.concat(frames, ignore_index=True)


class MultiEnvelopeActionPolicy:
    """Identify robust action sets without treating design points as probabilities.

    Each envelope is a declared policy stress test. The policy reports the
    minimax-regret winner within each envelope and chooses a single pre-capacity
    action only when the envelopes agree or a hard safety/conservative fallback
    applies. Suppression is intentionally harder to justify than passive display.
    """

    def __init__(
        self,
        design: pd.DataFrame,
        envelope_quantiles: Mapping[str, float],
        config: ActionSetConfig | None = None,
    ) -> None:
        self.design = design.copy()
        self.envelope_quantiles = dict(envelope_quantiles)
        self.config = config or ActionSetConfig()
        self.config.validate()
        if self.design.empty:
            raise ValueError("Uncertainty design cannot be empty")
        missing = {"envelope", *_REQUIRED_PARAMETERS} - set(self.design.columns)
        if missing:
            raise ValueError(f"Uncertainty design missing columns: {sorted(missing)}")
        unknown = set(self.design["envelope"].astype(str)) - set(self.envelope_quantiles)
        if unknown:
            raise ValueError(f"Missing regret quantile for envelope(s): {sorted(unknown)}")
        self._groups = {
            str(name): group.reset_index(drop=True)
            for name, group in self.design.groupby("envelope", sort=False)
        }

    @staticmethod
    def _tie_break(values: Mapping[str, float]) -> str:
        minimum = min(values.values())
        tied = {action for action, value in values.items() if np.isclose(value, minimum)}
        for action in _TIE_ORDER:
            if action in tied:
                return action
        return sorted(tied)[0]

    @staticmethod
    def _management_factor(state: ContextState, frame: pd.DataFrame) -> np.ndarray:
        if state.management_status in {
            "transition_confirmed",
            "protocol_supported_with_observed_monitoring",
            "intent_documented",
        }:
            return frame["managed_residual_fraction"].to_numpy(float)
        if state.management_status.startswith("protocol_possible"):
            return frame["partly_managed_residual_fraction"].to_numpy(float)
        return np.ones(len(frame), dtype=float)

    def _costs(self, state: ContextState, frame: pd.DataFrame) -> np.ndarray:
        n = len(frame)
        harm = min(1.2, max(0.0, state.contextual_harm_index / 10.0))
        support_index = min(
            1.0,
            max(0.0, state.evidence_support_index * state.evidence_completeness_index),
        )
        adjusted_harm = harm * self._management_factor(state, frame)
        unresolved = (1.0 - support_index) * adjusted_harm
        repeat_burden = 0.0
        if state.repeat_status == "repeat_unchanged_state":
            repeat_burden = min(1.0, np.log1p(state.episode_occurrence_index) / np.log(10.0))
        elif state.repeat_status == "repeat_with_material_change":
            repeat_burden = 0.0

        values = np.empty((n, len(ACTIONS)), dtype=float)
        values[:, _ACTION_INDEX["suppress"]] = (
            support_index * adjusted_harm
            + frame["uncertainty_penalty"].to_numpy(float) * unresolved
        )
        values[:, _ACTION_INDEX["passive"]] = (
            support_index * frame["rho_passive"].to_numpy(float) * adjusted_harm
            + frame["uncertainty_penalty"].to_numpy(float) * unresolved
            + frame["burden_passive"].to_numpy(float)
        )
        values[:, _ACTION_INDEX["alert"]] = (
            support_index * frame["rho_alert"].to_numpy(float) * adjusted_harm
            + frame["uncertainty_penalty"].to_numpy(float)
            * frame["uncertainty_alert_fraction"].to_numpy(float)
            * unresolved
            + frame["burden_alert"].to_numpy(float)
            + frame["repeat_alert_penalty"].to_numpy(float) * repeat_burden
        )
        values[:, _ACTION_INDEX["escalate"]] = (
            support_index * frame["rho_escalate"].to_numpy(float) * adjusted_harm
            + frame["uncertainty_penalty"].to_numpy(float)
            * frame["uncertainty_escalate_fraction"].to_numpy(float)
            * unresolved
            + frame["burden_escalate"].to_numpy(float)
            + frame["escalation_delay_penalty"].to_numpy(float)
            * support_index
            * adjusted_harm
        )
        admissible = set(parse_actions(state.admissible_actions))
        for action in ACTIONS:
            if action not in admissible:
                values[:, _ACTION_INDEX[action]] = np.inf
        return values

    def _evaluate_envelope(self, state: ContextState, name: str, frame: pd.DataFrame) -> dict:
        costs = self._costs(state, frame)
        best_indices = np.argmin(costs, axis=1)
        best_cost = costs[np.arange(len(costs)), best_indices]
        regret = costs - best_cost[:, None]
        supports = {
            action: float(np.mean(best_indices == _ACTION_INDEX[action]))
            for action in ACTIONS
        }
        quantile = float(self.envelope_quantiles[name])
        losses = {
            action: (
                float(np.quantile(regret[:, _ACTION_INDEX[action]], quantile))
                if np.isfinite(regret[:, _ACTION_INDEX[action]]).all()
                else float("inf")
            )
            for action in ACTIONS
        }
        winner = self._tie_break(losses)
        return {"winner": winner, "supports": supports, "losses": losses}

    def decide(self, state: ContextState) -> ActionSetDecision:
        evaluated = {
            name: self._evaluate_envelope(state, name, frame)
            for name, frame in self._groups.items()
        }
        expected_names = ("clinical_safety_priority", "workflow_balanced", "alert_burden_priority")
        if not all(name in evaluated for name in expected_names):
            raise ValueError(
                "Required envelopes are clinical_safety_priority, workflow_balanced, and alert_burden_priority"
            )
        winners = [evaluated[name]["winner"] for name in expected_names]
        robust_set = tuple(action for action in ACTIONS if action in set(winners))
        consensus = winners[0] if len(set(winners)) == 1 else None
        support_min = {
            action: min(evaluated[name]["supports"][action] for name in expected_names)
            for action in ACTIONS
        }
        loss_max = {
            action: max(evaluated[name]["losses"][action] for name in expected_names)
            for action in ACTIONS
        }
        consensus_floor = support_min[consensus] if consensus else 0.0
        # Suppression has its own stricter threshold. Use that threshold for
        # the diagnostic flag as well as for the action-selection branch.
        required_support = (
            self.config.suppress_support_threshold
            if consensus == "suppress"
            else self.config.consensus_support_threshold
        )
        calibration_sensitive = (
            consensus is None
            or consensus_floor < required_support
        )
        admissible = set(parse_actions(state.admissible_actions))
        suppression_allowed = bool(
            "suppress" in admissible
            and consensus == "suppress"
            and consensus_floor >= self.config.suppress_support_threshold
            and state.safety_class == "standard"
            and not state.material_change_since_prior
        )
        protected_alert = bool(state.safety_class == "must_interrupt" and "alert" in admissible)
        protected_review = bool(
            state.safety_class == "must_review_or_interrupt"
            and bool({"alert", "escalate"}.intersection(admissible))
        )
        reasons: list[str] = [
            "envelope winners=" + ",".join(winners),
            f"consensus_support_floor={consensus_floor:.3f}",
        ]

        if protected_alert:
            pre_capacity = "alert"
            reasons.append("hard safety class requires immediate interruption")
        elif protected_review:
            candidates = {action: loss_max[action] for action in ("alert", "escalate") if action in admissible}
            pre_capacity = self._tie_break(candidates)
            reasons.append("hard safety class requires alert or expert review")
        elif suppression_allowed:
            pre_capacity = "suppress"
            reasons.append("suppression passed stricter cross-envelope support rule")
        elif (
            consensus is not None
            and consensus != "suppress"
            and consensus_floor >= self.config.consensus_support_threshold
        ):
            pre_capacity = consensus
            reasons.append("cross-envelope consensus action selected")
        elif state.base_severity >= self.config.high_severity_floor:
            pre_capacity = "escalate" if "escalate" in admissible else "alert"
            reasons.append("high-severity calibration-sensitive case uses resource action")
        elif "passive" in admissible:
            pre_capacity = "passive"
            reasons.append("calibration-sensitive noncritical case uses conservative passive fallback")
        else:
            candidates = {
                action: loss_max[action] for action in admissible
                if action != "suppress" or suppression_allowed
            }
            if not candidates:
                raise ValueError(
                    f"Signal {state.signal_id} has no safe admissible action after suppression checks"
                )
            pre_capacity = self._tie_break(candidates)
            reasons.append("fallback chosen from remaining admissible actions")

        non_resource = [action for action in ("suppress", "passive") if action in admissible]
        if non_resource:
            fallback = self._tie_break({action: loss_max[action] for action in non_resource})
            if fallback == "suppress" and not suppression_allowed:
                if "passive" not in admissible:
                    raise ValueError(
                        f"Signal {state.signal_id} has no safe non-resource fallback after suppression checks"
                    )
                fallback = "passive"
        else:
            fallback = "passive"

        delta_frontline = (
            loss_max[fallback] - loss_max["alert"]
            if "alert" in admissible and np.isfinite(loss_max[fallback])
            else float("-inf")
        )
        best_non_expert = min(
            loss_max[action] for action in admissible if action != "escalate"
        ) if any(action != "escalate" for action in admissible) else float("inf")
        delta_expert = (
            best_non_expert - loss_max["escalate"]
            if "escalate" in admissible
            else float("-inf")
        )
        return ActionSetDecision(
            signal_id=state.signal_id,
            raw_winner_safety_priority=evaluated["clinical_safety_priority"]["winner"],
            raw_winner_workflow_balanced=evaluated["workflow_balanced"]["winner"],
            raw_winner_burden_priority=evaluated["alert_burden_priority"]["winner"],
            robust_action_set="|".join(robust_set),
            consensus_action=consensus,
            consensus_support_floor=float(consensus_floor),
            calibration_sensitive=bool(calibration_sensitive),
            pre_capacity_action=pre_capacity,
            fallback_non_resource_action=fallback,
            admissible_actions="|".join(state.admissible_actions),
            safety_class=state.safety_class,
            protected_alert=protected_alert,
            protected_review=protected_review,
            delta_frontline=float(delta_frontline),
            delta_expert=float(delta_expert),
            action_support_suppress_min=support_min["suppress"],
            action_support_passive_min=support_min["passive"],
            action_support_alert_min=support_min["alert"],
            action_support_escalate_min=support_min["escalate"],
            robust_loss_suppress_max=loss_max["suppress"],
            robust_loss_passive_max=loss_max["passive"],
            robust_loss_alert_max=loss_max["alert"],
            robust_loss_escalate_max=loss_max["escalate"],
            decision_reason="; ".join(reasons),
        )

    @staticmethod
    def _bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if pd.isna(value):
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}

    def decide_dataframe(self, states: pd.DataFrame) -> pd.DataFrame:
        """Evaluate unique controller signatures once, then map to all events."""
        if states.empty:
            return pd.DataFrame()
        signature_columns = [
            "base_severity", "evidence_support_index", "contextual_harm_index",
            "evidence_completeness_index", "certainty_class", "management_status",
            "repeat_status", "episode_occurrence_index", "material_change_since_prior",
            "admissible_actions", "safety_class",
        ]
        missing = set(signature_columns + ["signal_id"]) - set(states.columns)
        if missing:
            raise ValueError(f"State table missing columns: {sorted(missing)}")
        unique = states.loc[:, signature_columns].drop_duplicates().reset_index(drop=True)
        unique["_signature_id"] = np.arange(len(unique), dtype=int)
        working = states.merge(unique, on=signature_columns, how="left", validate="many_to_one")
        decisions: list[dict] = []
        for _, row in unique.iterrows():
            state = ContextState(
                signal_id=f"STATE-{int(row['_signature_id'])}",
                subject_id=0, hadm_id=0, order_id="", bundle_id="",
                timestamp=pd.Timestamp("2000-01-01"), rule_id="", rule_family="",
                base_severity=float(row["base_severity"]),
                evidence_support_index=float(row["evidence_support_index"]),
                contextual_harm_index=float(row["contextual_harm_index"]),
                evidence_completeness_index=float(row["evidence_completeness_index"]),
                certainty_class=str(row["certainty_class"]),
                current_creatinine=None, hours_since_current_creatinine=None,
                acute_renal_deterioration=False,
                acute_renal_deterioration_basis="", crrt_active=False,
                transplant_patient=True, transplant_date_source="",
                transplant_phase="", days_since_transplant=None, protocol_fit="",
                monitoring_status="", transition_status="", episode_id="",
                episode_occurrence_index=int(row["episode_occurrence_index"]),
                prior_lifetime_occurrences=0, prior_admission_occurrences=0,
                hours_since_prior_same_signal=None,
                material_change_since_prior=self._bool(row["material_change_since_prior"]),
                repeat_status=str(row["repeat_status"]), note_context_usable=False,
                management_status=str(row["management_status"]),
                safety_class=str(row["safety_class"]),
                admissible_actions=parse_actions(row["admissible_actions"]),
                missingness_flags=(), context_reason="",
            )
            record = self.decide(state).as_dict()
            record["_signature_id"] = int(row["_signature_id"])
            for duplicate_field in ("signal_id", "admissible_actions", "safety_class"):
                record.pop(duplicate_field, None)
            decisions.append(record)
        mapped = working.merge(pd.DataFrame(decisions), on="_signature_id", how="left", validate="many_to_one")
        return mapped.drop(columns=["_signature_id"])
