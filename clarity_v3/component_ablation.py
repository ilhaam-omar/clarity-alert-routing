from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
from unittest.mock import patch

import pandas as pd

from .models import ActiveMedication
from .reference import ReferenceStore
from .state import (
    ContextStateBuilder,
    ProgressCallback,
    StateBuilderConfig,
    _build_active_snapshots,
)
from .transplant import TransplantContextEngine


@dataclass(frozen=True)
class AblationSpec:
    name: str
    label: str
    remove_episode_memory: bool = False
    remove_acute_renal_context: bool = False
    remove_transplant_protocol_context: bool = False
    remove_transition_context: bool = False


FULL_CONTEXT = AblationSpec(
    name="full_context_reference",
    label="Full structured longitudinal context",
)

ABLATIONS = (
    AblationSpec(
        name="without_episode_memory",
        label="Medication-exposure episode memory removed",
        remove_episode_memory=True,
    ),
    AblationSpec(
        name="without_acute_renal_context",
        label="Acute renal trajectory removed; current creatinine retained",
        remove_acute_renal_context=True,
    ),
    AblationSpec(
        name="without_transplant_protocol_context",
        label="Transplant protocol and structured monitoring interpretation removed",
        remove_transplant_protocol_context=True,
    ),
    AblationSpec(
        name="without_taper_route_transition_context",
        label="Steroid-taper and route-transition interpretation removed",
        remove_transition_context=True,
    ),
)


class _ProtocolDisabledReference:
    """Delegate all reference data except transplant override matching."""

    def __init__(self, base: ReferenceStore) -> None:
        self._base = base

    def __getattr__(self, name: str):
        return getattr(self._base, name)

    def transplant_override(self, rule_id: str):
        return None


class _TransitionDisabledEngine(TransplantContextEngine):
    """Retain protocol interpretation while disabling taper/route transitions."""

    def assess(self, signal, order, active_meds, **kwargs):
        # Active medications are used by TransplantContextEngine only for the
        # taper and route-transition checks. Protocol matching remains intact.
        return super().assess(signal, order, (), **kwargs)


def build_reusable_active_snapshots(
    orders: pd.DataFrame,
    signals: pd.DataFrame,
    progress_callback: Optional[ProgressCallback] = None,
    progress_every: int = 5000,
) -> Dict[str, tuple[ActiveMedication, ...]]:
    return _build_active_snapshots(
        orders,
        set(signals["order_id"].astype(str)),
        progress_callback=progress_callback,
        progress_every=progress_every,
    )


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _signals_without_episode_memory(signals: pd.DataFrame) -> pd.DataFrame:
    """Make each signal's episode key unique without altering its clinical rule.

    The production state builder derives episode keys from different fields for
    each rule family. This analysis-only transform changes only those key fields;
    rule identifiers, severity, evidence, and hazard content remain unchanged.
    """

    result = signals.copy()
    suffix = "|ABLATION:" + result["signal_id"].astype(str)
    family = result["rule_family"].astype(str)

    ddi = family.eq("drug_drug_interaction")
    if "drug_pair" not in result.columns:
        result["drug_pair"] = ""
    result.loc[ddi, "drug_pair"] = _clean_text(result.loc[ddi, "drug_pair"]) + suffix[ddi]

    duplicate = family.eq("duplicate_therapy")
    if "therapy_class" not in result.columns:
        result["therapy_class"] = ""
    result.loc[duplicate, "therapy_class"] = (
        _clean_text(result.loc[duplicate, "therapy_class"]) + suffix[duplicate]
    )

    other = ~(ddi | duplicate)
    result.loc[other, "new_ingredient_key"] = (
        _clean_text(result.loc[other, "new_ingredient_key"]) + suffix[other]
    )
    return result


def build_component_variant(
    spec: AblationSpec,
    signals: pd.DataFrame,
    orders: pd.DataFrame,
    reference: ReferenceStore,
    config: StateBuilderConfig,
    active_snapshots: Optional[Dict[str, tuple[ActiveMedication, ...]]] = None,
    progress_callback_signals: Optional[ProgressCallback] = None,
    progress_every: int = 5000,
) -> pd.DataFrame:
    variant_signals = (
        _signals_without_episode_memory(signals)
        if spec.remove_episode_memory
        else signals
    )
    variant_orders = orders
    if spec.remove_acute_renal_context:
        variant_orders = orders.copy()
        variant_orders["acute_renal_deterioration"] = False
        if "acute_deterioration_basis" in variant_orders.columns:
            variant_orders["acute_deterioration_basis"] = "removed_by_ablation"

    variant_reference = (
        _ProtocolDisabledReference(reference)
        if spec.remove_transplant_protocol_context
        else reference
    )
    builder = ContextStateBuilder(
        variant_reference,  # type: ignore[arg-type]
        config=config,
        note_context_mode="disabled",
    )
    if spec.remove_transition_context:
        builder.transplant = _TransitionDisabledEngine(variant_reference)  # type: ignore[arg-type]

    if active_snapshots is None:
        return builder.build(
            variant_signals,
            variant_orders,
            progress_callback_signals=progress_callback_signals,
            progress_every=progress_every,
        )

    required_order_ids = set(variant_signals["order_id"].astype(str))
    missing = required_order_ids - set(active_snapshots)
    if missing:
        raise ValueError(
            f"Reusable active snapshots are missing {len(missing)} required order(s)"
        )
    # Keep the production state builder unchanged. Stage 09R injects the
    # deterministic cache only inside this analysis call.
    with patch("clarity_v3.state._build_active_snapshots", return_value=active_snapshots):
        return builder.build(
            variant_signals,
            variant_orders,
            progress_callback_signals=progress_callback_signals,
            progress_every=progress_every,
        )
