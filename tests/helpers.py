from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from clarity_v3.config import load_state_config, load_uncertainty_config
from clarity_v3.mimic import build_order_timeline, detect_signals_from_orders
from clarity_v3.notes import NoteFactStore
from clarity_v3.reference import ReferenceStore
from clarity_v3.state import ContextStateBuilder
from clarity_v3.uncertainty import MultiEnvelopeActionPolicy, sample_design

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def demo_frames():
    reference = ReferenceStore(ROOT / "reference")
    orders = build_order_timeline(
        ROOT / "demo_data/kidney_transplant_cohort.csv",
        ROOT / "demo_data/hosp/patients.csv",
        ROOT / "demo_data/hosp/admissions.csv",
        ROOT / "demo_data/hosp/prescriptions.csv",
        ROOT / "demo_data/hosp/labevents.csv",
        ROOT / "demo_data/hosp/procedures_icd.csv",
        ROOT / "demo_data/hosp/transfers.csv",
        ROOT / "demo_data/hosp/services.csv",
        ROOT / "demo_data/icu/procedureevents.csv",
        reference,
        expected_patients=4,
    )
    signals = detect_signals_from_orders(orders, reference)
    states = ContextStateBuilder(
        reference,
        note_store=NoteFactStore([]),
        config=load_state_config(ROOT / "config/state_builder.json"),
    ).build(signals, orders)
    envelopes, action_config = load_uncertainty_config(ROOT / "config/uncertainty_design.json")
    design = sample_design(envelopes)
    decisions = MultiEnvelopeActionPolicy(
        design,
        {item.name: item.regret_quantile for item in envelopes},
        action_config,
    ).decide_dataframe(states)
    return reference, orders, signals, states, design, decisions
