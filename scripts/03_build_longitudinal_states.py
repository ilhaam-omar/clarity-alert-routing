from __future__ import annotations

import pandas as pd

from common import atomic_csv, banner, parser, progress_printer
from clarity_v3.config import load_state_config
from clarity_v3.project import ensure_output_dirs, load_project_paths
from clarity_v3.reference import ReferenceStore
from clarity_v3.state import ContextStateBuilder


def main() -> None:
    args = parser("Build structured, phase-aware longitudinal signal states").parse_args()
    banner("STAGE 03 - STRUCTURED PHASE-AWARE LONGITUDINAL SIGNAL EPISODES")
    paths = load_project_paths(args.paths)
    ensure_output_dirs(paths)

    if paths.note_context_mode != "disabled":
        raise ValueError("Stage 03 in MIMIC Phase 1 requires note_context_mode='disabled'")
    if paths.note_results_file is not None or paths.discharge_notes_file is not None:
        raise ValueError("Stage 03 in MIMIC Phase 1 must not receive clinical-note inputs")
    if not paths.signals_file.exists() or not paths.orders_file.exists():
        raise FileNotFoundError("Run Stages 01 and 02 first")

    signals = pd.read_csv(paths.signals_file, low_memory=False)
    orders = pd.read_csv(paths.orders_file, low_memory=False)
    reference = ReferenceStore(paths.reference_dir)
    builder = ContextStateBuilder(
        reference,
        config=load_state_config(paths.root / "config" / "state_builder.json"),
        note_context_mode="disabled",
    )

    print("[DESIGN] Structured MIMIC context only.")
    states = builder.build(
        signals,
        orders,
        progress_callback_orders=progress_printer("Phase 1/2 active-order state"),
        progress_callback_signals=progress_printer("Phase 2/2 contextual states"),
    )
    atomic_csv(states, paths.states_file)

    print(f"Input signals preserved: {len(states):,}/{len(signals):,}")
    print(f"Distinct exposure episodes: {states['episode_id'].nunique():,}")
    print(f"Repeated unchanged states: {(states['repeat_status'] == 'repeat_unchanged_state').sum():,}")
    print(f"Repeated states with material change: {(states['repeat_status'] == 'repeat_with_material_change').sum():,}")
    print(f"Phase-unknown states: {(states['transplant_phase'] == 'prevalent_phase_unknown').sum():,}")
    print(
        "Phase-supported protocol states: "
        f"{states['protocol_fit'].isin(['phase_specific_match', 'phase_independent_match']).sum():,}"
    )
    print("Signals using clinical-note context: 0 (by Phase-1 design)")
    print("Safety-class distribution:")
    print(states["safety_class"].value_counts().to_string())
    print(f"Saved private state output: {paths.states_file}")
    print("\nSTAGE 03: COMPLETE - PASS")


if __name__ == "__main__":
    main()