from __future__ import annotations

import pandas as pd

from common import atomic_csv, banner, parser, progress_printer
from clarity_v3.mimic import detect_signals_from_orders
from clarity_v3.project import ensure_output_dirs, load_project_paths
from clarity_v3.reference import ReferenceStore


def main() -> None:
    args = parser("Run shared deterministic medication-hazard detector").parse_args()
    banner("STAGE 02 - SHARED DETERMINISTIC HAZARD DETECTION")
    paths = load_project_paths(args.paths)
    ensure_output_dirs(paths)
    if not paths.orders_file.exists():
        raise FileNotFoundError("Run Stage 01 first")
    orders = pd.read_csv(paths.orders_file, low_memory=False)
    reference = ReferenceStore(paths.reference_dir)
    signals = detect_signals_from_orders(
        orders,
        reference,
        progress_callback=progress_printer("Hazard detection"),
    )
    if signals.empty:
        raise ValueError("No medication-safety hazards were detected")
    atomic_csv(signals, paths.signals_file)
    print(f"Orders evaluated: {len(orders):,}")
    print(f"Signal events: {len(signals):,}")
    print(f"Unique order bundles with >=1 signal: {signals['bundle_id'].nunique():,}")
    print("Rule-family distribution:")
    print(signals["rule_family"].value_counts().to_string())
    print(f"High-severity signals: {(signals['base_severity'] >= 8).sum():,}")
    print(
        "Comparator B0: every signal event interrupts; Comparator B1: every order bundle "
        "with >=1 signal produces one interruption. Both are simulated upper bounds, not exact Cerner behavior."
    )
    print(f"Saved private signal output: {paths.signals_file}")
    print("\nSTAGE 02: COMPLETE - PASS")


if __name__ == "__main__":
    main()
