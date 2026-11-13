from __future__ import annotations

import pandas as pd

from common import atomic_csv, banner, parser
from clarity_v3.bundles import build_order_bundles
from clarity_v3.project import ensure_output_dirs, load_project_paths


def main() -> None:
    args = parser("Consolidate signal actions into workflow bundles").parse_args()
    banner("STAGE 05 - ORDER-BUNDLE CONSOLIDATION")
    paths = load_project_paths(args.paths)
    ensure_output_dirs(paths)
    if not paths.decisions_file.exists():
        raise FileNotFoundError("Run Stage 04 first")
    decisions = pd.read_csv(paths.decisions_file, low_memory=False)
    bundles = build_order_bundles(decisions)
    atomic_csv(bundles, paths.bundles_file)
    print(f"Signal rows: {len(decisions):,}")
    print(f"Order bundles: {len(bundles):,}")
    print(f"Signals consolidated per bundle, median: {bundles['signal_count'].median():.1f}")
    print(f"Bundles with >1 signal: {(bundles['signal_count'] > 1).sum():,}")
    print("Pre-capacity bundle action distribution:")
    print(bundles["pre_capacity_action"].value_counts().to_string())
    print(f"Calibration-sensitive bundles: {bundles['calibration_sensitive'].astype(bool).sum():,}")
    print(f"Saved private bundle output: {paths.bundles_file}")
    print("\nSTAGE 05: COMPLETE - PASS")


if __name__ == "__main__":
    main()
