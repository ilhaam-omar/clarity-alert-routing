from __future__ import annotations

import pandas as pd

from common import atomic_csv, banner, parser
from clarity_v3.config import load_uncertainty_config
from clarity_v3.project import ensure_output_dirs, load_project_paths
from clarity_v3.uncertainty import MultiEnvelopeActionPolicy, sample_design


def main() -> None:
    args = parser("Identify cross-envelope robust action sets").parse_args()
    banner("STAGE 04 - MULTI-ENVELOPE ACTION-SET IDENTIFICATION")
    paths = load_project_paths(args.paths)
    ensure_output_dirs(paths)
    if not paths.states_file.exists():
        raise FileNotFoundError("Run Stage 03 first")
    states = pd.read_csv(paths.states_file, low_memory=False)
    envelopes, action_config = load_uncertainty_config(
        paths.root / "config" / "uncertainty_design.json"
    )
    design = sample_design(envelopes)
    atomic_csv(design, paths.parameter_design_file)
    policy = MultiEnvelopeActionPolicy(
        design,
        {envelope.name: envelope.regret_quantile for envelope in envelopes},
        action_config,
    )
    decisions = policy.decide_dataframe(states)
    if len(decisions) != len(states):
        raise ValueError("Stage 04 did not preserve the signal denominator")
    atomic_csv(decisions, paths.decisions_file)
    print(f"Signals evaluated: {len(decisions):,}")
    print(f"Uncertainty envelopes: {len(envelopes):,}")
    print(f"Design points: {len(design):,}")
    print("Pre-capacity action distribution:")
    print(decisions["pre_capacity_action"].value_counts().to_string())
    print(f"Cross-envelope consensus actions: {decisions['consensus_action'].notna().sum():,}")
    print(f"Calibration-sensitive signals: {decisions['calibration_sensitive'].astype(bool).sum():,}")
    print(f"Protected immediate alerts: {decisions['protected_alert'].astype(bool).sum():,}")
    print(f"Protected review-or-alert signals: {decisions['protected_review'].astype(bool).sum():,}")
    print(
        "Interpretation: design points are policy stress tests, not calibrated probabilities; "
        "action support is a design-space fraction."
    )
    print(f"Saved uncertainty design: {paths.parameter_design_file}")
    print(f"Saved private action-set output: {paths.decisions_file}")
    print("\nSTAGE 04: COMPLETE - PASS")


if __name__ == "__main__":
    main()
