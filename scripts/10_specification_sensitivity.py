from __future__ import annotations

from dataclasses import replace

import pandas as pd

from common import atomic_csv, banner, parser
from clarity_v3.config import load_uncertainty_config
from clarity_v3.project import ensure_output_dirs, load_project_paths
from clarity_v3.uncertainty import ActionSetConfig, MultiEnvelopeActionPolicy, sample_design


def main() -> None:
    args = parser("Test action-set stability to numerical and policy specifications").parse_args()
    banner("STAGE 10 - ACTION-SET SPECIFICATION SENSITIVITY")
    paths = load_project_paths(args.paths)
    ensure_output_dirs(paths)
    states = pd.read_csv(paths.states_file, low_memory=False)
    primary = pd.read_csv(paths.decisions_file, usecols=["signal_id", "pre_capacity_action"]).rename(
        columns={"pre_capacity_action": "primary_action"}
    )
    envelopes, action_config = load_uncertainty_config(paths.root / "config" / "uncertainty_design.json")
    specifications: list[tuple[str, list, ActionSetConfig]] = []
    specifications.append(("half_design_points", [replace(e, n_points=max(32, e.n_points // 2)) for e in envelopes], action_config))
    specifications.append(("double_design_points", [replace(e, n_points=e.n_points * 2) for e in envelopes], action_config))
    specifications.append(("alternate_seeds", [replace(e, seed=e.seed + 101) for e in envelopes], action_config))
    specifications.append(("regret_quantile_0.90", [replace(e, regret_quantile=0.90) for e in envelopes], action_config))
    specifications.append(("regret_quantile_0.975", [replace(e, regret_quantile=0.975) for e in envelopes], action_config))
    specifications.append(("support_threshold_0.60", envelopes, replace(action_config, consensus_support_threshold=0.60)))
    specifications.append(("support_threshold_0.80", envelopes, replace(action_config, consensus_support_threshold=0.80)))
    specifications.append(("suppress_threshold_0.90", envelopes, replace(action_config, suppress_support_threshold=0.90)))

    rows: list[dict] = []
    for name, envs, action in specifications:
        design = sample_design(envs)
        policy = MultiEnvelopeActionPolicy(
            design,
            {item.name: item.regret_quantile for item in envs},
            action,
        )
        result = policy.decide_dataframe(states)[["signal_id", "pre_capacity_action", "calibration_sensitive"]].rename(
            columns={"pre_capacity_action": "alternative_action"}
        )
        compared = primary.merge(result, on="signal_id", validate="one_to_one")
        agreement = compared["primary_action"].eq(compared["alternative_action"])
        rows.append({
            "specification": name,
            "design_points": len(design),
            "action_agreement_percent": 100.0 * float(agreement.mean()),
            "calibration_sensitive_percent": 100.0 * float(result["calibration_sensitive"].astype(bool).mean()),
            "protected_safety_violations": int(
                ((states["safety_class"].eq("must_interrupt")) & ~result["alternative_action"].eq("alert")).sum()
            ),
        })
    frame = pd.DataFrame(rows)
    atomic_csv(frame, paths.evidence_dir / "specification_sensitivity.csv")
    print(frame.to_string(index=False))
    if frame["protected_safety_violations"].sum() != 0:
        raise AssertionError("A specification violated must-interrupt protection")
    print("\nSTAGE 10: COMPLETE - PASS")


if __name__ == "__main__":
    main()
