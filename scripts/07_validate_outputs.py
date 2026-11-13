from __future__ import annotations

import pandas as pd

from common import atomic_json, banner, parser
from clarity_v3.project import ensure_output_dirs, load_project_paths


def _bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin({"1","true","yes","y","t"})


def main() -> None:
    args = parser("Validate Phase-1 denominators and safety invariants").parse_args()
    banner("STAGE 07 - DENOMINATOR, SAFETY, AND CAPACITY VALIDATION")
    paths = load_project_paths(args.paths)
    ensure_output_dirs(paths)
    orders = pd.read_csv(paths.orders_file, low_memory=False)
    signals = pd.read_csv(paths.signals_file, low_memory=False)
    states = pd.read_csv(paths.states_file, low_memory=False)
    decisions = pd.read_csv(paths.decisions_file, low_memory=False)
    bundles = pd.read_csv(paths.bundles_file, low_memory=False)
    final = pd.read_csv(paths.final_bundles_file, low_memory=False)
    checks = {
        "signal_state_preservation": len(signals) == len(states),
        "state_action_preservation": len(states) == len(decisions),
        "signal_ids_unique": not signals["signal_id"].duplicated().any(),
        "state_ids_unique": not states["signal_id"].duplicated().any(),
        "bundle_ids_unique": not bundles["bundle_id"].duplicated().any(),
        "all_signal_bundles_present": set(signals["bundle_id"].astype(str)) == set(bundles["bundle_id"].astype(str)),
        "one_final_row_per_bundle_per_scenario": not final.duplicated(["bundle_id", "scenario"]).any(),
    }
    protected_alert_bundles = set(decisions.loc[_bool(decisions["protected_alert"]), "bundle_id"].astype(str))
    protected_review_bundles = set(decisions.loc[_bool(decisions["protected_review"]), "bundle_id"].astype(str))
    for scenario, group in final.groupby("scenario"):
        protected_actions = group[group["bundle_id"].astype(str).isin(protected_alert_bundles)]["final_action"]
        review_actions = group[group["bundle_id"].astype(str).isin(protected_review_bundles)]["final_action"]
        checks[f"{scenario}_protected_alerts_remain_alert"] = protected_actions.eq("alert").all()
        checks[f"{scenario}_protected_reviews_remain_resource_actions"] = review_actions.isin({"alert", "escalate"}).all()
        checks[f"{scenario}_no_duplicate_bundle"] = not group["bundle_id"].duplicated().any()
    standard_suppress = decisions[decisions["pre_capacity_action"].eq("suppress")]
    checks["suppression_only_standard_safety"] = standard_suppress["safety_class"].eq("standard").all()
    checks["suppression_not_material_change"] = ~_bool(standard_suppress["material_change_since_prior"]).any()
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    if not all(checks.values()):
        raise AssertionError("One or more output validation checks failed")
    summary = {
        "orders": len(orders),
        "signals": len(signals),
        "states": len(states),
        "action_rows": len(decisions),
        "bundles": len(bundles),
        "scenarios": int(final["scenario"].nunique()),
        "checks": checks,
    }
    atomic_json(summary, paths.evidence_dir / "technical_validation.json")
    print("\nSTAGE 07: COMPLETE - PASS")


if __name__ == "__main__":
    main()
