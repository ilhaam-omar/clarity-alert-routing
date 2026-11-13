"""Convergence test: is the priority limit a property of the problem or of Monte Carlo noise?

Changing only the sampling seed replaced about half of the medium-capacity review set. If the
1,024 combinations per envelope were too few, the priority scores would not have converged and
the instability would shrink as the number of combinations grows. This script recomputes the
review priority score with N combinations per envelope (default 1,024, 4,096 and 16,384), each
under three sampling seeds, and reports for each N:
  - review-set replacement between seeds at each capacity (same tie-breaking keys)
  - rank correlation of the priority scores between seeds
  - the robust core across the three seeds (exact rank bounds, every tie rule)
  - the largest change in any candidate's score between seeds
Interpretation: replacement that falls towards zero as N grows means sampling noise; replacement
that stays high means the candidates are genuinely near-tied, so the priority limit is real.

Only concerns in bundles that can compete for review are recomputed, so the run is much faster
than the full pipeline. Expect roughly 30-60 minutes in total for the three sizes.

Usage
  python experiments/convergence_test.py --project . --outputs outputs_corrected [--points 1024,4096,16384]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

KIT = Path(__file__).resolve().parent
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))
import allocation_robustness as ar  # noqa: E402
import parameter_sensitivity_allocation as psa  # noqa: E402
import robust_tier as rt  # noqa: E402

SCENARIOS = ("low_capacity", "medium_capacity", "high_capacity")
SEED_OFFSETS = (0, 101, 202)
TIE_SEEDS = 20


def tf(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "t"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--outputs", default="outputs_corrected")
    ap.add_argument("--points", default="1024,4096,16384")
    a = ap.parse_args()
    project = a.project.expanduser().resolve()
    sys.path.insert(0, str(project))
    from clarity_v3.config import load_uncertainty_config
    from clarity_v3.uncertainty import MultiEnvelopeActionPolicy, sample_design

    private = project / a.outputs / "private"
    states = pd.read_csv(private / "03_longitudinal_states.csv", low_memory=False)
    bundles = pd.read_csv(private / "05_order_bundles.csv", low_memory=False)
    final = pd.read_csv(private / "06_final_bundle_decisions.csv", low_memory=False)
    bundles["bundle_id"] = bundles["bundle_id"].astype(str)
    pool = bundles[bundles["pre_capacity_action"].astype(str).eq("escalate") & ~tf(bundles["protected_alert"])]
    pool_ids = set(pool["bundle_id"])
    slots = {sc: int(final.loc[final["scenario"].eq(sc), "expert_slots"].iloc[0])
             for sc in SCENARIOS if final["scenario"].eq(sc).any()}
    sub = states[states["bundle_id"].astype(str).isin(pool_ids)].copy()
    print(f"Review candidates: {len(pool_ids):,}; concerns recomputed: {len(sub):,}", flush=True)

    envelopes, action_config = load_uncertainty_config(project / "config" / "uncertainty_design.json")
    q = {e.name: e.regret_quantile for e in envelopes}
    frame = pd.DataFrame({"bundle_id": sorted(pool_ids)})
    base = ar.base_hashes(frame["bundle_id"])
    keys = [ar.tie_keys(base, s) for s in range(TIE_SEEDS)]

    out = project / ("convergence_test_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True)
    rows = []
    for n in [int(x) for x in a.points.split(",") if x.strip()]:
        names = []
        for off in SEED_OFFSETS:
            t0 = time.time()
            envs = [replace(e, n_points=n, seed=e.seed + off) for e in envelopes]
            decided = MultiEnvelopeActionPolicy(sample_design(envs), q, action_config).decide_dataframe(sub)
            col = f"n{n}_s{off}"
            frame[col] = frame["bundle_id"].map(psa.bundle_delta(decided)).fillna(0.0)
            names.append(col)
            print(f"  N={n:,} seed offset {off}: {time.time() - t0:.0f}s", flush=True)
        row = {"points_per_envelope": n, "seeds": len(names)}
        pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
        row["score_rank_correlation_mean"] = round(float(np.mean([psa.spearman(frame[x].to_numpy(), frame[y].to_numpy())
                                                                  for x, y in pairs])), 4)
        row["largest_score_change"] = round(float(max((frame[x] - frame[y]).abs().max() for x, y in pairs)), 5)
        for sc, K in slots.items():
            reps = []
            for x, y in pairs:
                for kk in keys:
                    reps.append(psa.compare_sets(psa.select(frame, x, kk, K), psa.select(frame, y, kk, K))[1])
            row[f"{sc}_replacement_between_seeds_percent"] = round(float(np.mean(reps)), 2)
            best, worst = rt.set_bounds(frame, names)
            row[f"{sc}_robust_core_across_seeds"] = int((worst <= K).sum())
            row[f"{sc}_robust_core_percent_of_slots"] = round(100 * int((worst <= K).sum()) / K, 1)
            b1, w1 = rt.rank_bounds(frame[names[0]].to_numpy())
            row[f"{sc}_decided_by_score_single_seed"] = int((w1 <= K).sum())
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / "convergence_summary.csv", index=False)   # saved after each size
    table = pd.DataFrame(rows)
    (out / "manifest.json").write_text(json.dumps({"points": a.points, "seed_offsets": SEED_OFFSETS,
                                                   "tie_breaking_seeds": TIE_SEEDS, "slots": slots}, indent=2))
    pd.set_option("display.width", 250)
    print(table.set_index("points_per_envelope").T.to_string())
    print("Saved:", out)


if __name__ == "__main__":
    main()
