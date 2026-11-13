"""Feasibility frontier: at which capacities can the safety rules be honoured?

Replaces the three capacity scenarios with the full relationship. For every interruption
allowance F (per 100 order bundles) and number of review slots K, the safety rules can be
honoured if and only if (main-article Eq. 8)
    M <= F_abs   and   M + max(0, P - K) <= F_abs,
where M is the number of bundles that must interrupt and P the further protected bundles
that need an interruptive alert or pharmacist review. Reported:
  - the allowance at which the rules become feasible with unlimited review (M only)
  - the allowance needed at each review capacity, and the review slots needed at each allowance
  - the three paper scenarios placed on the frontier
  - a figure of the frontier (frontier.png / frontier.pdf)
M and P come from the safety rules, so the frontier does not depend on routing, ranking or weights.

Usage
  python experiments/feasibility_frontier.py --project . --outputs outputs_corrected
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def tf(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "t"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--outputs", default="outputs_corrected")
    a = ap.parse_args()
    project = a.project.expanduser().resolve()
    private = project / a.outputs / "private"
    bundles = pd.read_csv(private / "05_order_bundles.csv", low_memory=False)
    final = pd.read_csv(private / "06_final_bundle_decisions.csv", low_memory=False)
    N = int(final["total_order_bundles"].iloc[0])
    pa, pr = tf(bundles["protected_alert"]), tf(bundles["protected_review"])
    M = int(pa.sum())
    P = int((pr & ~pa).sum())
    affected = len(bundles)

    out = project / ("feasibility_frontier_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True)

    def per100(x):
        return 100.0 * x / N

    rows = []
    for share in np.round(np.arange(0, 1.0001, 0.05), 2):
        K = int(np.floor(share * P))
        need = M + max(0, P - K)
        rows.append({"review_slots_share_of_protected_review_demand": share, "review_slots": K,
                     "interruptions_required": need,
                     "allowance_needed_per_100_order_bundles": round(per100(need), 3),
                     "allowance_needed_per_100_affected_bundles": round(100 * need / affected, 3)})
    by_slots = pd.DataFrame(rows)

    rows = []
    for f in np.round(np.arange(0.25, 10.0001, 0.25), 2):
        F = int(np.floor(f * N / 100))
        feasible_any = M <= F
        K_needed = max(0, M + P - F) if feasible_any else None
        rows.append({"allowance_per_100_order_bundles": f, "interruption_allowance": F,
                     "feasible_with_unlimited_review": feasible_any,
                     "review_slots_needed": K_needed,
                     "review_slots_needed_share_of_protected_review_demand":
                         round(K_needed / P, 3) if (K_needed is not None and P) else None})
    by_allowance = pd.DataFrame(rows)

    scen = []
    for sc in ("low_capacity", "medium_capacity", "high_capacity"):
        part = final[final["scenario"].eq(sc)]
        if not len(part):
            continue
        F = int(part["frontline_slots"].iloc[0]); K = int(part["expert_slots"].iloc[0])
        need = M + max(0, P - K)
        scen.append({"scenario": sc, "allowance": F, "allowance_per_100": round(per100(F), 3), "review_slots": K,
                     "interruptions_required": need, "feasible": need <= F,
                     "allowance_needed_at_these_slots_per_100": round(per100(need), 3),
                     "slots_needed_at_this_allowance": max(0, M + P - F) if M <= F else None})
    scenarios = pd.DataFrame(scen)

    key = {"order_bundles": N, "affected_bundles": affected, "must_interrupt_M": M, "protected_review_P": P,
           "allowance_threshold_unlimited_review_per_100": round(per100(M), 3),
           "allowance_threshold_no_review_per_100": round(per100(M + P), 3),
           "interpretation": "Feasible region: allowance >= M and allowance + review slots >= M + P. "
                             "Independent of routing method, ranking and weights."}
    by_slots.to_csv(out / "frontier_by_review_slots.csv", index=False)
    by_allowance.to_csv(out / "frontier_by_allowance.csv", index=False)
    scenarios.to_csv(out / "frontier_scenarios.csv", index=False)
    (out / "frontier_key_numbers.json").write_text(json.dumps(key, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        k = np.linspace(0, P, 400)
        ax.plot(k, per100(M + np.maximum(0, P - k)), color="#1f4e79", lw=2, label="Minimum allowance (Eq. 7)")
        ax.fill_between(k, per100(M + np.maximum(0, P - k)), per100(M + P) * 1.25, color="#1f4e79", alpha=0.12,
                        label="Feasible")
        ax.axhline(per100(M), color="#7f7f7f", ls="--", lw=1, label="Must-interrupt floor")
        for _, r in scenarios.iterrows():
            ax.plot(r["review_slots"], r["allowance_per_100"], "o", color="#b03a2e" if not r["feasible"] else "#2b7a4b")
            ax.annotate(r["scenario"].split("_")[0], (r["review_slots"], r["allowance_per_100"]),
                        textcoords="offset points", xytext=(6, -10), fontsize=8)
        ax.set_xlabel("Review slots (order bundles)")
        ax.set_ylabel("Interruption allowance per 100 order bundles")
        ax.set_xlim(0, P)
        ax.set_ylim(0, per100(M + P) * 1.25)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        for ext in ("png", "pdf"):
            fig.savefig(out / f"frontier.{ext}", dpi=300)
        plt.close(fig)
    except Exception as e:  # figure is optional
        print("Figure skipped:", e)

    print(json.dumps(key, indent=2))
    print(scenarios.to_string(index=False))
    print("Saved:", out)


if __name__ == "__main__":
    main()
