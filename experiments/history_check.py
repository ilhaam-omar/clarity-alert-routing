"""Does the patient-history result (main-article Table 7) depend on the weight design?

Recomputes CLARITY's unconstrained bundle routes under the primary design and under the
alternative weight designs of weight_grounding.py, then rebuilds Table 7: every route
transition from the snapshot severity-evidence comparator (Stage 10B, unchanged) to CLARITY,
with the share of bundles followed by any laboratory harm marker. Small cells (1-10) are masked.

Key contrasts reported for each design (as in main-article Section 5.5)
  kept_visible      snapshot suppression -> CLARITY passive notification
  both_suppressed   snapshot suppression -> CLARITY suppression
  moved_to_review   snapshot interruption -> CLARITY review
  both_interrupt    snapshot interruption -> CLARITY interruption

Usage
  python experiments/history_check.py --project . --outputs outputs_corrected \
      --markers lab_harm_markers_<date>/private/bundle_markers.csv \
      [--designs primary,declared_ranges,definitional_domain]

Markers reflect care actually received, not effects of any routing policy.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KIT = Path(__file__).resolve().parent
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))
import weight_grounding as wg  # noqa: E402
from comparator_harm_alignment import MARKERS, find_cols  # noqa: E402

MASK = 11
BOOT = 2000


def masked(n: int):
    return n if n == 0 or n >= MASK else f"<{MASK}"


def boot_ci(frame: pd.DataFrame, mask: pd.Series, seed: int = 7):
    """Patient-cluster bootstrap 95% interval of the marker share in a group."""
    g = frame.loc[mask, ["subject_id", "any_marker"]]
    if len(g) < MASK:
        return (np.nan, np.nan)
    per = g.groupby("subject_id")["any_marker"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(BOOT, len(per)))
    s, c = per["sum"].to_numpy()[idx].sum(1), per["count"].to_numpy()[idx].sum(1)
    v = 100 * s / np.maximum(c, 1)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--outputs", default="outputs_corrected")
    ap.add_argument("--markers", type=Path, required=True)
    ap.add_argument("--designs", default="primary,declared_ranges,definitional_domain")
    a = ap.parse_args()
    project = a.project.expanduser().resolve()
    sys.path.insert(0, str(project))
    from clarity_v3.config import load_uncertainty_config
    from clarity_v3.uncertainty import MultiEnvelopeActionPolicy, sample_design

    private = project / a.outputs / "private"
    states = pd.read_csv(private / "03_longitudinal_states.csv", low_memory=False)
    comp = pd.read_csv(private / "10b_snapshot_vs_clarity_bundle_comparison.csv", dtype=str)
    bid, scen, snap, _ = find_cols(comp)
    if not (bid and snap):
        raise SystemExit("STOPPED: snapshot columns not found: " + ", ".join(comp.columns))
    if scen:
        comp = comp[comp[scen].eq("unconstrained_reference")]
    comp = comp.rename(columns={bid: "bundle_id", snap: "snapshot_action"})[["bundle_id", "snapshot_action"]]
    mk = pd.read_csv(a.markers, dtype={"bundle_id": str})
    for m in MARKERS:
        mk[m] = mk[m].astype(str).str.lower().isin(["true", "1"])

    envelopes, action_config = load_uncertainty_config(project / "config" / "uncertainty_design.json")
    wg.N_POINTS = int(envelopes[0].n_points)
    quant = {e.name: e.regret_quantile for e in envelopes}
    out = project / ("history_check_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True)

    tables, contrasts = [], []
    for name in [d.strip() for d in a.designs.split(",") if d.strip()]:
        t0 = time.time()
        design = wg.design_for(name, envelopes, project)
        q = {n: quant.get(n, 0.95) for n in design["envelope"].unique()}
        decided = MultiEnvelopeActionPolicy(design, q, action_config).decide_dataframe(states)
        bundles = wg.bundles_of(decided)[["bundle_id", "subject_id", "pre_capacity_action"]]
        bundles = bundles.rename(columns={"pre_capacity_action": "clarity_action"})
        merged = comp.merge(bundles, on="bundle_id", how="inner", validate="one_to_one")
        merged = merged.merge(mk[["bundle_id"] + MARKERS], on="bundle_id", how="inner", validate="one_to_one")
        if len(merged) != len(comp):
            raise SystemExit(f"STOPPED ({name}): {len(comp) - len(merged)} bundles could not be linked")
        for (s, c), g in merged.groupby(["snapshot_action", "clarity_action"]):
            n = len(g)
            ev = int(g["any_marker"].sum())
            ok = n >= MASK and (ev == 0 or ev >= MASK)
            tables.append({"design": name, "snapshot_route": s, "clarity_route": c, "bundles": masked(n),
                           "any_marker_percent": round(100 * ev / n, 1) if ok else None})
        row = {"design": name, "clarity_suppressed": int(merged["clarity_action"].eq("suppress").sum()),
               "clarity_interrupt": int(merged["clarity_action"].eq("alert").sum()),
               "clarity_review": int(merged["clarity_action"].eq("escalate").sum())}
        pairs = {"kept_visible": ("suppress", "passive"), "both_suppressed": ("suppress", "suppress"),
                 "moved_to_review": ("alert", "escalate"), "both_interrupt": ("alert", "alert")}
        for label, (s, c) in pairs.items():
            m = merged["snapshot_action"].eq(s) & merged["clarity_action"].eq(c)
            n = int(m.sum())
            row[f"{label}_bundles"] = masked(n)
            if n >= MASK:
                lo, hi = boot_ci(merged, m)
                row[f"{label}_marker_percent"] = round(100 * merged.loc[m, "any_marker"].mean(), 1)
                row[f"{label}_ci"] = f"{lo:.1f}-{hi:.1f}"
        contrasts.append(row)
        print(f"{name}: {time.time() - t0:.0f}s", flush=True)

    pd.DataFrame(tables).to_csv(out / "history_transitions_by_design.csv", index=False)
    c = pd.DataFrame(contrasts)
    c.to_csv(out / "history_contrasts_by_design.csv", index=False)
    pd.set_option("display.width", 250)
    print(c.set_index("design").T.to_string())
    print("Saved:", out)


if __name__ == "__main__":
    main()
