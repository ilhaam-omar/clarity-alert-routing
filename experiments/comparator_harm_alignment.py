"""Do the routing differences between CLARITY and a severity-tiered comparator target risk?

Links the per-bundle routes of the snapshot severity-evidence comparator (Stage 10B) with the
per-bundle laboratory markers (lab_harm_markers.py) and reports marker frequencies for every
route transition, without capacity limits. Only aggregate counts are written; counts from 1 to 10
are masked. Markers reflect care actually received, not effects of any routing policy.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

MASK = 11
MARKERS = ["aki", "hyperkalaemia", "hypoglycaemia", "high_inr", "high_tacrolimus", "any_marker"]

def masked(n):
    n = int(n)
    return n if n == 0 or n >= MASK else f"<{MASK}"

def find_cols(frame):
    cols = {c.lower(): c for c in frame.columns}
    pick = lambda *names: next((cols[n] for n in names if n in cols), None)
    return pick("bundle_id"), pick("scenario"), pick("snapshot_action", "snapshot_final_action", "snapshot_route"), \
        pick("clarity_action", "clarity_final_action", "clarity_route")

def summarise(merged):
    rows = []
    for (snap, clar), g in merged.groupby(["snapshot_action", "clarity_action"]):
        n = len(g)
        row = {"snapshot_route": snap, "clarity_route": clar, "bundles": masked(n)}
        for m in MARKERS:
            ev = int(g[m].sum())
            ok = n >= MASK and (ev == 0 or ev >= MASK)
            row[f"{m}_percent"] = round(100 * ev / n, 2) if ok else None
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--markers", type=Path, required=True, help="lab_harm_markers_*/private/bundle_markers.csv")
    a = ap.parse_args()
    comp_path = a.project / "outputs_corrected/private/10b_snapshot_vs_clarity_bundle_comparison.csv"
    comp = pd.read_csv(comp_path, dtype=str)
    bid, scen, snap, clar = find_cols(comp)
    if not all((bid, snap, clar)):
        raise SystemExit("STOPPED: expected columns not found. Column names: " + ", ".join(comp.columns))
    if scen:
        comp = comp[comp[scen].eq("unconstrained_reference")]
    comp = comp.rename(columns={bid: "bundle_id", snap: "snapshot_action", clar: "clarity_action"})[["bundle_id", "snapshot_action", "clarity_action"]]
    mk = pd.read_csv(a.markers, dtype={"bundle_id": str})
    for m in MARKERS:
        mk[m] = mk[m].astype(str).str.lower().isin(["true", "1"])
    merged = comp.merge(mk[["bundle_id"] + MARKERS], on="bundle_id", how="inner", validate="one_to_one")
    if len(merged) != len(comp):
        raise SystemExit(f"STOPPED: {len(comp) - len(merged)} bundles lacked markers")
    out = a.project / "comparator_harm_alignment" / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    table = summarise(merged)
    table.to_csv(out / "transition_harm_markers.csv", index=False)
    json.dump({"bundles": len(merged), "scenario": "unconstrained_reference", "small_cell_rule": "counts 1-10 masked",
               "interpretation": "markers reflect care received; not effects of routing"}, open(out / "manifest.json", "w"), indent=2)
    print(table.to_string(index=False))
    print("COMPLETE. Aggregate outputs:", out)

if __name__ == "__main__":
    main()
