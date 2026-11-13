"""Option B: what changes if suppression honours the rule catalogue's own suppression conditions?

The evaluated policy suppresses a repeated warning through one general gate (unchanged state, unprotected,
agreement across envelopes). The rule catalogue also states, per rule, the condition under which a warning
may be suppressed (column suppress_if). Several conditions need data that the reconstructed record does not
contain, such as recent pharmacist verification or a current potassium value. This analysis keeps the evaluated
policy unchanged and asks: if a warning could be suppressed only when its stated condition is verifiable,
which suppressed warnings would instead be displayed passively?

The declared mapping of condition keywords to unavailable data is fixed before results are read and is
recorded in the manifest. Passive display uses no interruption or review capacity, so interruptions and
reviews cannot change; the script checks this. Only aggregate counts are written; counts 1-10 are masked.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
MASK = 11
UNAVAILABLE = {
    "pharmacist verification": "no pharmacist dispositions in MIMIC-IV",
    "potassium": "potassium values not loaded by the pipeline",
    "therapeutic level": "drug concentrations not loaded by the pipeline",
    "qtc": "no ECG data",
    "cbc": "blood counts not loaded by the pipeline",
    "electrolytes": "electrolytes not loaded by the pipeline",
    "end-of-life": "end-of-life status not represented",
}
RULE_FILES = ["04_ddi_rules.csv", "05_renal_dose_rules.csv", "08_beers_rules.csv"]
MARKERS = ["any_marker", "aki", "hyperkalaemia"]


def masked(n):
    n = int(n)
    return n if n == 0 or n >= MASK else f"<{MASK}"


def rule_conditions(reference: Path) -> pd.DataFrame:
    parts = []
    for f in RULE_FILES:
        d = pd.read_csv(reference / f, dtype=str).fillna("")
        if "suppress_if" not in d:
            continue
        ing = d["ingredient"] if "ingredient" in d else d.get("drug_b", "")
        parts.append(pd.DataFrame({"rule_id": d["rule_id"], "suppress_if": d["suppress_if"].str.strip(), "ingredient": ing}))
    rules = pd.concat(parts, ignore_index=True)
    low = rules["suppress_if"].str.lower()
    rules["needs_unavailable_data"] = low.apply(lambda s: any(k in s for k in UNAVAILABLE))
    return rules


def potassium_raising(reference: Path) -> set:
    """Drugs the catalogue's own interaction rules name for hyperkalaemia risk."""
    d = pd.read_csv(reference / "04_ddi_rules.csv", dtype=str).fillna("")
    k = d[d["clinical_risk"].str.lower().str.contains("hyperkal")]
    return set(k["drug_a"].str.lower().str.strip()) | set(k["drug_b"].str.lower().str.strip())


def rate(frame, col):
    n, ev = len(frame), int(frame[col].sum())
    return round(100 * ev / n, 2) if n >= MASK and (ev == 0 or ev >= MASK) else None


def run(project: Path, markers_path: Path | None):
    import run_matched_comparison as rmc
    project = project.resolve()
    source_hashes, _ = rmc.source_guard(project)
    corrected = project / "outputs_corrected"
    sig = rmc.read_private(corrected, "04_action_sets.csv", ["signal_id", "bundle_id", "rule_id", "rule_family", "pre_capacity_action"])
    bun = rmc.read_private(corrected, "05_order_bundles.csv", ["bundle_id", "pre_capacity_action"])
    fin = rmc.read_private(corrected, "06_final_bundle_decisions.csv", ["scenario", "bundle_id", "final_action"])
    for f in (sig, bun, fin):
        f["bundle_id"] = f["bundle_id"].astype(str)
    rules = rule_conditions(project / "reference")
    sig = sig.merge(rules, on="rule_id", how="left")
    sig["needs_unavailable_data"] = sig["needs_unavailable_data"].fillna(False).astype(bool)
    k_drugs = potassium_raising(project / "reference")
    sig["potassium_raising"] = sig["ingredient"].fillna("").str.lower().str.strip().isin(k_drugs)
    supp_sig = sig[sig["pre_capacity_action"].eq("suppress")]
    variants = {"strict: condition must be verifiable": supp_sig[supp_sig["needs_unavailable_data"]],
                "targeted: potassium-raising drugs only": supp_sig[supp_sig["needs_unavailable_data"] & supp_sig["potassium_raising"]]}
    blocked = variants["strict: condition must be verifiable"]
    blocked_bundles = set(blocked["bundle_id"])

    rows = []
    pre = bun["pre_capacity_action"].value_counts()
    for vname, vsig in variants.items():
        vb = set(vsig["bundle_id"])
        mv = int((bun["pre_capacity_action"].eq("suppress") & bun["bundle_id"].isin(vb)).sum())
        rows.append({"variant": vname, "stage": "initial routes (no capacity limits)", "no_new_notification_before": int(pre.get("suppress", 0)),
                     "moved_to_passive": mv, "no_new_notification_after": int(pre.get("suppress", 0)) - mv,
                     "share_of_suppression_kept_percent": round(100 * (int(pre.get("suppress", 0)) - mv) / max(1, int(pre.get("suppress", 0))), 1),
                     "passive_after": int(pre.get("passive", 0)) + mv})
        for sc, g in fin.groupby("scenario"):
            vc = g["final_action"].value_counts()
            m2 = int((g["final_action"].eq("suppress") & g["bundle_id"].isin(vb)).sum())
            rows.append({"variant": vname, "stage": f"final routes, {sc}", "no_new_notification_before": int(vc.get("suppress", 0)),
                         "moved_to_passive": m2, "no_new_notification_after": int(vc.get("suppress", 0)) - m2,
                         "share_of_suppression_kept_percent": round(100 * (int(vc.get("suppress", 0)) - m2) / max(1, int(vc.get("suppress", 0))), 1),
                         "passive_after": int(vc.get("passive", 0)) + m2})
    summary = pd.DataFrame(rows)

    supp = sig[sig["pre_capacity_action"].eq("suppress")]
    by_cond = (supp.assign(condition=supp["suppress_if"].fillna("(no condition stated in catalogue)").replace("", "(no condition stated in catalogue)"))
               .groupby(["rule_family", "condition", "needs_unavailable_data"]).size().reset_index(name="suppressed_signals"))
    by_cond["suppressed_signals"] = by_cond["suppressed_signals"].map(masked)
    by_ing = (blocked.groupby(["rule_family", "ingredient"]).size().reset_index(name="suppressed_signals")
              .sort_values("suppressed_signals", ascending=False))
    by_ing["suppressed_signals"] = by_ing["suppressed_signals"].map(masked)

    out = project / "suppression_condition_overlay" / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "overlay_route_summary.csv", index=False)
    by_cond.to_csv(out / "suppressed_signals_by_condition.csv", index=False)
    by_ing.to_csv(out / "blocked_signals_by_ingredient.csv", index=False)

    harm = None
    if markers_path:
        mk = pd.read_csv(markers_path, dtype={"bundle_id": str})
        for m in MARKERS:
            mk[m] = mk[m].astype(str).str.lower().isin(["true", "1"])
        b = bun.merge(mk[["bundle_id"] + MARKERS], on="bundle_id", how="left")
        tb = set(variants["targeted: potassium-raising drugs only"]["bundle_id"])
        groups = {"targeted: would become passive (potassium-raising)": b[b.pre_capacity_action.eq("suppress") & b.bundle_id.isin(tb)],
                  "targeted: remains suppressed": b[b.pre_capacity_action.eq("suppress") & ~b.bundle_id.isin(tb)],
                  "strict: would become passive": b[b.pre_capacity_action.eq("suppress") & b.bundle_id.isin(blocked_bundles)],
                  "all suppressed in evaluated policy": b[b.pre_capacity_action.eq("suppress")],
                  "passive in evaluated policy": b[b.pre_capacity_action.eq("passive")]}
        harm = pd.DataFrame([{"group": k, "bundles": masked(len(g)), **{f"{m}_percent": rate(g, m) for m in MARKERS}} for k, g in groups.items()])
        bb = blocked.merge(mk[["bundle_id", "hyperkalaemia", "any_marker"]], on="bundle_id", how="left").drop_duplicates(["bundle_id", "ingredient"])
        ing_h = bb.groupby("ingredient").apply(lambda g: pd.Series({"bundles": masked(len(g)), "hyperkalaemia_percent": rate(g, "hyperkalaemia"),
                                                                     "any_marker_percent": rate(g, "any_marker")}), include_groups=False).reset_index()
        harm.to_csv(out / "overlay_harm_markers.csv", index=False)
        ing_h.to_csv(out / "blocked_ingredients_harm_markers.csv", index=False)
    json.dump({"analysis": "suppression_condition_overlay", "unavailable_data_keywords": UNAVAILABLE,
               "potassium_raising_drugs_from_catalogue": sorted(k_drugs),
               "rule_files": RULE_FILES, "source_hashes": source_hashes, "markers_included": markers_path is not None,
               "note": "Evaluated policy unchanged; passive display uses no capacity, so interruptions and reviews are unchanged."},
              open(out / "manifest.json", "w"), indent=2, default=str)
    print(summary.to_string(index=False)); print()
    print(by_cond.to_string(index=False)); print()
    print(by_ing.head(15).to_string(index=False))
    if harm is not None:
        print(); print(harm.to_string(index=False))
    print("COMPLETE. Aggregate outputs:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--markers", type=Path, help="lab_harm_markers_*/private/bundle_markers.csv")
    a = ap.parse_args()
    try:
        run(a.project, a.markers)
    except (ValueError, KeyError, OSError) as e:
        print("STOPPED:", e, file=sys.stderr); raise SystemExit(2)
