"""Laboratory harm markers following each affected order bundle.

Question: are the concerns that the policy routes, protects and prioritises the ones
that are more often followed by laboratory evidence of harm?

For every affected order bundle, with decision time t, the script records five
pre-specified, new-onset laboratory markers in a window after t:

    acute kidney injury   creatinine rise >= 0.3 mg/dL within 48 h above the latest value
                          in the previous 48 h, or >= 1.5 x the lowest value in the previous
                          7 days within 7 days (creatinine criteria of KDIGO)
    hyperkalaemia         potassium >= 6.0 mmol/L within 72 h (latest earlier value < 6.0 or none)
    hypoglycaemia         glucose < 54 mg/dL within 72 h (latest earlier value >= 54 or none)
    high INR              INR > 4.0 within 7 days (latest earlier value <= 4.0 or none)
    high tacrolimus       tacrolimus > 15 ng/mL within 7 days (latest earlier value <= 15 or none)

Each marker also records whether the laboratory test was measured in the window at all,
because sicker patients are tested more often (ascertainment).

The markers are compared across initial routes, final routes at each capacity, priority
rules and, when the private specification scores from robust_tier.py are supplied, the
robustness tiers.

Interpretation boundary: these are markers of risk observed under the care patients
actually received. They are not effects of any routing policy, which was never deployed,
and they are not adjudicated adverse drug events. Counts from 1 to 10 are masked.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KIT = Path(__file__).resolve().parent
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

import allocation_robustness as ar  # noqa: E402

HOUR = np.timedelta64(1, "h")
MASK_BELOW = 11
MARKERS = ["aki", "hyperkalaemia", "hypoglycaemia", "high_inr", "high_tacrolimus"]

# Laboratory items are identified by label so that the choice is visible and checkable.
LAB_RULES = {
    "creatinine": {"labels": [r"^creatinine$"], "fluid": "blood", "fixed_itemids": [50912]},
    "potassium": {"labels": [r"^potassium$", r"^potassium, whole blood$"], "fluid": "blood"},
    "glucose": {"labels": [r"^glucose$"], "fluid": "blood"},
    "inr": {"labels": [r"^inr\(pt\)$"], "fluid": "blood"},
    "tacrolimus": {"labels": [r"tacro", r"fk ?506", r"prograf"], "fluid": None, "optional": True},
}


# ---------------------------------------------------------------------------
# Laboratory loading
# ---------------------------------------------------------------------------

def identify_items(d_labitems: pd.DataFrame) -> dict[str, list[int]]:
    items = d_labitems.copy()
    items["label_l"] = items["label"].astype(str).str.strip().str.lower()
    items["fluid_l"] = items["fluid"].astype(str).str.strip().str.lower()
    found = {}
    for name, rule in LAB_RULES.items():
        if rule.get("fixed_itemids"):
            found[name] = list(rule["fixed_itemids"])
            continue
        fluid_ok = items["fluid_l"].eq(rule["fluid"]) if rule.get("fluid") else True
        mask = fluid_ok & items["label_l"].apply(lambda s: any(re.search(p, s) for p in rule["labels"]))
        found[name] = sorted(int(i) for i in items.loc[mask, "itemid"])
    return found


def load_labs(path: Path, itemids: dict[str, list[int]], subjects: set[int], chunksize: int = 2_000_000) -> pd.DataFrame:
    lookup = {i: name for name, ids in itemids.items() for i in ids}
    parts = []
    for chunk in pd.read_csv(path, usecols=["subject_id", "itemid", "charttime", "valuenum"], chunksize=chunksize,
                             low_memory=False):
        chunk = chunk[chunk["itemid"].isin(lookup) & chunk["subject_id"].isin(subjects)]
        if len(chunk):
            parts.append(chunk.dropna(subset=["valuenum", "charttime"]))
    if not parts:
        return pd.DataFrame(columns=["subject_id", "lab", "time", "value"])
    labs = pd.concat(parts, ignore_index=True)
    labs["lab"] = labs["itemid"].map(lookup)
    labs["time"] = pd.to_datetime(labs["charttime"], errors="coerce")
    labs["value"] = pd.to_numeric(labs["valuenum"], errors="coerce")
    labs = labs.dropna(subset=["time", "value"])
    return labs[["subject_id", "lab", "time", "value"]].sort_values(["subject_id", "lab", "time"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Marker computation
# ---------------------------------------------------------------------------

def _series(labs: pd.DataFrame) -> dict:
    out = {}
    if labs.empty:
        return out
    for (sid, lab), g in labs.groupby(["subject_id", "lab"], sort=False):
        out[(int(sid), lab)] = (g["time"].to_numpy(dtype="datetime64[ns]"), g["value"].to_numpy(dtype=float))
    return out


def _window(series, t, before_h, after_h):
    """Values in [t - before, t] and (t, t + after]."""
    times, values = series
    lo = np.searchsorted(times, t - before_h * HOUR, side="left")
    mid = np.searchsorted(times, t, side="right")
    hi = np.searchsorted(times, t + after_h * HOUR, side="right")
    return values[lo:mid], values[mid:hi]


def threshold_marker(series, t, before_h, after_h, test):
    """New-onset marker: some post value meets test and the latest earlier value does not."""
    if series is None:
        return False, False
    pre, post = _window(series, t, before_h, after_h)
    if len(post) == 0:
        return False, False
    if len(pre) and test(pre[-1]):
        return True, False
    return True, bool(np.any(test(post)))


def aki_marker(series, t):
    if series is None:
        return False, False
    pre48, post48 = _window(series, t, 48, 48)
    pre7d, post7d = _window(series, t, 168, 168)
    if len(post7d) == 0 or len(pre7d) == 0:
        return len(post7d) > 0, False
    rise = len(pre48) > 0 and len(post48) > 0 and np.max(post48) >= pre48[-1] + 0.3
    ratio = np.max(post7d) >= 1.5 * np.min(pre7d)
    return True, bool(rise or ratio)


def compute_markers(bundles: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    series = _series(labs)
    rows = []
    for sid, bid, ts in zip(bundles["subject_id"].astype(int), bundles["bundle_id"].astype(str),
                            pd.to_datetime(bundles["timestamp"]).to_numpy(dtype="datetime64[ns]")):
        get = lambda lab: series.get((sid, lab))
        res = {"bundle_id": bid}
        res["aki_measured"], res["aki"] = aki_marker(get("creatinine"), ts)
        res["hyperkalaemia_measured"], res["hyperkalaemia"] = threshold_marker(get("potassium"), ts, 72, 72, lambda v: v >= 6.0)
        res["hypoglycaemia_measured"], res["hypoglycaemia"] = threshold_marker(get("glucose"), ts, 72, 72, lambda v: v < 54)
        res["high_inr_measured"], res["high_inr"] = threshold_marker(get("inr"), ts, 168, 168, lambda v: v > 4.0)
        res["high_tacrolimus_measured"], res["high_tacrolimus"] = threshold_marker(get("tacrolimus"), ts, 168, 168, lambda v: v > 15)
        rows.append(res)
    out = pd.DataFrame(rows)
    out["any_marker"] = out[MARKERS].any(axis=1)
    out["any_measured"] = out[[m + "_measured" for m in MARKERS]].any(axis=1)
    return out


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def masked(n):
    n = int(n)
    return n if n == 0 or n >= MASK_BELOW else f"<{MASK_BELOW}"


def summarise(frame: pd.DataFrame, group_col: str, label: dict) -> list[dict]:
    rows = []
    for key, g in frame.groupby(group_col, sort=True):
        n = len(g)
        row = {**label, "group": key, "bundles": masked(n)}
        for m in MARKERS + ["any_marker"]:
            meas = g["any_measured"] if m == "any_marker" else g[m + "_measured"]
            ev = int(g[m].sum())
            row[f"{m}_events"] = masked(ev)
            row[f"{m}_measured"] = masked(int(meas.sum()))
            ok = n >= MASK_BELOW and (ev == 0 or ev >= MASK_BELOW)
            row[f"{m}_percent_of_bundles"] = round(100 * ev / n, 2) if ok else None
            row[f"{m}_percent_of_measured"] = round(100 * ev / meas.sum(), 2) if ok and meas.sum() else None
        rows.append(row)
    return rows


def priority_capture(pool: pd.DataFrame, markers: pd.DataFrame, limits: list[dict], seeds: int) -> list[dict]:
    frame = pool.merge(markers[["bundle_id", "any_marker", "aki", "hyperkalaemia"]], on="bundle_id", how="left")
    for col in ("any_marker", "aki", "hyperkalaemia"):
        frame[col] = frame[col].fillna(False).astype(bool)
    base = ar.base_hashes(frame["bundle_id"])
    rows = []
    for budget in limits:
        k = min(int(budget["expert_slots"]), len(frame))
        for rule in ar.PRIORITY_RULES:
            captured = {c: [] for c in ("any_marker", "aki", "hyperkalaemia")}
            for seed in range(seeds):
                order = ar.order_by_random(frame, rule, ar.tie_keys(base, seed))[:k]
                for c in captured:
                    captured[c].append(int(frame[c].to_numpy()[order].sum()))
            rows.append({"scenario": budget["scenario"], "review_slots": k, "rule": rule, "seeds": seeds,
                         "pool_bundles_with_any_marker": int(frame["any_marker"].sum()),
                         **{f"selected_with_{c}_mean": round(float(np.mean(v)), 1) for c, v in captured.items()},
                         "selected_with_any_marker_percent": round(100 * float(np.mean(captured["any_marker"])) / k, 2) if k else None})
    return rows


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def run(project: Path, labevents: Path, d_labitems: Path, scores: Path | None, seeds: int, output: Path | None = None):
    import run_matched_comparison as rmc
    import robust_tier as rt

    started = time.time()
    project = project.expanduser().resolve()
    print("1/6 Checking source and frozen inputs.", flush=True)
    source_hashes, _ = rmc.source_guard(project)
    corrected = project / "outputs_corrected"
    decisions = rmc.read_private(corrected, "04_action_sets.csv")
    bundles = rmc.read_private(corrected, "05_order_bundles.csv")
    final = rmc.read_private(corrected, "06_final_bundle_decisions.csv")
    bundles["bundle_id"] = bundles["bundle_id"].astype(str)

    print("2/6 Identifying laboratory items by label.", flush=True)
    items = identify_items(pd.read_csv(d_labitems))
    unavailable = []
    for name, ids in items.items():
        print(f"   {name}: itemids {ids}", flush=True)
        if not ids and LAB_RULES[name].get("optional"):
            unavailable.append(name)
            print(f"   NOTE: no {name} test found; that marker will be reported as not available.", flush=True)
        elif not ids:
            raise ValueError(f"No laboratory item found for {name}; check d_labitems")

    print("3/6 Loading laboratory results for cohort patients (this can take several minutes).", flush=True)
    labs = load_labs(labevents, items, set(bundles["subject_id"].astype(int)))
    print(f"   {len(labs):,} relevant laboratory results", flush=True)

    print("4/6 Computing markers for each affected order bundle.", flush=True)
    markers = compute_markers(bundles, labs)
    frame = bundles[["bundle_id", "pre_capacity_action"]].merge(markers, on="bundle_id", validate="one_to_one")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = output.resolve() if output else project / ("lab_harm_markers_" + stamp)
    aggregate, private = output / "aggregate", output / "private"
    aggregate.mkdir(parents=True, exist_ok=False)
    private.mkdir()
    markers.to_csv(private / "bundle_markers.csv", index=False)

    print("5/6 Summarising by route, capacity, priority rule and robustness tier.", flush=True)
    rows = summarise(frame.assign(all="all affected bundles"), "all", {"comparison": "overall"})
    rows += summarise(frame, "pre_capacity_action", {"comparison": "initial_route"})
    final = final.copy()
    final["bundle_id"] = final["bundle_id"].astype(str)
    for sc in ("low_capacity", "medium_capacity", "high_capacity"):
        f = final[final["scenario"].eq(sc)][["bundle_id", "final_action"]].merge(markers, on="bundle_id")
        rows += summarise(f, "final_action", {"comparison": f"final_route_{sc}"})
    limits = [b for b in rmc.capacity_limits(final, {"clarity": bundles}) if b["scenario"] != "unconstrained_reference"]
    pool = ar.review_pool(bundles, ar.bundle_context(decisions))
    if scores is not None:
        saved = pd.read_csv(scores, dtype={"bundle_id": str})
        tiered = pool[["bundle_id"]].merge(saved, on="bundle_id", how="left", validate="one_to_one")
        best, worst = rt.set_bounds(tiered, rt.SETS[rt.MAIN_SET])
        for budget in limits:
            status = rt.classify(best, worst, int(budget["expert_slots"]))
            t = tiered[["bundle_id"]].assign(tier=status).merge(markers, on="bundle_id")
            rows += summarise(t, "tier", {"comparison": f"robust_tier_{budget['scenario']}"})
    pd.DataFrame(rows).to_csv(aggregate / "harm_markers_by_group.csv", index=False)
    overlap = []
    for route, g in frame.groupby("pre_capacity_action"):
        overlap.append({"initial_route": route, "bundles": masked(len(g)),
                        "aki_and_hyperkalaemia": masked((g["aki"] & g["hyperkalaemia"]).sum()),
                        "aki_only": masked((g["aki"] & ~g["hyperkalaemia"]).sum()),
                        "hyperkalaemia_only": masked((~g["aki"] & g["hyperkalaemia"]).sum()),
                        "markers_per_bundle_mean": round(float(g[MARKERS].sum(axis=1).mean()), 3)})
    pd.DataFrame(overlap).to_csv(aggregate / "harm_markers_overlap_by_route.csv", index=False)
    pd.DataFrame(priority_capture(pool, markers, limits, seeds)).to_csv(aggregate / "harm_markers_priority_capture.csv", index=False)

    print("6/6 Writing manifest.", flush=True)
    manifest = {
        "experiment": "lab_harm_markers", "version": "1.0",
        "laboratory_itemids": items, "lab_results_used": int(len(labs)),
        "laboratory_tests_not_available": unavailable,
        "definitions": {
            "aki": "creatinine rise >= 0.3 mg/dL within 48 h above latest value in prior 48 h, or >= 1.5 x lowest value in prior 7 days within 7 days",
            "hyperkalaemia": "potassium >= 6.0 within 72 h; latest prior value (72 h) < 6.0 or none",
            "hypoglycaemia": "glucose < 54 mg/dL within 72 h; latest prior value (72 h) >= 54 or none",
            "high_inr": "INR > 4.0 within 7 days; latest prior value (7 days) <= 4.0 or none",
            "high_tacrolimus": "tacrolimus > 15 ng/mL within 7 days; latest prior value (7 days) <= 15 or none",
            "time_reference": "bundle decision timestamp; laboratory chart time",
        },
        "robust_tier_included": scores is not None, "tie_seeds": seeds,
        "source_hashes": source_hashes, "script_hash": sha(Path(__file__)),
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "runtime_seconds": round(time.time() - started),
        "small_cell_rule": f"Counts from 1 to {MASK_BELOW - 1} are masked.",
        "interpretation_boundaries": [
            "Markers reflect care actually received; they are not effects of any routing policy.",
            "Markers are laboratory signals, not adjudicated adverse drug events.",
            "Testing frequency differs between groups; compare event rates with measurement rates.",
            "Bundles from the same patient are not independent.",
        ],
    }
    (aggregate / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print("COMPLETE. Aggregate outputs: " + str(aggregate), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--labevents", type=Path, required=True, help="MIMIC-IV hosp labevents.csv")
    parser.add_argument("--d-labitems", type=Path, required=True, help="MIMIC-IV hosp d_labitems.csv")
    parser.add_argument("--scores", type=Path, help="robust_tier_*/private/specification_scores.csv")
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    run(a.project, a.labevents, a.d_labitems, a.scores, a.seeds, a.output)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, AssertionError, OSError, KeyError) as error:
        print("STOPPED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
