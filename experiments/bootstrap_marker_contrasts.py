"""Patient-level bootstrap intervals for the comparator harm-marker contrasts (main article, Section 5.6).

Bundles from the same patient are not independent, so whole patients are resampled with replacement.
For each route transition between routing by severity alone and CLARITY (no capacity limits), the script
reports the share of bundles followed by any new-onset laboratory marker, with a 95% percentile interval,
and the difference between the two contrasts reported in the article:
  A. suppress -> passive  versus  suppress -> suppress   (48.0% vs 37.6%)
  B. alert    -> escalate versus  alert    -> alert      (41.8% vs 50.2%)
The point estimates must reproduce results/comparator_harm_alignment/transition_harm_markers.csv,
otherwise the script stops. Only aggregate values are written.

Usage (from the repository root, same inputs as comparator_harm_alignment.py):
  python experiments/bootstrap_marker_contrasts.py --project . --markers <lab_harm_markers_DATE>/private/bundle_markers.csv
"""
from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd

REPS = 2000
SEED = 20261007
EXPECTED = {("suppress", "passive"): 48.02, ("suppress", "suppress"): 37.59,
            ("alert", "escalate"): 41.78, ("alert", "alert"): 50.24}
CONTRASTS = {"A_history_kept_visible_minus_both_suppressed": (("suppress", "passive"), ("suppress", "suppress")),
             "B_moved_to_review_minus_both_interrupted": (("alert", "escalate"), ("alert", "alert"))}
COMPARISON = "outputs_corrected/private/10b_snapshot_vs_clarity_bundle_comparison.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(project: Path, markers_path: Path) -> pd.DataFrame:
    """Join each affected bundle's route under both policies with its any-marker flag and patient."""
    comp = pd.read_csv(project / COMPARISON, dtype=str)
    cols = {c.lower(): c for c in comp.columns}
    pick = lambda *n: next((cols[x] for x in n if x in cols), None)
    bid, scen = pick("bundle_id"), pick("scenario")
    snap = pick("snapshot_action", "snapshot_final_action", "snapshot_route")
    clar = pick("clarity_action", "clarity_final_action", "clarity_route")
    if not all((bid, snap, clar)):
        raise SystemExit("STOPPED: expected columns not found: " + ", ".join(comp.columns))
    if scen:
        comp = comp[comp[scen].eq("unconstrained_reference")]
    comp = comp.rename(columns={bid: "bundle_id", snap: "snap", clar: "clar"})[["bundle_id", "snap", "clar"]]
    mk = pd.read_csv(markers_path, dtype={"bundle_id": str}, usecols=["bundle_id", "any_marker"])
    mk["any_marker"] = mk["any_marker"].astype(str).str.lower().isin(["true", "1"])
    df = comp.merge(mk, on="bundle_id", how="inner", validate="one_to_one")
    if len(df) != len(comp):
        raise SystemExit(f"STOPPED: {len(comp) - len(df)} bundles lacked markers")
    # bundle identifiers begin with the patient identifier: "<subject_id>-<hadm_id>-..."
    df["subject_id"] = df["bundle_id"].str.split("-", n=1).str[0]
    return df


def patient_counts(df: pd.DataFrame, groups):
    """Bundles and marker-followed bundles per patient for each route transition."""
    patients = np.sort(df["subject_id"].unique())
    pidx = {p: i for i, p in enumerate(patients)}
    n, e = {}, {}
    for g in groups:
        sub = df[(df["snap"] == g[0]) & (df["clar"] == g[1])]
        idx = sub["subject_id"].map(pidx).to_numpy()
        n[g] = np.bincount(idx, minlength=len(patients)).astype(float)
        e[g] = np.bincount(idx, weights=sub["any_marker"].to_numpy(float), minlength=len(patients))
    return patients, n, e


def bootstrap(n, e, n_patients: int, groups):
    """Resample patients with replacement; return the marker percentage of each group per resample."""
    rng = np.random.default_rng(SEED)
    boot = {g: np.empty(REPS) for g in groups}
    for r in range(REPS):
        w = np.bincount(rng.integers(0, n_patients, n_patients), minlength=n_patients)
        for g in groups:
            boot[g][r] = 100 * (w @ e[g]) / (w @ n[g])
    return boot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--markers", type=Path, required=True)
    a = ap.parse_args()
    df = load(a.project, a.markers)
    groups = sorted({g for pair in CONTRASTS.values() for g in pair})
    patients, n, e = patient_counts(df, groups)

    point = {g: 100 * e[g].sum() / n[g].sum() for g in groups}
    for g in groups:
        if abs(round(point[g], 2) - EXPECTED[g]) > 0.01:
            raise SystemExit(f"STOPPED: {g} gives {point[g]:.2f}%, expected {EXPECTED[g]}%")
    print("Reproduction check passed:", {f"{g[0]}->{g[1]}": round(point[g], 2) for g in groups})

    boot = bootstrap(n, e, len(patients), groups)
    rows = []
    for g in groups:
        lo, hi = np.percentile(boot[g], [2.5, 97.5])
        rows.append({"measure": f"{g[0]}->{g[1]} any marker (%)", "bundles": int(n[g].sum()),
                     "patients": int((n[g] > 0).sum()), "estimate": round(point[g], 1),
                     "ci_low": round(lo, 1), "ci_high": round(hi, 1)})
    for name, (g1, g2) in CONTRASTS.items():
        lo, hi = np.percentile(boot[g1] - boot[g2], [2.5, 97.5])
        rows.append({"measure": f"{name} (percentage points)", "bundles": None, "patients": None,
                     "estimate": round(point[g1] - point[g2], 1), "ci_low": round(lo, 1), "ci_high": round(hi, 1)})
    out = pd.DataFrame(rows)
    dest = a.project / "bootstrap_marker_contrasts" / "aggregate"
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / "bootstrap_marker_contrasts.csv", index=False)
    manifest = {"script_sha256": sha256(Path(__file__)),
                "inputs_sha256": {"comparison": sha256(a.project / COMPARISON), "markers": sha256(a.markers)},
                "reps": REPS, "seed": SEED, "resampling_unit": "patient", "patients": int(len(patients)),
                "interval": "95% percentile", "reproduction_check": "passed",
                "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(out.to_string(index=False))
    print(f"\nWritten to {dest}")


if __name__ == "__main__":
    main()
