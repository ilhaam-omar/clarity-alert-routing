"""Robust-tier analysis: which review selections does the model actually determine?

For every review candidate and every specification, the candidate's possible
positions in the priority order form an interval:

    best rank  = 1 + number of candidates with a strictly higher score
    worst rank = number of candidates with a score at least as high

Any tie-breaking rule places the candidate somewhere in that interval. Across a
set of specifications, a candidate is therefore

    robustly selected  at K review slots if its worst rank is <= K in every specification,
    robustly excluded  at K review slots if its best rank  >  K in every specification,
    undetermined       otherwise.

This is exact for every tie-breaking rule, so no random sampling is involved.
A random-key simulation is run only as an internal consistency check.

Nested specification sets are reported:

    tie_breaking_only    primary specification only (ties are the only freedom)
    numerical_settings   + half/double design points, alternate seeds, regret quantiles 0.90/0.975
    plus_range_width     + parameter ranges contracted by 50% around their midpoints
    plus_fixed_point     + single workflow-balanced midpoint vector

The support and suppression thresholds do not change the priority score, so they
are not needed here (Experiment 2 showed rank correlation 1.0 for them).

"Robust" means determined by the model under the stated assumptions. It is not
evidence that the selection is clinically correct. Counts from 1 to 10 in the
characteristics table are masked.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
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
from matched_policies import booleans  # noqa: E402

SETS = {
    "tie_breaking_only": ["primary"],
    "numerical_settings": ["primary", "half_design_points", "double_design_points", "alternate_seeds",
                           "regret_quantile_0.90", "regret_quantile_0.975"],
}
SETS["plus_range_width"] = SETS["numerical_settings"] + ["contracted_ranges_50pct"]
SETS["plus_fixed_point"] = SETS["plus_range_width"] + ["fixed_midpoint_workflow"]
MAIN_SET = "plus_range_width"
SCORE_SPECS = [s for s in SETS["plus_fixed_point"] if s != "primary"]
MASK_BELOW = 11


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------

def rank_bounds(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best and worst 1-based descending rank of each score under any tie rule."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    ascending = np.sort(scores)
    greater = n - np.searchsorted(ascending, scores, side="right")
    at_least = n - np.searchsorted(ascending, scores, side="left")
    return greater + 1, at_least


def set_bounds(score_frame: pd.DataFrame, specs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Best rank over specifications (minimum) and worst rank (maximum)."""
    best = np.full(len(score_frame), np.iinfo(np.int64).max, dtype=np.int64)
    worst = np.zeros(len(score_frame), dtype=np.int64)
    for spec in specs:
        b, w = rank_bounds(score_frame[spec].to_numpy(dtype=float))
        best = np.minimum(best, b)
        worst = np.maximum(worst, w)
    return best, worst


def classify(best: np.ndarray, worst: np.ndarray, slots: int) -> np.ndarray:
    status = np.full(len(best), "undetermined", dtype=object)
    status[worst <= slots] = "robustly_selected"
    status[best > slots] = "robustly_excluded"
    return status


def curve(best: np.ndarray, worst: np.ndarray) -> pd.DataFrame:
    n = len(best)
    k = np.arange(0, n + 1)
    selected = np.searchsorted(np.sort(worst), k, side="right")
    excluded = n - np.searchsorted(np.sort(best), k, side="right")
    return pd.DataFrame({"review_slots": k, "robustly_selected": selected,
                         "robustly_excluded": excluded, "undetermined": n - selected - excluded})


def simulate_check(score_frame: pd.DataFrame, specs: list[str], slots: int, status: np.ndarray, seeds: int = 25) -> bool:
    """Every random tie-break under every specification must respect the classification."""
    base = ar.base_hashes(score_frame["bundle_id"])
    must = status == "robustly_selected"
    never = status == "robustly_excluded"
    k = min(int(slots), len(score_frame))
    for spec in specs:
        score = -score_frame[spec].to_numpy(dtype=float)
        for seed in range(seeds):
            order = np.lexsort([ar.tie_keys(base, seed), score])
            chosen = np.zeros(len(score_frame), dtype=bool)
            chosen[order[:k]] = True
            if (must & ~chosen).any() or (never & chosen).any():
                return False
    return True


def masked(count: int):
    return count if count == 0 or count >= MASK_BELOW else f"<{MASK_BELOW}"


def characteristics(pool: pd.DataFrame, families: pd.DataFrame, status: np.ndarray, label: dict) -> list[dict]:
    frame = pool.assign(tier=status).merge(families, on="bundle_id", how="left", validate="one_to_one")
    family_columns = [c for c in families.columns if c.startswith("family_")]
    rows = []
    for tier in ("robustly_selected", "undetermined", "robustly_excluded"):
        part = frame[frame["tier"].eq(tier)]
        n = len(part)
        row = {**label, "tier": tier, "bundles": masked(n)}
        if n >= MASK_BELOW:
            row.update({
                "distinct_patients": masked(int(part["subject_id"].astype(str).nunique())),
                "acute_renal_deterioration_percent": 100 * part["any_acute_renal_deterioration"].mean(),
                "material_change_percent": 100 * part["any_material_change"].mean(),
                "mean_signal_count": pd.to_numeric(part["signal_count"], errors="coerce").mean(),
                "mean_primary_score": part["primary"].mean(),
                "primary_score_zero_percent": 100 * part["primary"].eq(0).mean(),
            })
            for col in family_columns:
                hits = int(part[col].sum())
                row[col.replace("family_", "contains_") + "_percent"] = (
                    100 * hits / n if hits == 0 or hits >= MASK_BELOW else f"<{MASK_BELOW} bundles")
        rows.append(row)
    return rows


def rule_families(decisions: pd.DataFrame) -> pd.DataFrame:
    families = decisions[["bundle_id", "rule_family"]].copy()
    families["bundle_id"] = families["bundle_id"].astype(str)
    families["rule_family"] = "family_" + families["rule_family"].astype(str).str.strip().str.lower()
    table = pd.crosstab(families["bundle_id"], families["rule_family"]).clip(upper=1)
    return table.reset_index()


def plot(curves: pd.DataFrame, limits: list[dict], pool_size: int, directory: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    main = curves[curves["specification_set"].eq(MAIN_SET)]
    ax = axes[0]
    ax.stackplot(main["review_slots"], main["robustly_selected"], main["undetermined"], main["robustly_excluded"],
                 labels=["Robustly selected", "Undetermined", "Robustly excluded"],
                 colors=["#2b7a4b", "#d9a441", "#b8b8b8"], alpha=0.9)
    for budget in limits:
        ax.axvline(budget["expert_slots"], color="black", lw=0.8, ls=":")
        ax.text(budget["expert_slots"], pool_size * 1.01, budget["scenario"].split("_")[0], ha="center", fontsize=8)
    ax.set_xlim(0, pool_size)
    ax.set_ylim(0, pool_size * 1.06)
    ax.set_xlabel("Review slots")
    ax.set_ylabel("Review candidates (order bundles)")
    ax.set_title("(a) Model uncertainty set: numerical settings + range width", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8, frameon=False)

    ax = axes[1]
    styles = {"tie_breaking_only": (":", "#7f7f7f"), "numerical_settings": ("--", "#1f5f8b"),
              "plus_range_width": ("-", "#2b7a4b"), "plus_fixed_point": ("-.", "#b03a2e")}
    names = {"tie_breaking_only": "Tie-breaking only", "numerical_settings": "+ numerical settings",
             "plus_range_width": "+ range width", "plus_fixed_point": "+ fixed point"}
    for name, (style, colour) in styles.items():
        part = curves[curves["specification_set"].eq(name)]
        share = 100 * part["robustly_selected"] / part["review_slots"].clip(lower=1)
        ax.plot(part["review_slots"], share, style, color=colour, lw=1.4, label=names[name])
    for budget in limits:
        ax.axvline(budget["expert_slots"], color="black", lw=0.8, ls=":")
    ax.set_xlim(0, pool_size)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Review slots")
    ax.set_ylabel("Slots filled by robustly selected candidates (%)")
    ax.set_title("(b) Determined share as assumptions widen", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"figure_robust_tier.{suffix}", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def compute_scores(states: pd.DataFrame, envelopes, action_config, policy_class, sample_design) -> dict:
    scores = {}
    variants = {name: (envs, cfg) for name, envs, cfg in psa.specifications(envelopes, action_config)}
    for name in SCORE_SPECS:
        t0 = time.time()
        if name == "fixed_midpoint_workflow":
            vector = psa.midpoint_vector(next(e for e in envelopes if e.name == "workflow_balanced"))
            decided = psa.make_point_policy(policy_class)(vector).decide_dataframe(states)
        else:
            envs, cfg = variants[name]
            decided = policy_class(sample_design(envs), {e.name: e.regret_quantile for e in envs}, cfg).decide_dataframe(states)
        scores[name] = psa.bundle_delta(decided)
        print(f"   {name}: {time.time() - t0:.0f}s", flush=True)
    return scores


def run(project: Path, output: Path | None = None, scores_file: Path | None = None) -> Path:
    import run_matched_comparison as rmc

    project = project.expanduser().resolve()
    sys.path.insert(0, str(project))
    from clarity_v3.config import load_uncertainty_config
    from clarity_v3.uncertainty import MultiEnvelopeActionPolicy, sample_design

    started = time.time()
    print("1/5 Checking source and frozen inputs.", flush=True)
    source_hashes, _ = rmc.source_guard(project)
    corrected = project / "outputs_corrected"
    decisions = rmc.read_private(corrected, "04_action_sets.csv")
    bundles = rmc.read_private(corrected, "05_order_bundles.csv")
    final = rmc.read_private(corrected, "06_final_bundle_decisions.csv")
    limits = [b for b in rmc.capacity_limits(final, {"clarity": bundles}) if b["scenario"] != "unconstrained_reference"]
    pool = ar.review_pool(bundles, ar.bundle_context(decisions))
    pool["primary"] = pool["delta_expert"].astype(float)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = output.resolve() if output else project / ("robust_tier_" + stamp)
    output.mkdir(parents=True, exist_ok=False)
    aggregate, private = output / "aggregate", output / "private"
    aggregate.mkdir()
    private.mkdir()

    if scores_file:
        print("2/5 Reusing saved private specification scores: " + str(scores_file), flush=True)
        saved = pd.read_csv(scores_file, dtype={"bundle_id": str})
        pool = pool.merge(saved.drop(columns=["primary"], errors="ignore"), on="bundle_id", how="left", validate="one_to_one")
    else:
        print(f"2/5 Recomputing {len(SCORE_SPECS)} specification scores (about 2-5 minutes each).", flush=True)
        states = rmc.read_private(corrected, "03_longitudinal_states.csv")
        envelopes, action_config = load_uncertainty_config(project / "config/uncertainty_design.json")
        for name, series in compute_scores(states, envelopes, action_config, MultiEnvelopeActionPolicy, sample_design).items():
            pool[name] = pool["bundle_id"].map(series).fillna(0.0)
        pool[["bundle_id", "primary", *SCORE_SPECS]].to_csv(private / "specification_scores.csv", index=False)
    missing = [s for s in SETS["plus_fixed_point"] if s not in pool or pool[s].isna().any()]
    if missing:
        raise ValueError("Missing specification scores: " + ", ".join(missing))

    print("3/5 Classifying candidates and checking against random tie-breaks.", flush=True)
    families = rule_families(decisions[decisions["bundle_id"].astype(str).isin(pool["bundle_id"])])
    curve_frames, summary_rows, character_rows, checks = [], [], [], {}
    for set_name, specs in SETS.items():
        best, worst = set_bounds(pool, specs)
        frame = curve(best, worst)
        frame.insert(0, "specification_set", set_name)
        curve_frames.append(frame)
        for budget in limits:
            slots = int(budget["expert_slots"])
            status = classify(best, worst, slots)
            ok = simulate_check(pool, specs, slots, status)
            checks[f"{set_name}_{budget['scenario']}_random_tiebreaks_respect_classification"] = ok
            if not ok:
                raise AssertionError(f"Classification violated by a simulated tie-break: {set_name}/{budget['scenario']}")
            counts = pd.Series(status).value_counts()
            summary_rows.append({"specification_set": set_name, "specifications": len(specs),
                                 "scenario": budget["scenario"], "review_slots": slots, "candidates": len(pool),
                                 **{t: int(counts.get(t, 0)) for t in ("robustly_selected", "undetermined", "robustly_excluded")},
                                 "slots_filled_by_robust_percent": 100 * int(counts.get("robustly_selected", 0)) / slots,
                                 "slots_left_to_choice": slots - int(counts.get("robustly_selected", 0))})
            character_rows += characteristics(pool, families, status,
                                              {"specification_set": set_name, "scenario": budget["scenario"]})
    curves = pd.concat(curve_frames, ignore_index=True)

    print("4/5 Writing tables and figure.", flush=True)
    pd.DataFrame(summary_rows).to_csv(aggregate / "robust_tier_summary.csv", index=False)
    curves.to_csv(aggregate / "robust_tier_curve.csv", index=False)
    pd.DataFrame(character_rows).to_csv(aggregate / "robust_tier_characteristics.csv", index=False)
    plot(curves, limits, len(pool), aggregate)

    print("5/5 Writing manifest.", flush=True)
    manifest = {
        "experiment": "robust_tier", "version": "1.0", "specification_sets": SETS, "main_set": MAIN_SET,
        "definition": "Robustly selected at K: worst possible rank <= K under every specification and every tie rule; "
                      "robustly excluded: best possible rank > K under every specification.",
        "capacity_limits": limits, "checks": checks, "source_hashes": source_hashes,
        "script_hash": sha(Path(__file__)), "python": sys.version.split()[0], "numpy": np.__version__,
        "pandas": pd.__version__, "runtime_seconds": round(time.time() - started),
        "small_cell_rule": f"Counts from 1 to {MASK_BELOW - 1} are masked in the characteristics table.",
        "interpretation_boundaries": [
            "Robust means determined by the model under the stated assumptions, not clinically correct.",
            "Undetermined candidates are those whose selection depends on tie rules or numerical assumptions.",
            "The specification sets are declared stress tests, not a probability distribution.",
        ],
        "clinical_validation": False,
    }
    (aggregate / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print("COMPLETE. Aggregate outputs: " + str(aggregate), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scores", type=Path, help="Reuse private/specification_scores.csv from an earlier run")
    args = parser.parse_args()
    run(args.project, args.output, args.scores)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, AssertionError, OSError, KeyError) as error:
        print("STOPPED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
