"""Experiment 2: parameter sensitivity carried through to review selection.

The archived specification analysis reports only signal-level action agreement.
High agreement there cannot show that the same bundles receive review, because
review priority depends on the regret score, not only on the selected route.

For each alternative specification this script reports three levels:

1. Signal level: agreement of initial actions with the primary specification.
2. Ranking only: the primary review pool and suppression gate are held fixed;
   only the priority score is recomputed. This isolates score sensitivity.
3. Through allocation: routes, bundles, the review pool and the suppression
   gate are recomputed and allocated with the fixed primary absolute budgets.

Two comparators are added to the archived specifications:

* contracted_ranges_50pct: every envelope range halved around its midpoint,
  keeping seeds and the existing order restrictions. This tests dependence on
  assumed range width, not calibration.
* fixed_midpoint_workflow: a single parameter vector (the workflow-balanced
  envelope midpoints, after the same order restrictions). Score = best
  admissible non-review cost minus review cost, clipped at zero and summed per
  bundle, exactly as the regret score is aggregated. Ranking only: it asks
  whether the multi-envelope uncertainty calculation ranks reviews differently
  from a single point assumption.

Review selections use the reproducible random tie keys from Experiment 1, with
identical keys for the primary and alternative scores. Variation across tie
keys is algorithmic variability, not a clinical confidence interval. An
alternative that agrees with the primary score shows computational stability,
not clinical validity.
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
from matched_capacity import allocate_absolute  # noqa: E402
from matched_policies import booleans, shared_suppression_gate  # noqa: E402

PARAMETERS = (
    "rho_passive", "rho_alert", "rho_escalate", "burden_passive", "burden_alert",
    "burden_escalate", "uncertainty_penalty", "uncertainty_alert_fraction",
    "uncertainty_escalate_fraction", "repeat_alert_penalty", "managed_residual_fraction",
    "partly_managed_residual_fraction", "escalation_delay_penalty",
)
ACTIONS = ("suppress", "passive", "alert", "escalate")


# ---------------------------------------------------------------------------
# Specifications
# ---------------------------------------------------------------------------

def contracted(envelope, fraction: float = 0.5):
    """Shrink every range around its midpoint; keep seed and point count."""
    ranges = {}
    for name, (low, high) in envelope.ranges.items():
        mid, half = (low + high) / 2.0, (high - low) / 2.0 * fraction
        ranges[name] = (mid - half, mid + half)
    return replace(envelope, ranges=ranges)


def midpoint_vector(envelope) -> dict:
    """Envelope midpoints with the same partial-order restrictions as sampling.
    For a grounded (judgement-defined) envelope, the median of its accepted weightings."""
    if getattr(envelope, "anchor_rules", ()):
        from clarity_v3.uncertainty import sample_envelope
        frame = sample_envelope(envelope)
        return {name: float(frame[name].median()) for name in PARAMETERS}
    v = {name: (float(low) + float(high)) / 2.0 for name, (low, high) in envelope.ranges.items()}
    v["rho_alert"] = min(v["rho_alert"], v["rho_passive"] - 1e-6)
    v["rho_escalate"] = min(v["rho_escalate"], v["rho_passive"] - 1e-6)
    v["burden_alert"] = max(v["burden_alert"], v["burden_passive"] + 1e-6)
    v["uncertainty_escalate_fraction"] = min(v["uncertainty_escalate_fraction"], v["uncertainty_alert_fraction"])
    v["partly_managed_residual_fraction"] = max(v["partly_managed_residual_fraction"], v["managed_residual_fraction"])
    return {name: v[name] for name in PARAMETERS}


def specifications(envelopes, action_config):
    """Archived Stage-10 specifications plus the contracted-range comparator."""
    specs = [
        ("half_design_points", [replace(e, n_points=max(32, e.n_points // 2)) for e in envelopes], action_config),
        ("double_design_points", [replace(e, n_points=e.n_points * 2) for e in envelopes], action_config),
        ("alternate_seeds", [replace(e, seed=e.seed + 101) for e in envelopes], action_config),
        ("regret_quantile_0.90", [replace(e, regret_quantile=0.90) for e in envelopes], action_config),
        ("regret_quantile_0.975", [replace(e, regret_quantile=0.975) for e in envelopes], action_config),
        ("support_threshold_0.60", envelopes, replace(action_config, consensus_support_threshold=0.60)),
        ("support_threshold_0.80", envelopes, replace(action_config, consensus_support_threshold=0.80)),
        ("suppress_threshold_0.90", envelopes, replace(action_config, suppress_support_threshold=0.90)),
        ("contracted_ranges_50pct", [contracted(e) for e in envelopes], action_config),
    ]
    return specs


def make_point_policy(policy_class):
    """Single-parameter-vector scorer reusing the engine's own cost function."""

    class _Record:
        def __init__(self, values):
            self.values = values

        def as_dict(self):
            return dict(self.values)

    class PointCostPolicy(policy_class):
        def __init__(self, vector: dict):
            design = pd.DataFrame([{"envelope": "point", **vector}])
            super().__init__(design, {"point": 0.95})
            self._frame = design

        def decide(self, state):
            costs = self._costs(state, self._frame)[0]
            index = {a: i for i, a in enumerate(ACTIONS)}
            admissible = set(state.admissible_actions)
            if "escalate" in admissible and any(a != "escalate" for a in admissible):
                best_other = min(costs[index[a]] for a in admissible if a != "escalate")
                delta = float(best_other - costs[index["escalate"]])
            else:
                delta = float("-inf")
            return _Record({"signal_id": state.signal_id, "admissible_actions": "", "safety_class": "",
                            "delta_expert": delta})

    return PointCostPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bundle_delta(decisions: pd.DataFrame) -> pd.Series:
    """Positive finite signal regret differences summed per bundle (as in bundles.py)."""
    values = pd.to_numeric(decisions["delta_expert"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return values.clip(lower=0).groupby(decisions["bundle_id"].astype(str)).sum()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def select(pool: pd.DataFrame, score: str, keys: np.ndarray, slots: int) -> set:
    frame = pool[["bundle_id", score]].rename(columns={score: "_score"})
    sort_keys = [keys, -frame["_score"].to_numpy(dtype=float)]
    order = np.lexsort(sort_keys)
    return set(frame["bundle_id"].to_numpy()[order[:min(int(slots), len(frame))]])


def compare_sets(a: set, b: set) -> tuple[float, float]:
    union = a | b
    jac = 100.0 * len(a & b) / len(union) if union else float("nan")
    rep = 100.0 * len(a - b) / len(a) if a else float("nan")
    return jac, rep


def selection_rows(name: str, level: str, primary_pool: pd.DataFrame, primary_score: str,
                   alt_pool: pd.DataFrame, alt_score: str, limits: list[dict], seeds: int) -> list[dict]:
    rows = []
    primary_base = ar.base_hashes(primary_pool["bundle_id"])
    alt_base = ar.base_hashes(alt_pool["bundle_id"])
    primary_keys = [ar.tie_keys(primary_base, s) for s in range(seeds)]
    alt_keys = [ar.tie_keys(alt_base, s) for s in range(seeds)]
    for budget in limits:
        slots = int(budget["expert_slots"])
        values = [compare_sets(select(primary_pool, primary_score, pk, slots), select(alt_pool, alt_score, ak, slots))
                  for pk, ak in zip(primary_keys, alt_keys)]
        jac = ar.summarise_values([v[0] for v in values])
        rep = ar.summarise_values([v[1] for v in values])
        n = len(primary_pool)
        k = min(slots, n)
        expected = 100.0 * (k - k * k / n) / k if k and n else float("nan")
        cut = ar.cutoff_ties(pd.DataFrame({"delta_expert": alt_pool[alt_score].to_numpy(dtype=float)}),
                             "regret", slots) if len(alt_pool) else {}
        rows.append({"specification": name, "level": level, "scenario": budget["scenario"], "review_slots": slots,
                     "primary_pool": n, "alternative_pool": len(alt_pool),
                     **{f"jaccard_{s}": x for s, x in jac.items()},
                     **{f"replaced_{s}": x for s, x in rep.items()},
                     "uniform_random_expected_replaced": expected,
                     "alternative_decided_by_score": cut.get("decided_by_score"),
                     "alternative_decided_by_tie_rule": cut.get("decided_by_tie_rule")})
    return rows


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def run(project: Path, seeds: int, output: Path | None = None, only: list[str] | None = None) -> Path:
    import run_matched_comparison as rmc

    project = project.expanduser().resolve()
    sys.path.insert(0, str(project))
    from clarity_v3.bundles import build_order_bundles
    from clarity_v3.config import load_uncertainty_config
    from clarity_v3.uncertainty import MultiEnvelopeActionPolicy, sample_design

    started = time.time()
    print("1/5 Checking source and frozen inputs.", flush=True)
    source_hashes, _ = rmc.source_guard(project)
    corrected = project / "outputs_corrected"
    states = rmc.read_private(corrected, "03_longitudinal_states.csv")
    primary = rmc.read_private(corrected, "04_action_sets.csv")
    saved_bundles = rmc.read_private(corrected, "05_order_bundles.csv")
    final = rmc.read_private(corrected, "06_final_bundle_decisions.csv")
    orders = rmc.read_private(corrected, "01_order_timeline.csv", ["bundle_id"])
    total_orders = int(orders["bundle_id"].nunique())
    envelopes, action_config = load_uncertainty_config(project / "config/uncertainty_design.json")
    threshold = float(action_config.suppress_support_threshold)

    print("2/5 Recomputing the primary specification and checking it against Stage 04.", flush=True)
    primary_design = sample_design(envelopes)
    recomputed = MultiEnvelopeActionPolicy(primary_design, {e.name: e.regret_quantile for e in envelopes},
                                           action_config).decide_dataframe(states)
    check = primary[["signal_id", "pre_capacity_action", "delta_expert"]].merge(
        recomputed[["signal_id", "pre_capacity_action", "delta_expert"]], on="signal_id",
        suffixes=("_saved", "_new"), validate="one_to_one")
    if len(check) != len(primary) or not check.pre_capacity_action_saved.eq(check.pre_capacity_action_new).all():
        raise AssertionError("Recomputed primary actions differ from saved Stage 04")
    a = pd.to_numeric(check.delta_expert_saved, errors="coerce").to_numpy(float)
    b = pd.to_numeric(check.delta_expert_new, errors="coerce").to_numpy(float)
    if not np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True):
        raise AssertionError("Recomputed primary regret scores differ from saved Stage 04")
    saved = saved_bundles.set_index(saved_bundles["bundle_id"].astype(str))["delta_expert"].astype(float)
    if not np.allclose(bundle_delta(primary).reindex(saved.index).to_numpy(), saved.to_numpy(), atol=1e-9):
        raise AssertionError("Vectorised bundle score does not reproduce saved bundles")
    checks = {"primary_stage04_recomputed_exactly": True, "bundle_score_aggregation_matches_saved": True}

    limits = rmc.capacity_limits(final, {"clarity": saved_bundles})
    limits = [b for b in limits if b["scenario"] != "unconstrained_reference"]
    context = ar.bundle_context(primary)
    primary_pool = ar.review_pool(saved_bundles, context)
    primary_actions = primary[["signal_id", "pre_capacity_action"]].rename(columns={"pre_capacity_action": "primary_action"})
    primary_bundle_routes = saved_bundles.set_index(saved_bundles["bundle_id"].astype(str))["pre_capacity_action"]

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = output.resolve() if output else project / ("parameter_sensitivity_" + stamp)
    output.mkdir(parents=True, exist_ok=False)
    aggregate = output / "aggregate"
    aggregate.mkdir()

    signal_rows, ranking_rows, allocation_rows = [], [], []
    specs = specifications(envelopes, action_config)
    if only:
        specs = [s for s in specs if s[0] in only]
    print(f"3/5 Running {len(specs)} uncertainty specifications (each recomputes all signals).", flush=True)
    for index, (name, envs, config) in enumerate(specs, 1):
        t0 = time.time()
        design = sample_design(envs)
        decided = MultiEnvelopeActionPolicy(design, {e.name: e.regret_quantile for e in envs}, config).decide_dataframe(states)
        merged = primary_actions.merge(decided[["signal_id", "pre_capacity_action", "calibration_sensitive", "safety_class"]],
                                       on="signal_id", validate="one_to_one")
        signal_rows.append({
            "specification": name, "design_points": len(design),
            "action_agreement_percent": 100.0 * float(merged.primary_action.eq(merged.pre_capacity_action).mean()),
            "calibration_sensitive_percent": 100.0 * float(booleans(merged.calibration_sensitive.astype(str)).mean()),
            "protected_alert_violations": int((merged.safety_class.eq("must_interrupt") & ~merged.pre_capacity_action.eq("alert")).sum()),
        })
        # Level 2: ranking only (pool and gate fixed).
        ranked = primary_pool.copy()
        ranked["alt_score"] = ranked["bundle_id"].map(bundle_delta(decided)).fillna(0.0)
        for row in selection_rows(name, "ranking_only", primary_pool, "delta_expert", ranked, "alt_score", limits, seeds):
            row["score_spearman"] = spearman(primary_pool["delta_expert"].to_numpy(), ranked["alt_score"].to_numpy())
            ranking_rows.append(row)
        # Level 3: through allocation (routes, pool and gate recomputed; budgets fixed).
        gate = shared_suppression_gate(decided, float(config.suppress_support_threshold))
        bundles = build_order_bundles(decided)
        bundles["bundle_id"] = bundles["bundle_id"].astype(str)
        bundles["bundle_suppression_eligible"] = bundles["bundle_id"].map(gate.groupby(decided["bundle_id"].astype(str)).all())
        alt_pool = ar.review_pool(bundles, context)
        changed_routes = int((bundles.set_index("bundle_id")["pre_capacity_action"]
                              .reindex(primary_bundle_routes.index) != primary_bundle_routes).sum())
        selections = {r["scenario"]: r for r in selection_rows(name, "through_allocation", primary_pool, "delta_expert",
                                                              alt_pool, "delta_expert", limits, seeds)}
        for budget in limits:
            result = allocate_absolute(bundles, budget["frontline_slots"], budget["expert_slots"], "regret",
                                       budget["scenario"], total_orders)
            counts = result["final_action"].value_counts()
            allocation_rows.append({
                **selections[budget["scenario"]],
                "changed_initial_bundle_routes": changed_routes,
                "review_pool_overlap_with_primary": len(set(alt_pool.bundle_id) & set(primary_pool.bundle_id)),
                **{f"final_{a}": int(counts.get(a, 0)) for a in ACTIONS},
                "interruption_overrun": max(0, int(counts.get("alert", 0)) - int(budget["frontline_slots"])),
            })
        print(f"   {index}/{len(specs)} {name}: {time.time() - t0:.0f}s", flush=True)

    print("4/5 Fixed-midpoint comparator (single parameter vector, ranking only).", flush=True)
    workflow = next(e for e in envelopes if e.name == "workflow_balanced")
    vector = midpoint_vector(workflow)
    point = make_point_policy(MultiEnvelopeActionPolicy)(vector).decide_dataframe(states)
    ranked = primary_pool.copy()
    ranked["alt_score"] = ranked["bundle_id"].map(bundle_delta(point)).fillna(0.0)
    for row in selection_rows("fixed_midpoint_workflow", "ranking_only", primary_pool, "delta_expert",
                              ranked, "alt_score", limits, seeds):
        row["score_spearman"] = spearman(primary_pool["delta_expert"].to_numpy(), ranked["alt_score"].to_numpy())
        ranking_rows.append(row)

    print("5/5 Writing aggregate outputs.", flush=True)
    pd.DataFrame(signal_rows).to_csv(aggregate / "exp2_signal_level.csv", index=False)
    pd.DataFrame(ranking_rows).to_csv(aggregate / "exp2_ranking_only.csv", index=False)
    pd.DataFrame(allocation_rows).to_csv(aggregate / "exp2_through_allocation.csv", index=False)
    if any(r["protected_alert_violations"] for r in signal_rows):
        raise AssertionError("A specification violated must-interrupt protection")
    manifest = {
        "experiment": "parameter_sensitivity_allocation", "version": "1.0", "tie_seeds": seeds,
        "fixed_midpoint_vector_workflow_balanced": vector,
        "contracted_ranges_fraction": 0.5,
        "capacity_limits": limits, "checks": checks, "source_hashes": source_hashes,
        "script_hash": sha(Path(__file__)), "python": sys.version.split()[0],
        "numpy": np.__version__, "pandas": pd.__version__, "runtime_seconds": round(time.time() - started),
        "interpretation_boundaries": [
            "Agreement across specifications is computational stability, not clinical validity.",
            "The fixed-midpoint vector is a declared assumption, not an estimate.",
            "Contracted ranges test dependence on range width, not calibration.",
            "Variation across tie keys is algorithmic variability, not a confidence interval.",
        ],
        "clinical_validation": False,
    }
    (aggregate / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print("COMPLETE. Aggregate outputs: " + str(aggregate), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=200, help="Random tie-breakings per comparison")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--only", nargs="*", help="Run only these specification names (for testing)")
    args = parser.parse_args()
    if args.seeds < 2:
        raise SystemExit("Use at least 2 seeds")
    run(args.project, args.seeds, args.output, args.only)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, AssertionError, OSError, KeyError) as error:
        print("STOPPED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
