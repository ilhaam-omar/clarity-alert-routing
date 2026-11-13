"""Tie-aware priority robustness and protection-feasibility analyses.

Experiment 1 (priority and ties)
    Holds initial routing fixed and compares review-priority rules when ties are
    broken by reproducible random keys instead of by bundle identifier. Bundle
    identifiers begin with the MIMIC subject identifier, so identifier ordering
    selects whole patients in identifier order. Random keys depend only on the
    seed and the bundle identifier: they are independent of input row order and
    identical for the same bundle in every policy.

Experiment 3 (protection feasibility)
    Separates interruptions that the encoded protection rules force under any
    priority (protection-only bound) from interruptions added by the current
    policy locking initial routes (fixed-route bound), and checks both against
    the actual allocator over a grid of review capacities.

Every number describes algorithmic allocation behaviour. Variation across
random tie-breakings is algorithmic variability, not a clinical confidence
interval. Nothing here measures clinical appropriateness or patient benefit.

This module adds analyses without changing any fingerprinted source file, so
the frozen matched comparison remains reproducible.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KIT = Path(__file__).resolve().parent
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

from matched_capacity import allocate_absolute  # noqa: E402
from matched_policies import booleans, build_policy_bundles  # noqa: E402

MASK64 = (1 << 64) - 1
SCENARIOS = ["low_capacity", "medium_capacity", "high_capacity", "unconstrained_reference"]

# Score columns are sorted descending; remaining ties use the tie rule.
PRIORITY_RULES = {
    "severity": ["highest_base_severity"],
    "regret": ["delta_expert"],
    "context": ["highest_base_severity", "any_acute_renal_deterioration", "any_material_change"],
    "uniform_random": [],
}


# ---------------------------------------------------------------------------
# Reproducible tie keys
# ---------------------------------------------------------------------------

def base_hashes(bundle_ids: pd.Series) -> np.ndarray:
    """Stable 64-bit hash of each bundle identifier (independent of row order)."""
    values = [
        int.from_bytes(hashlib.blake2b(str(b).encode("utf-8"), digest_size=8).digest(), "little")
        for b in bundle_ids
    ]
    return np.array(values, dtype=np.uint64)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64, copy=True)
    x += np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def tie_keys(base: np.ndarray, seed: int) -> np.ndarray:
    """Random key per bundle for one seed; identical across policies."""
    salt = np.uint64((int(seed) * 0xD1B54A32D192ED03 + 0x632BE59BD9B4E019) & MASK64)
    return _splitmix64(base ^ salt)


# ---------------------------------------------------------------------------
# Candidate pools and ordering
# ---------------------------------------------------------------------------

def bundle_context(decisions: pd.DataFrame) -> pd.DataFrame:
    """Any-occurrence aggregation of contextual fields per bundle."""
    frame = pd.DataFrame({
        "bundle_id": decisions["bundle_id"].astype(str),
        "any_acute_renal_deterioration": booleans(decisions["acute_renal_deterioration"]).astype(int),
        "any_material_change": booleans(decisions["material_change_since_prior"]).astype(int),
    })
    return frame.groupby("bundle_id", as_index=False).max()


def review_pool(bundles: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Bundles that compete for review slots, exactly as in allocate_absolute."""
    working = bundles.copy()
    working["bundle_id"] = working["bundle_id"].astype(str)
    protected_alert = booleans(working["protected_alert"])
    mask = working["pre_capacity_action"].astype(str).eq("escalate") & ~protected_alert
    pool = working.loc[mask].copy()
    pool["delta_expert"] = pd.to_numeric(pool["delta_expert"], errors="coerce").fillna(0.0)
    pool["highest_base_severity"] = pd.to_numeric(pool["highest_base_severity"], errors="raise")
    pool = pool.merge(context, on="bundle_id", how="left", validate="one_to_one")
    for col in ("any_acute_renal_deterioration", "any_material_change"):
        if pool[col].isna().any():
            raise AssertionError("Context missing for a review candidate: " + col)
        pool[col] = pool[col].astype(int)
    return pool.reset_index(drop=True)


def order_by_identifier(pool: pd.DataFrame, rule: str) -> np.ndarray:
    """Current deterministic convention: score descending, then bundle_id ascending."""
    cols = PRIORITY_RULES[rule]
    frame = pool.sort_values(cols + ["bundle_id"], ascending=[False] * len(cols) + [True], kind="stable")
    return frame.index.to_numpy()


def order_by_random(pool: pd.DataFrame, rule: str, keys: np.ndarray) -> np.ndarray:
    """Score descending, then random key ascending. Returns pool positions."""
    sort_keys = [keys]
    for col in reversed(PRIORITY_RULES[rule]):
        sort_keys.append(-pool[col].to_numpy(dtype=float))
    return np.lexsort(sort_keys)


def cutoff_ties(pool: pd.DataFrame, rule: str, slots: int) -> dict:
    """How many selections the score decides and how many the tie rule decides."""
    n = len(pool)
    k = min(int(slots), n)
    cols = PRIORITY_RULES[rule]
    if k == 0 or k == n:
        return {"selected": k, "decided_by_score": k, "decided_by_tie_rule": 0,
                "tie_group_at_cutoff": 0, "cutoff_inside_tie_group": False}
    if not cols:
        return {"selected": k, "decided_by_score": 0, "decided_by_tie_rule": k,
                "tie_group_at_cutoff": n, "cutoff_inside_tie_group": True}
    scores = pool[cols].to_numpy(dtype=float)
    ordered = scores[np.lexsort([-scores[:, i] for i in reversed(range(scores.shape[1]))])]
    boundary = ordered[k - 1]
    equal = np.all(scores == boundary, axis=1)
    above = np.zeros(n, dtype=bool)
    for i in range(scores.shape[1]):
        prefix_equal = np.all(scores[:, :i] == boundary[:i], axis=1) if i else np.ones(n, dtype=bool)
        above |= prefix_equal & (scores[:, i] > boundary[i])
    n_above = int(above.sum())
    tie_group = int(equal.sum())
    from_tie_group = k - n_above
    inside = 0 < from_tie_group < tie_group
    by_tie = from_tie_group if inside else 0
    return {"selected": k, "decided_by_score": k - by_tie, "decided_by_tie_rule": by_tie,
            "tie_group_at_cutoff": tie_group, "cutoff_inside_tie_group": bool(inside)}


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return 100.0 * inter / union if union else float("nan")


def replaced_percent(reference: np.ndarray, other: np.ndarray) -> float:
    size = int(np.count_nonzero(reference))
    return 100.0 * int(np.count_nonzero(reference & ~other)) / size if size else float("nan")


def summarise_values(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "p2_5": float(np.percentile(arr, 2.5)), "p97_5": float(np.percentile(arr, 97.5)),
            "min": float(arr.min()), "max": float(arr.max())}


# ---------------------------------------------------------------------------
# Experiment 1
# ---------------------------------------------------------------------------

def experiment_priority_ties(pool: pd.DataFrame, limits: list[dict], replicates: int, routing: str):
    """Compare review-priority rules under reproducible random tie-breaking."""
    n = len(pool)
    base = base_hashes(pool["bundle_id"])
    subjects = pool["subject_id"].astype(str).to_numpy()
    chars = {
        "acute_renal_deterioration_percent": pool["any_acute_renal_deterioration"].to_numpy(float),
        "material_change_percent": pool["any_material_change"].to_numpy(float),
        "mean_signal_count": pd.to_numeric(pool["signal_count"], errors="coerce").to_numpy(float),
        "mean_regret_score": pool["delta_expert"].to_numpy(float),
    }
    rules = list(PRIORITY_RULES)
    reference_orders = {f"{r}_identifier": order_by_identifier(pool, r) for r in ("severity", "regret", "context")}

    discrimination = []
    for col in ("highest_base_severity", "any_acute_renal_deterioration", "any_material_change"):
        counts = pool[col].value_counts().sort_index()
        for value, count in counts.items():
            discrimination.append({"routing": routing, "field": col, "value": value, "candidates": int(count)})
    discrimination.append({"routing": routing, "field": "delta_expert_distinct_values",
                           "value": int(pool["delta_expert"].nunique()), "candidates": n})

    cutoff_rows, metric_rows, freq_rows, char_rows = [], [], [], []
    pair_specs = [("regret", "severity"), ("context", "severity"), ("regret", "context"),
                  ("severity", "uniform_random"), ("regret", "uniform_random"), ("context", "uniform_random")]

    for budget in limits:
        scenario, slots = budget["scenario"], int(budget["expert_slots"])
        k = min(slots, n)
        for rule in rules:
            cutoff_rows.append({"routing": routing, "scenario": scenario, "rule": rule,
                                "review_candidates": n, **cutoff_ties(pool, rule, slots)})

        def mask_from(order):
            m = np.zeros(n, dtype=bool)
            m[order[:k]] = True
            return m

        def describe(m):
            if not m.any():
                return {c: float("nan") for c in chars}
            return {c: float(100 * v[m].mean()) if "percent" in c else float(np.nanmean(v[m]))
                    for c, v in chars.items()}

        refs = {name: mask_from(order) for name, order in reference_orders.items()}
        for name, m in refs.items():
            char_rows.append({"routing": routing, "scenario": scenario, "selection": name, "statistic": "single_run",
                              "distinct_patients": int(len(set(subjects[m]))), **describe(m)})

        counts = {r: np.zeros(n, dtype=np.int64) for r in rules}
        collected = {key: [] for key in
                     [f"{r}_random|{ref}" for r in rules for ref in refs] +
                     [f"{a}_random|{b}_random" for a, b in pair_specs] +
                     [f"{r}_random|self_next_seed" for r in rules]}
        patient_counts = {r: [] for r in rules}
        char_values = {r: {c: [] for c in chars} for r in rules}
        previous = {}
        for seed in range(replicates):
            keys = tie_keys(base, seed)
            masks = {r: mask_from(order_by_random(pool, r, keys)) for r in rules}
            for r, m in masks.items():
                counts[r] += m
                patient_counts[r].append(len(set(subjects[m])))
                for c, value in describe(m).items():
                    char_values[r][c].append(value)
                for ref, rm in refs.items():
                    collected[f"{r}_random|{ref}"].append((jaccard(m, rm), replaced_percent(rm, m)))
                if r in previous:
                    collected[f"{r}_random|self_next_seed"].append((jaccard(m, previous[r]), replaced_percent(previous[r], m)))
            for a, b in pair_specs:
                collected[f"{a}_random|{b}_random"].append((jaccard(masks[a], masks[b]), replaced_percent(masks[b], masks[a])))
            previous = masks

        expected_overlap = k * k / n if n else float("nan")
        expected_jaccard = 100 * expected_overlap / (2 * k - expected_overlap) if k else float("nan")
        expected_replaced = 100 * (k - expected_overlap) / k if k else float("nan")
        for key, values in collected.items():
            if not values:
                continue
            left, right = key.split("|")
            jac = summarise_values([v[0] for v in values])
            rep = summarise_values([v[1] for v in values])
            metric_rows.append({"routing": routing, "scenario": scenario, "review_slots": slots,
                                "left": left, "right": right, "replicates": len(values),
                                **{f"jaccard_{s}": x for s, x in jac.items()},
                                **{f"replaced_{s}": x for s, x in rep.items()},
                                "uniform_random_expected_jaccard": expected_jaccard,
                                "uniform_random_expected_replaced": expected_replaced})
        for r in rules:
            freq = counts[r] / float(replicates)
            freq_rows.append({"routing": routing, "scenario": scenario, "rule": r, "review_slots": slots,
                              "candidates": n, "always_selected": int(np.sum(freq == 1.0)),
                              "never_selected": int(np.sum(freq == 0.0)),
                              "sometimes_selected": int(np.sum((freq > 0) & (freq < 1))),
                              "selected_at_least_95pct": int(np.sum(freq >= 0.95)),
                              "selected_at_most_5pct": int(np.sum(freq <= 0.05))})
            char_rows.append({"routing": routing, "scenario": scenario, "selection": f"{r}_random",
                              "statistic": f"mean_of_{replicates}_replicates",
                              "distinct_patients": float(np.mean(patient_counts[r])),
                              **{c: float(np.mean(v)) for c, v in char_values[r].items()}})
    return (pd.DataFrame(cutoff_rows), pd.DataFrame(metric_rows), pd.DataFrame(freq_rows),
            pd.DataFrame(char_rows), pd.DataFrame(discrimination))


# ---------------------------------------------------------------------------
# Experiment 3
# ---------------------------------------------------------------------------

def protection_counts(bundles: pd.DataFrame) -> dict:
    pa = booleans(bundles["protected_alert"])
    pr = booleans(bundles["protected_review"])
    action = bundles["pre_capacity_action"].astype(str)
    return {
        "must_interrupt": int(pa.sum()),
        "flexible_protected": int((pr & ~pa).sum()),
        "locked_initial_alerts": int((pa | (action.eq("alert") & pr & ~pa)).sum()),
        "protected_review_candidates": int((action.eq("escalate") & pr & ~pa).sum()),
        "review_candidates": int((action.eq("escalate") & ~pa).sum()),
        "discretionary_alerts": int((action.eq("alert") & ~pa & ~pr).sum()),
    }


def bounds(counts: dict, frontline: int, review: int) -> dict:
    protection_min = counts["must_interrupt"] + max(0, counts["flexible_protected"] - review)
    fixed_min = counts["locked_initial_alerts"] + max(0, counts["protected_review_candidates"] - review)
    return {
        "protection_only_min_interruptions": protection_min,
        "fixed_route_min_interruptions": fixed_min,
        "policy_added_interruptions": fixed_min - protection_min,
        "protection_feasible": protection_min <= frontline,
        "fixed_route_feasible": fixed_min <= frontline,
        "unavoidable_overrun": max(0, protection_min - frontline),
        "policy_overrun": max(0, fixed_min - frontline),
        "review_slots_unused_by_pool": max(0, review - counts["review_candidates"]),
        "feasible_at_some_review_capacity": counts["must_interrupt"] <= frontline,
        "min_review_slots_for_feasibility": (
            max(0, counts["must_interrupt"] + counts["flexible_protected"] - frontline)
            if counts["must_interrupt"] <= frontline else None),
    }


def experiment_feasibility(policy_bundles: dict, limits: list[dict], total_orders: int,
                           extra_review_grid: list[int], frontline_grid: list[int]):
    rows = []
    count_rows = []
    for routing, bundles in policy_bundles.items():
        counts = protection_counts(bundles)
        count_rows.append({"routing": routing, **counts})
        review_grid = sorted({0, *[int(b["expert_slots"]) for b in limits], counts["flexible_protected"],
                              counts["review_candidates"], *extra_review_grid})
        for frontline in sorted(set(frontline_grid)):
            for review in review_grid:
                result = allocate_absolute(bundles, int(frontline), int(review), "regret", "grid", total_orders)
                observed = int(result["final_action"].eq("alert").sum())
                b = bounds(counts, int(frontline), int(review))
                rows.append({"routing": routing, "interruption_allowance": int(frontline), "review_slots": int(review),
                             "observed_interruptions": observed,
                             "observed_reviews": int(result["final_action"].eq("escalate").sum()), **b,
                             "observed_minus_fixed_route_bound": observed - b["fixed_route_min_interruptions"],
                             "bound_respected": observed >= b["protection_only_min_interruptions"]})
    grid = pd.DataFrame(rows)
    if not grid["bound_respected"].all():
        raise AssertionError("Observed interruptions fell below the protection-only lower bound")
    return grid, pd.DataFrame(count_rows)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def run(project: Path, replicates: int, output: Path | None = None) -> Path:
    import run_matched_comparison as rmc

    project = project.expanduser().resolve()
    print("1/6 Checking source and frozen inputs.", flush=True)
    source_hashes, _ = rmc.source_guard(project)
    corrected = project / "outputs_corrected"
    fields = ["signal_id", "bundle_id", "subject_id", "hadm_id", "timestamp", "base_severity", "certainty_class",
              "safety_class", "admissible_actions", "material_change_since_prior", "acute_renal_deterioration",
              "consensus_action", "action_support_suppress_min", "raw_winner_safety_priority",
              "raw_winner_workflow_balanced", "raw_winner_burden_priority", "pre_capacity_action",
              "fallback_non_resource_action", "protected_alert", "protected_review", "calibration_sensitive",
              "delta_frontline", "delta_expert"]
    decisions = rmc.read_private(corrected, "04_action_sets.csv", fields)
    saved_bundles = rmc.read_private(corrected, "05_order_bundles.csv")
    final = rmc.read_private(corrected, "06_final_bundle_decisions.csv")
    orders = rmc.read_private(corrected, "01_order_timeline.csv", ["bundle_id"])
    total_orders = int(orders["bundle_id"].nunique())

    sys.path.insert(0, str(project))
    from clarity_v3.bundles import build_order_bundles
    threshold = float(rmc.read_json(project / "config/uncertainty_design.json")["action_set"]["suppress_support_threshold"])
    severity = float(rmc.read_json(project / "config/state_builder.json")["high_severity_floor"])

    print("2/6 Rebuilding routing policies and checking them against saved outputs.", flush=True)
    policy_bundles, _, _ = build_policy_bundles(decisions, build_order_bundles, threshold, severity)
    rmc.check_saved_bundles(policy_bundles["clarity"], saved_bundles)
    limits = rmc.capacity_limits(final, policy_bundles)
    context = bundle_context(decisions)

    checks = {}
    print("3/6 Confirming the fast selector reproduces the allocator.", flush=True)
    for routing, bundles in policy_bundles.items():
        pool = review_pool(bundles, context)
        for budget in limits:
            for rule, ranking in (("severity", "severity"), ("regret", "regret")):
                result = allocate_absolute(bundles, budget["frontline_slots"], budget["expert_slots"],
                                           ranking, budget["scenario"], total_orders)
                allocated = set(result.loc[result["final_action"].eq("escalate"), "bundle_id"].astype(str))
                fast = set(pool.loc[order_by_identifier(pool, rule)[:min(budget["expert_slots"], len(pool))], "bundle_id"])
                if allocated != fast:
                    raise AssertionError(f"Fast selector differs from allocator: {routing}/{rule}/{budget['scenario']}")
                if routing == "clarity" and ranking == "regret":
                    saved = final.loc[final["scenario"].eq(budget["scenario"]), ["bundle_id", "final_action"]]
                    merged = result[["bundle_id", "final_action"]].merge(saved, on="bundle_id", suffixes=("_new", "_saved"))
                    if not merged["final_action_new"].eq(merged["final_action_saved"]).all():
                        raise AssertionError("Saved CLARITY routes not replayed in " + budget["scenario"])
        checks[f"{routing}_fast_selector_matches_allocator"] = True
    checks["saved_clarity_routes_replayed"] = True

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = output.resolve() if output else project / ("allocation_robustness_" + stamp)
    output.mkdir(parents=True, exist_ok=False)
    aggregate = output / "aggregate"
    aggregate.mkdir()

    print(f"4/6 Experiment 1: priority rules with {replicates} random tie-breakings.", flush=True)
    frames = {name: [] for name in ("cutoff_ties", "priority_comparisons", "selection_frequency",
                                    "selected_set_characteristics", "context_discrimination")}
    for routing in ("clarity", "static"):
        pool = review_pool(policy_bundles[routing], context)
        results = experiment_priority_ties(pool, limits, replicates, routing)
        for name, frame in zip(frames, results):
            frames[name].append(frame)
    for name, parts in frames.items():
        pd.concat(parts, ignore_index=True).to_csv(aggregate / f"exp1_{name}.csv", index=False)

    print("5/6 Experiment 3: protection feasibility grid (runs the real allocator).", flush=True)
    frontline_grid = [int(b["frontline_slots"]) for b in limits]
    flexible = protection_counts(policy_bundles["clarity"])["flexible_protected"]
    extra = [int(round(flexible * f)) for f in (0.1, 0.4, 0.6, 0.9)]
    grid, counts = experiment_feasibility(policy_bundles, limits, total_orders, extra, frontline_grid)
    grid.to_csv(aggregate / "exp3_feasibility_grid.csv", index=False)
    counts.to_csv(aggregate / "exp3_protection_counts.csv", index=False)
    checks["observed_interruptions_never_below_protection_bound"] = True

    print("6/6 Writing manifest.", flush=True)
    manifest = {
        "experiment": "allocation_robustness",
        "version": "1.0",
        "replicates": replicates,
        "tie_key": "splitmix64(blake2b64(bundle_id) XOR seed salt); independent of row order and shared by all policies",
        "priority_rules": PRIORITY_RULES,
        "context_rule_status": "Explicit experimental assumption (severity, then acute renal deterioration, then material change); not a validated clinical standard.",
        "capacity_limits": limits,
        "checks": checks,
        "source_hashes": source_hashes,
        "script_hash": sha(Path(__file__)),
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "interpretation_boundaries": [
            "Variation across random tie-breakings is algorithmic variability, not a clinical confidence interval.",
            "Overlap and replacement describe which bundles are selected, not which selection is clinically better.",
            "Feasibility bounds are properties of the encoded rules, not validated staffing requirements.",
            "The regret score is the model's own preference; selecting high-regret bundles is not independent validation.",
        ],
        "clinical_validation": False,
    }
    (aggregate / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print("COMPLETE. Aggregate outputs: " + str(aggregate), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True, help="Repository root containing outputs_corrected")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.replicates < 2:
        raise SystemExit("Use at least 2 replicates")
    run(args.project, args.replicates, args.output)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, AssertionError, OSError, KeyError) as error:
        print("STOPPED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
