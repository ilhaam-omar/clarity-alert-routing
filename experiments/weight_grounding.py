"""Weight-grounding experiment: do the conclusions depend on the declared weight ranges?

Reruns response selection (Stage 04 engine, unchanged) under alternative weight
designs and compares each with the primary run on every quantity the paper
reports as a conclusion:

  1. concern responses          agreement with the primary response per concern
  2. feasibility                protected bundles and the Eq. (7) interruption minimum
  3. review demand and pool     bundles routed to review, and the review pool
  4. review ranking             review-set replacement and the robust core
  5. suppression                bundles suppressed without capacity limits

Weight designs
  primary              the three viewpoints of config/uncertainty_design.json (the grounded,
                       judgement-defined design in the final analysis)
  declared_*           the earlier declared-range design (config/uncertainty_design_declared_ranges.json),
                       as declared, with the three ranges merged, and with every range widened by 50%
  definitional_domain  the primary domain with the reference-case judgements removed
  one_envelope         union of the three ranges, no viewpoints
  widened_50pct        every range widened by 50% around its midpoint
  definitional_domain  no declared ranges: shares in [0, 1], every other weight in
                       [0, UPPER], only the clinical ordering constraints
  anchored_viewpoints  definitional domain, filtered by anchor-case judgments:
                         core (all viewpoints)
                           A1 serious (h=0.8), well-supported (q=0.9), new concern:
                              interruption or review is preferred to passive notification
                           A2 low-severity (h=0.3), well-supported, unchanged third
                              repeat: passive notification or suppression is
                              preferred to interruption
                         disputed judgment, which separates the viewpoints
                           A3 moderate (h=0.5), well-supported, new concern
                              clinical safety : interruption or review preferred
                              alert burden    : passive notification preferred
                              balanced        : no judgment (core only)
                       The weight ranges of each viewpoint are then an output,
                       reported in weight_grounding_anchored_ranges.csv.

Usage
  python experiments/weight_grounding.py --project /path/to/clarity-ress-reproducibility \
      [--outputs outputs_corrected] [--scores robust_tier_*/private/specification_scores.csv] \
      [--designs primary,declared_ranges,declared_one_envelope,declared_widened_50pct,definitional_domain]

Results are computed decisions on replayed records, not clinical effects.
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

NAMES = ("clinical_safety_priority", "workflow_balanced", "alert_burden_priority")
SHARES = ("rho_passive", "rho_alert", "rho_escalate", "uncertainty_alert_fraction",
          "uncertainty_escalate_fraction", "managed_residual_fraction", "partly_managed_residual_fraction")
OTHERS = ("burden_passive", "burden_alert", "burden_escalate", "uncertainty_penalty",
          "repeat_alert_penalty", "escalation_delay_penalty")
UPPER = 1.0          # no response weighs more than the full harm of the most serious certain concern
N_POINTS = 1024      # overwritten by the n_points in config/uncertainty_design.json at run time
SCENARIOS = ("low_capacity", "medium_capacity", "high_capacity")
SEEDS = 20           # tie-breaking seeds for review-set replacement


# ---------------------------------------------------------------------------
# Weight designs
# ---------------------------------------------------------------------------
def enforce_orders(c: dict) -> dict:
    """The same structural orders as clarity_v3.uncertainty.sample_envelope."""
    c["rho_alert"] = np.minimum(c["rho_alert"], c["rho_passive"] - 1e-6)
    c["rho_escalate"] = np.minimum(c["rho_escalate"], c["rho_passive"] - 1e-6)
    c["burden_alert"] = np.maximum(c["burden_alert"], c["burden_passive"] + 1e-6)
    c["uncertainty_escalate_fraction"] = np.minimum(c["uncertainty_escalate_fraction"], c["uncertainty_alert_fraction"])
    c["partly_managed_residual_fraction"] = np.maximum(c["partly_managed_residual_fraction"], c["managed_residual_fraction"])
    return c


def domain_bounds(name: str) -> tuple[float, float]:
    return (0.0, 1.0) if name in SHARES else (0.0, UPPER)


def ranged_envelopes(envelopes, how: str):
    out = []
    if how == "one_envelope":
        union = {k: (min(e.ranges[k][0] for e in envelopes), max(e.ranges[k][1] for e in envelopes))
                 for k in envelopes[0].ranges}
        for e in envelopes:
            out.append(replace(e, ranges=union))
    elif how == "widened_50pct":
        for e in envelopes:
            r = {}
            for k, (lo, hi) in e.ranges.items():
                mid, half = (lo + hi) / 2, 0.75 * (hi - lo)
                dlo, dhi = domain_bounds(k)
                r[k] = (max(dlo, mid - half), min(max(dhi, hi), mid + half))
            out.append(replace(e, ranges=r))
    elif how == "definitional_domain":
        dom = {k: domain_bounds(k) for k in envelopes[0].ranges}
        for e in envelopes:
            out.append(replace(e, ranges=dom))
    else:
        raise ValueError(how)
    return out


DECLARED_CONFIG = "config/uncertainty_design_declared_ranges.json"
DEFAULT_DESIGNS = "primary,declared_ranges,declared_one_envelope,declared_widened_50pct,definitional_domain"


def design_for(name: str, envelopes, project: Path):
    """Design points for a named weight design (primary = config/uncertainty_design.json)."""
    from clarity_v3.config import load_uncertainty_config
    from clarity_v3.uncertainty import sample_design
    n = int(envelopes[0].n_points)
    grounded = bool(getattr(envelopes[0], "anchor_rules", ()))
    if name == "primary":
        return sample_design(envelopes)
    if name.startswith("declared"):
        declared, _ = load_uncertainty_config(project / DECLARED_CONFIG)
        declared = [replace(e, n_points=n) for e in declared]
        if name == "declared_ranges":
            return sample_design(declared)
        return sample_design(ranged_envelopes(declared, name.replace("declared_", "")))
    if name == "definitional_domain":
        if grounded:
            return sample_design([replace(e, anchor_rules=()) for e in envelopes])
        return sample_design(ranged_envelopes(envelopes, "definitional_domain"))
    if name == "anchored_viewpoints":
        return anchored_design()[0]
    return sample_design(ranged_envelopes(envelopes, name))


def grounded_ranges(design: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for env, g in design.groupby("envelope"):
        for k in SHARES + OTHERS:
            v = g[k].to_numpy()
            rows.append({"viewpoint": env, "parameter": k, "p05": round(float(np.percentile(v, 5)), 3),
                         "median": round(float(np.median(v)), 3), "p95": round(float(np.percentile(v, 95)), 3)})
    return pd.DataFrame(rows)


def acceptance_shares(envelopes, candidates: int = 200_000) -> pd.DataFrame:
    from clarity_v3.uncertainty import anchor_mask, _draw
    rows = []
    for e in envelopes:
        rules = getattr(e, "anchor_rules", ())
        if not rules:
            continue
        cand = _draw(e, candidates, 7)
        rows.append({"viewpoint": e.name, "rules": "; ".join("/".join(r) for r in rules),
                     "share_of_domain_satisfying_rules_percent": round(100 * float(anchor_mask(cand, rules).mean()), 2)})
    return pd.DataFrame(rows)


def anchor_costs(c: dict, h: float, q: float, repeat: float) -> dict:
    """Engine cost function (uncertainty.py, _costs) for one reference concern, no management credit."""
    u = (1 - q) * h
    lam = c["uncertainty_penalty"]
    return {
        "suppress": q * h + lam * u,
        "passive": q * c["rho_passive"] * h + lam * u + c["burden_passive"],
        "alert": q * c["rho_alert"] * h + lam * c["uncertainty_alert_fraction"] * u + c["burden_alert"]
                 + c["repeat_alert_penalty"] * repeat,
        "escalate": q * c["rho_escalate"] * h + lam * c["uncertainty_escalate_fraction"] * u + c["burden_escalate"]
                    + c["escalation_delay_penalty"] * q * h,
    }


def anchored_design(seed: int = 20261031, candidates: int = 400_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    names = SHARES + OTHERS
    c = {k: rng.uniform(*domain_bounds(k), candidates) for k in names}
    c = enforce_orders(c)
    repeat3 = min(1.0, np.log1p(3) / np.log(10.0))
    a1 = anchor_costs(c, 0.8, 0.9, 0.0)
    a2 = anchor_costs(c, 0.3, 0.9, repeat3)
    a3 = anchor_costs(c, 0.5, 0.9, 0.0)
    core = (np.minimum(a1["alert"], a1["escalate"]) < a1["passive"]) & \
           (np.minimum(a2["passive"], a2["suppress"]) < a2["alert"])
    act3 = np.minimum(a3["alert"], a3["escalate"]) < a3["passive"]
    masks = {"clinical_safety_priority": core & act3,
             "workflow_balanced": core,
             "alert_burden_priority": core & ~act3}
    frames, summary = [], []
    frame_all = pd.DataFrame(c)
    for i, (name, mask) in enumerate(masks.items()):
        idx = np.flatnonzero(mask)
        if len(idx) < 32:
            raise ValueError(f"Too few accepted weight vectors for {name}: {len(idx)}")
        pick = np.random.default_rng(seed + i).choice(idx, size=min(N_POINTS, len(idx)), replace=False)
        f = frame_all.iloc[pick].reset_index(drop=True)
        f.insert(0, "parameter_id", np.arange(len(f)))
        f.insert(0, "envelope", name)
        frames.append(f)
        for k in names:
            v = frame_all[k].to_numpy()[idx]
            summary.append({"viewpoint": name, "parameter": k, "accepted_share_percent": 100 * len(idx) / candidates,
                            "p05": np.percentile(v, 5), "median": np.median(v), "p95": np.percentile(v, 95)})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summary)


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------
def bundles_of(decided: pd.DataFrame) -> pd.DataFrame:
    from clarity_v3.bundles import build_order_bundles
    b = build_order_bundles(decided)
    b["bundle_id"] = b["bundle_id"].astype(str)
    return b


def as_bool(s: pd.Series) -> pd.Series:
    return s.map(lambda v: str(v).strip().lower() in {"1", "true", "yes", "t"} if not isinstance(v, bool) else v)


def measures(name, decided, bundles, primary_decided, primary_bundles, slots, extra_scores, primary_pool):
    row = {"design": name}
    m = primary_decided[["signal_id", "pre_capacity_action"]].merge(
        decided[["signal_id", "pre_capacity_action"]], on="signal_id", suffixes=("_p", "_d"), validate="one_to_one")
    row["concern_agreement_percent"] = 100 * m["pre_capacity_action_p"].eq(m["pre_capacity_action_d"]).mean()
    row["calibration_sensitive_percent"] = 100 * as_bool(decided["calibration_sensitive"]).mean()
    route = bundles["pre_capacity_action"].value_counts()
    for a in ("suppress", "passive", "alert", "escalate"):
        row[f"bundles_{a}"] = int(route.get(a, 0))
    pa, pr = as_bool(bundles["protected_alert"]), as_bool(bundles["protected_review"])
    must = int(pa.sum())
    prot_review = int((pr & ~pa).sum())
    row["must_interrupt_bundles"] = must
    row["protected_review_or_interrupt_bundles"] = prot_review
    row["protected_total"] = must + prot_review
    pool_mask = bundles["pre_capacity_action"].eq("escalate") & ~pa
    pool_ids = set(bundles.loc[pool_mask, "bundle_id"])
    row["review_pool"] = len(pool_ids)
    row["review_pool_same_as_primary_percent"] = 100 * len(pool_ids & set(primary_pool["bundle_id"])) / max(1, len(primary_pool))
    for sc, (F, K) in slots.items():
        row[f"{sc}_min_interruptions_eq7"] = must + max(0, prot_review - K)
        row[f"{sc}_overrun"] = must + max(0, prot_review - K) - F
    # review ranking on the primary candidate pool (as in robust_tier.py)
    score = psa.bundle_delta(decided)
    pool = primary_pool[["bundle_id", "primary"]].copy()
    pool["design"] = pool["bundle_id"].map(score).fillna(0.0)
    base = ar.base_hashes(pool["bundle_id"])
    for sc, (F, K) in slots.items():
        reps = []
        for s in range(SEEDS):
            keys = ar.tie_keys(base, s)
            a = psa.select(pool, "primary", keys, K)
            b = psa.select(pool, "design", keys, K)
            reps.append(psa.compare_sets(a, b)[1])
        row[f"{sc}_review_set_replacement_percent"] = float(np.mean(reps))
        for label, specs in (("primary_plus_design", ["primary", "design"]),):
            best, worst = set_bounds(pool, specs)
            row[f"{sc}_robust_selected_{label}"] = int((worst <= K).sum())
        if extra_scores is not None:
            full = pool.merge(extra_scores.drop(columns=["primary"], errors="ignore"), on="bundle_id", how="left").fillna(0.0)
            main = [c for c in extra_scores.columns if c not in ("bundle_id", "primary", "fixed_midpoint_workflow")]
            for label, specs in (("paper_main_set", ["primary", *main]), ("paper_main_set_plus_design", ["primary", *main, "design"])):
                best, worst = set_bounds(full, specs)
                row[f"{sc}_robust_selected_{label}"] = int((worst <= K).sum())
    return row


def set_bounds(frame, specs):
    import robust_tier as rt
    return rt.set_bounds(frame, specs)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--outputs", default="outputs_corrected", help="output folder holding private/03-06 CSV files")
    ap.add_argument("--scores", type=Path, default=None, help="robust_tier private/specification_scores.csv (optional)")
    ap.add_argument("--designs", default=DEFAULT_DESIGNS)
    args = ap.parse_args()
    project = args.project.expanduser().resolve()
    sys.path.insert(0, str(project))
    from clarity_v3.config import load_uncertainty_config
    from clarity_v3.uncertainty import MultiEnvelopeActionPolicy, sample_design

    private = project / args.outputs / "private"
    states = pd.read_csv(private / "03_longitudinal_states.csv", low_memory=False)
    saved = pd.read_csv(private / "04_action_sets.csv", low_memory=False)
    final = pd.read_csv(private / "06_final_bundle_decisions.csv", low_memory=False)
    slots = {}
    for sc in SCENARIOS:
        part = final[final["scenario"].eq(sc)]
        if len(part):
            slots[sc] = (int(part["frontline_slots"].iloc[0]), int(part["expert_slots"].iloc[0]))
    envelopes, action_config = load_uncertainty_config(project / "config" / "uncertainty_design.json")
    global N_POINTS
    N_POINTS = int(envelopes[0].n_points)
    quant = {e.name: e.regret_quantile for e in envelopes}
    extra = pd.read_csv(args.scores, dtype={"bundle_id": str}) if args.scores else None

    out = project / ("weight_grounding_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True)
    designs = [d.strip() for d in args.designs.split(",") if d.strip()]
    if "primary" not in designs:
        designs = ["primary", *designs]

    results, primary_decided, primary_bundles, primary_pool = [], None, None, None
    for name in designs:
        t0 = time.time()
        design = design_for(name, envelopes, project)
        if name == "primary":
            grounded_ranges(design).to_csv(out / "weight_ranges_primary_design.csv", index=False)
            acc = acceptance_shares(envelopes)
            if len(acc):
                acc.to_csv(out / "grounded_rule_acceptance.csv", index=False)
        q = {n: quant.get(n, 0.95) for n in design["envelope"].unique()}
        decided = MultiEnvelopeActionPolicy(design, q, action_config).decide_dataframe(states)
        bundles = bundles_of(decided)
        if name == "primary":
            agree = saved[["signal_id", "pre_capacity_action"]].merge(
                decided[["signal_id", "pre_capacity_action"]], on="signal_id", suffixes=("_s", "_r"))
            share = agree["pre_capacity_action_s"].eq(agree["pre_capacity_action_r"]).mean()
            print(f"Primary rerun reproduces saved Stage 04 responses: {100 * share:.2f}%")
            if share < 0.9999:
                raise AssertionError("Primary rerun does not reproduce the saved decisions; check the project version.")
            primary_decided, primary_bundles = decided, bundles
            pa = as_bool(bundles["protected_alert"])
            primary_pool = bundles.loc[bundles["pre_capacity_action"].eq("escalate") & ~pa, ["bundle_id"]].copy()
            primary_pool["primary"] = primary_pool["bundle_id"].map(psa.bundle_delta(decided)).fillna(0.0)
        results.append(measures(name, decided, bundles, primary_decided, primary_bundles, slots, extra, primary_pool))
        print(f"{name}: {time.time() - t0:.0f}s", flush=True)

    table = pd.DataFrame(results)
    table.to_csv(out / "weight_grounding_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.set_index("design").T.to_string())
    (out / "manifest.json").write_text(json.dumps({
        "designs": designs, "upper_bound_non_share_weights": UPPER, "points_per_viewpoint": N_POINTS,
        "tie_breaking_seeds": SEEDS, "slots": slots,
        "interpretation": "Computed decisions on replayed records. Anchored ranges follow from the stated "
                          "anchor judgments and the definitional domain; they are not calibrated estimates."}, indent=2))
    print("Saved:", out)


if __name__ == "__main__":
    main()
