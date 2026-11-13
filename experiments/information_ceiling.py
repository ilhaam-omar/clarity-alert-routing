"""Information ceiling: how many review choices can ANY model determine from these records?

Part A - the ceiling
  Two review candidates whose concerns have identical recorded states receive identical
  scores under every scoring rule that depends only on those states, whatever its weights
  or method. Such candidates can be separated only by tie-breaking. Candidates are grouped
  into classes of identical bundle signatures (the multiset of the concern states CLARITY's
  priority score reads). At K review slots, a ranking determines its selection only for
  classes ranked wholly above the cut-off, so the largest number of determinable choices is
  the largest sum of whole class sizes that fits within K (a subset-sum bound). This ceiling
  holds for every state-based model, not only CLARITY.

Part B - what would raise it (value of information for prioritisation)
  Each candidate variable from the replayed record that the priority score does not read is
  added to the signature in turn. Reported for each: classes, candidates still
  indistinguishable from another, the new ceiling at each capacity, and (with --markers)
  how much of the variation in later laboratory harm markers the classes separate, raw, by
  chance (label-permutation null) and beyond chance, because more classes separate more by chance.

Signatures
  exact   the values the score reads (adjusted-harm index, evidence support, completeness,
          management status, repeat status, occurrence index as capped by the repeat
          penalty, safety class and admissible responses)
  coarse  the same with completeness rounded to 0.1 (a sensitivity check)

Usage
  python experiments/information_ceiling.py --project . --outputs outputs_corrected \
      [--markers lab_harm_markers_<date>/private/bundle_markers.csv]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCENARIOS = ("low_capacity", "medium_capacity", "high_capacity")
MASK = 11
PERMUTATIONS = 500


def tf(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "t"])


def management_class(v: str) -> str:
    v = str(v)
    if v in {"transition_confirmed", "protocol_supported_with_observed_monitoring", "intent_documented"}:
        return "managed"
    if v.startswith("protocol_possible"):
        return "partly_managed"
    return "unmanaged"


def repeat_level(status: str, k) -> str:
    if str(status) != "repeat_unchanged_state":
        return str(status)
    try:
        k = float(k)
    except (TypeError, ValueError):
        return "repeat_unchanged_k_na"
    return f"repeat_unchanged_k{int(min(k, 9))}"     # penalty is capped at log1p(9)/log(10) = 1


def concern_signature(d: pd.DataFrame, coarse: bool) -> pd.Series:
    comp = pd.to_numeric(d["evidence_completeness_index"], errors="coerce")
    comp = comp.round(1) if coarse else comp.round(6)
    parts = [
        pd.to_numeric(d["contextual_harm_index"], errors="coerce").round(6).astype(str),
        pd.to_numeric(d["evidence_support_index"], errors="coerce").round(6).astype(str),
        comp.astype(str),
        d["management_status"].map(management_class),
        [repeat_level(s, k) for s, k in zip(d["repeat_status"], d["episode_occurrence_index"])],
        d["safety_class"].astype(str),
        d["admissible_actions"].astype(str),
    ]
    return pd.Series(["|".join(map(str, t)) for t in zip(*parts)], index=d.index)


def binned(name: str, s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    edges = {
        "current_creatinine": [-np.inf, 1.0, 1.5, 2.0, 3.0, np.inf],
        "hours_since_current_creatinine": [-np.inf, 12, 24, 48, 96, np.inf],
        "hours_since_prior_same_signal": [-np.inf, 24, 72, 168, np.inf],
        "days_since_transplant": [-np.inf, 0, 7, 30, 90, 365, np.inf],
        "prior_lifetime_occurrences": [-np.inf, 0, 1, 4, np.inf],
        "prior_admission_occurrences": [-np.inf, 0, 1, 4, np.inf],
    }.get(name)
    if edges is None:
        return s.astype(str)
    out = pd.cut(x, edges).astype(str)
    return out.where(x.notna(), "missing")


EXTRA = ["rule_id", "rule_family", "current_creatinine", "hours_since_current_creatinine",
         "acute_renal_deterioration", "crrt_active", "transplant_phase", "days_since_transplant",
         "protocol_fit", "monitoring_status", "transition_status", "material_change_since_prior",
         "prior_lifetime_occurrences", "prior_admission_occurrences", "hours_since_prior_same_signal"]


def bundle_signature(decisions: pd.DataFrame, pool_ids: set, concern_sig: pd.Series) -> pd.Series:
    d = pd.DataFrame({"bundle_id": decisions["bundle_id"].astype(str), "sig": concern_sig})
    d = d[d["bundle_id"].isin(pool_ids)]
    return d.groupby("bundle_id")["sig"].agg(lambda s: "||".join(sorted(s)))


def ceiling(sizes: np.ndarray, k: int) -> int:
    """Largest sum of whole class sizes not exceeding k (bitset subset sum)."""
    reach, mask = 1, (1 << (k + 1)) - 1
    for n in sizes:
        if n <= k:
            reach = (reach | (reach << int(n))) & mask
    return reach.bit_length() - 1


def summarise(label: str, sig: pd.Series, slots: dict, markers: pd.Series | None) -> dict:
    sizes = sig.value_counts()
    n = len(sig)
    row = {"signature": label, "candidates": n, "classes": len(sizes),
           "candidates_in_shared_classes": int(sizes[sizes > 1].sum()),
           "shared_percent": round(100 * sizes[sizes > 1].sum() / max(n, 1), 1),
           "largest_class": int(sizes.max()) if len(sizes) else 0,
           "singleton_classes": int((sizes == 1).sum())}
    for sc, k in slots.items():
        c = ceiling(sizes.to_numpy(), k)
        row[f"{sc}_slots"] = k
        row[f"{sc}_ceiling"] = c
        row[f"{sc}_ceiling_percent_of_slots"] = round(100 * c / max(k, 1), 1)
    if markers is not None:
        y = pd.Series(sig.index, index=sig.index).map(markers).astype(float).to_numpy()
        ok = ~np.isnan(y)
        if ok.sum() > 0:
            codes = pd.factorize(sig.values[ok])[0]
            yy = y[ok]
            def eta2(v):
                k = codes.max() + 1
                cnt = np.bincount(codes, minlength=k)
                means = np.bincount(codes, weights=v, minlength=k) / np.maximum(cnt, 1)
                grand = v.mean()
                ssb = float((cnt * (means - grand) ** 2).sum())
                sst = float(((v - grand) ** 2).sum())
                return ssb / sst if sst > 0 else np.nan, ssb, sst, k
            raw, ssb, sst, k = eta2(yy)
            n_ok = len(yy)
            msw = (sst - ssb) / max(n_ok - k, 1)
            omega = (ssb - (k - 1) * msw) / (sst + msw) if sst > 0 else np.nan
            rng = np.random.default_rng(11)
            null = np.array([eta2(rng.permutation(yy))[0] for _ in range(PERMUTATIONS)])
            row["harm_variation_separated_percent"] = round(100 * raw, 1)
            row["harm_separated_by_chance_percent"] = round(100 * float(np.mean(null)), 1)
            row["harm_separated_beyond_chance_percent"] = round(100 * (raw - float(np.mean(null))), 1)
            row["harm_separated_adjusted_omega2_percent"] = round(100 * omega, 1)
            row["harm_permutation_p"] = round(float((np.sum(null >= raw) + 1) / (PERMUTATIONS + 1)), 4)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--outputs", default="outputs_corrected")
    ap.add_argument("--markers", type=Path, default=None)
    a = ap.parse_args()
    project = a.project.expanduser().resolve()
    private = project / a.outputs / "private"
    decisions = pd.read_csv(private / "04_action_sets.csv", low_memory=False)
    bundles = pd.read_csv(private / "05_order_bundles.csv", low_memory=False)
    final = pd.read_csv(private / "06_final_bundle_decisions.csv", low_memory=False)
    slots = {}
    for sc in SCENARIOS:
        part = final[final["scenario"].eq(sc)]
        if len(part):
            slots[sc] = int(part["expert_slots"].iloc[0])
    pool = bundles[bundles["pre_capacity_action"].astype(str).eq("escalate") & ~tf(bundles["protected_alert"])]
    pool_ids = set(pool["bundle_id"].astype(str))
    markers = None
    if a.markers:
        mk = pd.read_csv(a.markers, dtype={"bundle_id": str})
        markers = mk.set_index("bundle_id")["any_marker"].astype(str).str.lower().isin(["true", "1"]).astype(float)

    out = project / ("information_ceiling_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True)
    rows = []
    base_exact = concern_signature(decisions, coarse=False)
    base_coarse = concern_signature(decisions, coarse=True)
    rows.append(summarise("model_state_exact", bundle_signature(decisions, pool_ids, base_exact), slots, markers))
    rows.append(summarise("model_state_coarse", bundle_signature(decisions, pool_ids, base_coarse), slots, markers))
    for var in EXTRA:
        if var not in decisions.columns:
            print("skipped (not in record):", var)
            continue
        sig = base_coarse + "#" + binned(var, decisions[var]).astype(str)
        rows.append(summarise("coarse_plus_" + var, bundle_signature(decisions, pool_ids, sig), slots, markers))
    table = pd.DataFrame(rows)
    # class-size distribution for the main signature (aggregate only)
    sizes = bundle_signature(decisions, pool_ids, base_exact).value_counts()
    dist = sizes.value_counts().sort_index().rename_axis("class_size").reset_index(name="classes")
    dist["candidates"] = dist["class_size"] * dist["classes"]
    small = dist["classes"].between(1, MASK - 1)
    dist["classes"] = dist["classes"].astype(object)
    dist.loc[small, "classes"] = f"<{MASK}"
    table.to_csv(out / "information_ceiling_summary.csv", index=False)
    dist.to_csv(out / "information_ceiling_class_sizes.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "candidates": len(pool_ids), "slots": slots,
        "definition": "Ceiling at K = largest sum of whole signature-class sizes <= K. It bounds the review "
                      "choices any scoring rule that depends only on the signature can determine without tie-breaking.",
        "small_cell_rule": "class counts 1-10 masked in the class-size table"}, indent=2))
    pd.set_option("display.width", 250)
    print(table.set_index("signature").T.to_string())
    print("Saved:", out)


if __name__ == "__main__":
    main()
