"""Synthetic checks for tie-aware priority and protection-feasibility analyses."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import allocation_robustness as ar  # noqa: E402
from matched_capacity import allocate_absolute  # noqa: E402


def bundles(n_alert=4, n_ar=1, n_review=30, n_passive=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []

    def add(kind, i, action, **kw):
        subject = 10000000 + (i % 7)
        rows.append({"bundle_id": f"{subject}-{2000 + i}-POE-{kind}{i}", "subject_id": subject,
                     "signal_count": 1 + i % 3, "pre_capacity_action": action,
                     "fallback_non_resource_action": "passive", "bundle_suppression_eligible": False,
                     "delta_frontline": 0.0, **kw})
    i = 0
    for _ in range(n_alert):
        i += 1
        add("A", i, "alert", protected_alert=True, protected_review=False, admissible_actions="alert",
            highest_base_severity=8, delta_expert=0.0)
    for _ in range(n_ar):
        i += 1
        add("R", i, "alert", protected_alert=False, protected_review=True, admissible_actions="alert|escalate",
            highest_base_severity=8, delta_expert=0.5)
    for j in range(n_review):
        i += 1
        add("E", i, "escalate", protected_alert=False, protected_review=True, admissible_actions="alert|escalate",
            highest_base_severity=8, delta_expert=float([0.0, 0.0, 0.3, 0.7, 0.7, 1.2][j % 6]))
    for _ in range(n_passive):
        i += 1
        add("P", i, "passive", protected_alert=False, protected_review=False,
            admissible_actions="passive|alert|escalate", highest_base_severity=5, delta_expert=0.0)
    frame = pd.DataFrame(rows)
    context = pd.DataFrame({"bundle_id": frame.bundle_id,
                            "any_acute_renal_deterioration": rng.integers(0, 2, len(frame)),
                            "any_material_change": rng.integers(0, 2, len(frame))})
    return frame, context


class TieKeyTests(unittest.TestCase):
    def test_keys_reproducible_and_row_order_invariant(self):
        frame, _ = bundles()
        base = ar.base_hashes(frame.bundle_id)
        shuffled = frame.sample(frac=1, random_state=3)
        base_shuffled = ar.base_hashes(shuffled.bundle_id)
        a = dict(zip(frame.bundle_id, ar.tie_keys(base, 5)))
        b = dict(zip(shuffled.bundle_id, ar.tie_keys(base_shuffled, 5)))
        self.assertEqual(a, b)
        self.assertTrue(np.array_equal(ar.tie_keys(base, 5), ar.tie_keys(base, 5)))

    def test_seeds_change_keys(self):
        frame, _ = bundles()
        base = ar.base_hashes(frame.bundle_id)
        self.assertFalse(np.array_equal(ar.tie_keys(base, 0), ar.tie_keys(base, 1)))


class SelectorTests(unittest.TestCase):
    def test_identifier_order_reproduces_allocator(self):
        frame, context = bundles()
        pool = ar.review_pool(frame, context)
        for rule in ("severity", "regret"):
            for k in (0, 5, 12, 30, 40):
                result = allocate_absolute(frame, 3, k, rule, "t", 1000)
                allocated = set(result.loc[result.final_action.eq("escalate"), "bundle_id"])
                fast = set(pool.loc[ar.order_by_identifier(pool, rule)[:min(k, len(pool))], "bundle_id"])
                self.assertEqual(allocated, fast, (rule, k))

    def test_random_order_respects_score_and_is_order_invariant(self):
        frame, context = bundles()
        pool = ar.review_pool(frame, context)
        keys = ar.tie_keys(ar.base_hashes(pool.bundle_id), 9)
        order = ar.order_by_random(pool, "regret", keys)
        scores = pool.delta_expert.to_numpy()[order]
        self.assertTrue(np.all(np.diff(scores) <= 0))
        shuffled = pool.sample(frac=1, random_state=1).reset_index(drop=True)
        keys2 = ar.tie_keys(ar.base_hashes(shuffled.bundle_id), 9)
        order2 = ar.order_by_random(shuffled, "regret", keys2)
        self.assertEqual(list(pool.bundle_id[order]), list(shuffled.bundle_id[order2]))

    def test_cutoff_tie_counts(self):
        frame, context = bundles()
        pool = ar.review_pool(frame, context)
        # Scores per 6 candidates: 0,0,0.3,0.7,0.7,1.2 -> 30 candidates: 5x1.2, 10x0.7, 5x0.3, 10x0.0
        result = ar.cutoff_ties(pool, "regret", 10)
        self.assertEqual(result["decided_by_score"], 5)
        self.assertEqual(result["decided_by_tie_rule"], 5)
        self.assertEqual(result["tie_group_at_cutoff"], 10)
        self.assertTrue(result["cutoff_inside_tie_group"])
        clean = ar.cutoff_ties(pool, "regret", 15)
        self.assertFalse(clean["cutoff_inside_tie_group"])
        self.assertEqual(clean["decided_by_score"], 15)
        self.assertEqual(ar.cutoff_ties(pool, "severity", 10)["decided_by_score"], 0)


class ExperimentTests(unittest.TestCase):
    def test_priority_experiment_outputs(self):
        frame, context = bundles()
        pool = ar.review_pool(frame, context)
        limits = [{"scenario": "s", "frontline_slots": 3, "expert_slots": 10}]
        cut, metrics, freq, chars, disc = ar.experiment_priority_ties(pool, limits, 20, "clarity")
        regret = freq[freq.rule.eq("regret")].iloc[0]
        self.assertEqual(regret.always_selected, 5)       # strictly ranked top scores
        self.assertEqual(regret.never_selected, 15)
        severity = freq[freq.rule.eq("severity")].iloc[0]
        self.assertEqual(severity.always_selected, 0)     # all tied: decided by chance
        same = metrics[(metrics.left.eq("severity_random")) & (metrics.right.eq("uniform_random_random"))].iloc[0]
        self.assertAlmostEqual(same.jaccard_mean, 100.0)  # one severity value => identical to uniform

    def test_feasibility_bounds_match_allocator(self):
        frame, _ = bundles()
        limits = [{"scenario": "s", "frontline_slots": 3, "expert_slots": 10}]
        grid, counts = ar.experiment_feasibility({"clarity": frame}, limits, 1000, [5], [3, 8, 40])
        self.assertTrue(grid.bound_respected.all())
        self.assertTrue(grid.observed_minus_fixed_route_bound.eq(0).all())
        row = counts.iloc[0]
        self.assertEqual(row.must_interrupt, 4)
        self.assertEqual(row.flexible_protected, 31)
        infeasible = grid[grid.interruption_allowance.eq(3)]
        self.assertFalse(infeasible.protection_feasible.any())
        self.assertFalse(infeasible.feasible_at_some_review_capacity.any())


if __name__ == "__main__":
    unittest.main()
