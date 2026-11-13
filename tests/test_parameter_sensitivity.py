"""Synthetic checks for the Experiment 2 parameter-sensitivity helpers."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import parameter_sensitivity_allocation as psa  # noqa: E402
from clarity_v3.bundles import build_order_bundles  # noqa: E402
from clarity_v3.config import load_uncertainty_config  # noqa: E402
from clarity_v3.uncertainty import MultiEnvelopeActionPolicy, sample_envelope  # noqa: E402

ENVELOPES, ACTION_CONFIG = load_uncertainty_config(ROOT / "config" / "uncertainty_design.json")


class SpecificationTests(unittest.TestCase):
    def test_contracted_ranges_halve_width_around_midpoint(self):
        env = ENVELOPES[0]
        small = psa.contracted(env)
        for name, (low, high) in env.ranges.items():
            a, b = small.ranges[name]
            self.assertAlmostEqual((a + b) / 2, (low + high) / 2)
            self.assertAlmostEqual(b - a, (high - low) / 2)
        self.assertEqual(small.seed, env.seed)
        sample_envelope(small)  # order restrictions still apply without error

    def test_midpoint_vector_respects_order_restrictions(self):
        v = psa.midpoint_vector(next(e for e in ENVELOPES if e.name == "workflow_balanced"))
        self.assertEqual(set(v), set(psa.PARAMETERS))
        self.assertLess(v["rho_alert"], v["rho_passive"])
        self.assertLess(v["rho_escalate"], v["rho_passive"])
        self.assertGreater(v["burden_alert"], v["burden_passive"])
        self.assertLessEqual(v["uncertainty_escalate_fraction"], v["uncertainty_alert_fraction"])
        self.assertGreaterEqual(v["partly_managed_residual_fraction"], v["managed_residual_fraction"])

    def test_archived_specifications_are_all_present(self):
        names = [s[0] for s in psa.specifications(ENVELOPES, ACTION_CONFIG)]
        for expected in ("half_design_points", "double_design_points", "alternate_seeds",
                         "regret_quantile_0.90", "regret_quantile_0.975", "support_threshold_0.60",
                         "support_threshold_0.80", "suppress_threshold_0.90", "contracted_ranges_50pct"):
            self.assertIn(expected, names)


class ScoreTests(unittest.TestCase):
    def decisions(self):
        return pd.DataFrame({
            "bundle_id": ["b1", "b1", "b2", "b3"], "subject_id": [1, 1, 2, 3], "hadm_id": [1, 1, 2, 3],
            "timestamp": ["2100-01-01"] * 4, "signal_id": ["s1", "s2", "s3", "s4"],
            "base_severity": [8, 8, 8, 5], "safety_class": ["must_review_or_interrupt"] * 3 + ["standard"],
            "pre_capacity_action": ["escalate"] * 3 + ["passive"], "fallback_non_resource_action": ["passive"] * 4,
            "protected_alert": [False] * 4, "protected_review": [True] * 3 + [False],
            "calibration_sensitive": [False] * 4, "delta_frontline": [0.0] * 4,
            "delta_expert": [0.4, -0.2, float("-inf"), 0.3],
        })

    def test_bundle_delta_matches_engine_aggregation(self):
        d = self.decisions()
        fast = psa.bundle_delta(d)
        slow = build_order_bundles(d).set_index("bundle_id")["delta_expert"]
        for key in slow.index:
            self.assertAlmostEqual(fast[key], slow[key])
        self.assertAlmostEqual(fast["b1"], 0.4)   # negative difference clipped to zero
        self.assertAlmostEqual(fast["b2"], 0.0)   # non-finite difference dropped

    def test_point_policy_delta_equals_manual_cost_difference(self):
        vector = psa.midpoint_vector(ENVELOPES[1])
        policy = psa.make_point_policy(MultiEnvelopeActionPolicy)(vector)
        row = {"signal_id": "s", "base_severity": 8.0, "evidence_support_index": 0.6,
               "contextual_harm_index": 7.0, "evidence_completeness_index": 0.8, "certainty_class": "low",
               "management_status": "unmanaged", "repeat_status": "repeat_unchanged_state",
               "episode_occurrence_index": 3, "material_change_since_prior": False,
               "admissible_actions": "alert|escalate", "safety_class": "must_review_or_interrupt"}
        out = policy.decide_dataframe(pd.DataFrame([row]))
        harm, support = 0.7, 0.48
        unresolved = (1 - support) * harm
        repeat = min(1.0, np.log1p(3) / np.log(10.0))
        alert = (support * vector["rho_alert"] * harm + vector["uncertainty_penalty"] * vector["uncertainty_alert_fraction"]
                 * unresolved + vector["burden_alert"] + vector["repeat_alert_penalty"] * repeat)
        review = (support * vector["rho_escalate"] * harm + vector["uncertainty_penalty"]
                  * vector["uncertainty_escalate_fraction"] * unresolved + vector["burden_escalate"]
                  + vector["escalation_delay_penalty"] * support * harm)
        self.assertAlmostEqual(float(out["delta_expert"].iloc[0]), alert - review, places=10)


class SelectionTests(unittest.TestCase):
    def test_identical_scores_give_identical_selections(self):
        pool = pd.DataFrame({"bundle_id": [f"b{i}" for i in range(50)], "s": np.repeat([3.0, 2.0, 1.0, 0.0, 0.0], 10)})
        rows = psa.selection_rows("same", "ranking_only", pool, "s", pool.copy(), "s",
                                  [{"scenario": "x", "expert_slots": 25}], 10)
        self.assertAlmostEqual(rows[0]["jaccard_mean"], 100.0)
        self.assertEqual(rows[0]["alternative_decided_by_score"], 20)

    def test_reversed_scores_replace_selection(self):
        pool = pd.DataFrame({"bundle_id": [f"b{i}" for i in range(40)], "s": np.arange(40, dtype=float)})
        alt = pool.assign(s=-pool["s"])
        rows = psa.selection_rows("rev", "ranking_only", pool, "s", alt, "s", [{"scenario": "x", "expert_slots": 10}], 5)
        self.assertAlmostEqual(rows[0]["replaced_mean"], 100.0)
        self.assertAlmostEqual(psa.spearman(pool.s.to_numpy(), alt.s.to_numpy()), -1.0)


if __name__ == "__main__":
    unittest.main()
