"""Synthetic checks for the robust-tier classification."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import robust_tier as rt  # noqa: E402


class RankBoundTests(unittest.TestCase):
    def test_bounds_with_ties(self):
        best, worst = rt.rank_bounds(np.array([5.0, 3.0, 3.0, 3.0, 1.0]))
        self.assertEqual(best.tolist(), [1, 2, 2, 2, 5])
        self.assertEqual(worst.tolist(), [1, 4, 4, 4, 5])

    def test_classification_single_spec(self):
        best, worst = rt.rank_bounds(np.array([5.0, 3.0, 3.0, 3.0, 1.0]))
        status = rt.classify(best, worst, 2).tolist()
        self.assertEqual(status, ["robustly_selected", "undetermined", "undetermined", "undetermined",
                                  "robustly_excluded"])
        self.assertEqual(rt.classify(best, worst, 4).tolist()[:4], ["robustly_selected"] * 4)

    def test_set_bounds_widen_with_disagreeing_specs(self):
        frame = pd.DataFrame({"bundle_id": list("abcd"), "p": [4.0, 3.0, 2.0, 1.0], "q": [3.0, 4.0, 2.0, 1.0]})
        best, worst = rt.set_bounds(frame, ["p", "q"])
        status = rt.classify(best, worst, 1).tolist()
        self.assertEqual(status, ["undetermined", "undetermined", "robustly_excluded", "robustly_excluded"])
        self.assertEqual(rt.classify(best, worst, 2).tolist()[:2], ["robustly_selected"] * 2)

    def test_curve_matches_classification(self):
        rng = np.random.default_rng(1)
        frame = pd.DataFrame({"bundle_id": [f"b{i}" for i in range(60)],
                              "p": rng.integers(0, 8, 60).astype(float), "q": rng.integers(0, 8, 60).astype(float)})
        best, worst = rt.set_bounds(frame, ["p", "q"])
        table = rt.curve(best, worst).set_index("review_slots")
        for k in (0, 7, 30, 60):
            counts = pd.Series(rt.classify(best, worst, k)).value_counts()
            self.assertEqual(table.loc[k, "robustly_selected"], counts.get("robustly_selected", 0))
            self.assertEqual(table.loc[k, "robustly_excluded"], counts.get("robustly_excluded", 0))

    def test_random_tiebreaks_never_violate_classification(self):
        rng = np.random.default_rng(2)
        frame = pd.DataFrame({"bundle_id": [f"b{i}" for i in range(80)],
                              "p": rng.integers(0, 5, 80).astype(float), "q": rng.integers(0, 5, 80).astype(float)})
        best, worst = rt.set_bounds(frame, ["p", "q"])
        for k in (10, 40, 70):
            self.assertTrue(rt.simulate_check(frame, ["p", "q"], k, rt.classify(best, worst, k), seeds=30))

    def test_small_counts_are_masked(self):
        self.assertEqual(rt.masked(0), 0)
        self.assertEqual(rt.masked(5), "<11")
        self.assertEqual(rt.masked(11), 11)


if __name__ == "__main__":
    unittest.main()
