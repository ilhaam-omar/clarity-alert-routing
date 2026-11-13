"""Synthetic checks for laboratory harm-marker definitions."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import lab_harm_markers as lh  # noqa: E402

T0 = pd.Timestamp("2100-01-10 12:00")


def labs(rows):
    return pd.DataFrame([{"subject_id": s, "lab": l, "time": T0 + pd.Timedelta(hours=h), "value": v} for s, l, h, v in rows])


def bundle(subject=1):
    return pd.DataFrame({"bundle_id": [f"b{subject}"], "subject_id": [subject], "timestamp": [T0]})


class MarkerTests(unittest.TestCase):
    def run_one(self, rows):
        return lh.compute_markers(bundle(), labs(rows)).iloc[0]

    def test_aki_by_absolute_rise(self):
        r = self.run_one([(1, "creatinine", -6, 1.0), (1, "creatinine", 24, 1.35)])
        self.assertTrue(r.aki_measured and r.aki)

    def test_aki_by_ratio_within_seven_days(self):
        r = self.run_one([(1, "creatinine", -100, 1.0), (1, "creatinine", -2, 1.2), (1, "creatinine", 120, 1.6)])
        self.assertTrue(r.aki)

    def test_no_aki_for_small_rise(self):
        r = self.run_one([(1, "creatinine", -6, 1.0), (1, "creatinine", 24, 1.2)])
        self.assertTrue(r.aki_measured)
        self.assertFalse(r.aki)

    def test_aki_needs_baseline(self):
        r = self.run_one([(1, "creatinine", 24, 3.0)])
        self.assertTrue(r.aki_measured)
        self.assertFalse(r.aki)

    def test_hyperkalaemia_new_onset_only(self):
        self.assertTrue(self.run_one([(1, "potassium", -5, 5.0), (1, "potassium", 10, 6.2)]).hyperkalaemia)
        self.assertFalse(self.run_one([(1, "potassium", -5, 6.3), (1, "potassium", 10, 6.4)]).hyperkalaemia)
        self.assertFalse(self.run_one([(1, "potassium", 80, 6.4)]).hyperkalaemia)  # outside 72 h

    def test_hypoglycaemia_and_inr_and_tacrolimus(self):
        r = self.run_one([(1, "glucose", 5, 48), (1, "inr", 100, 4.5), (1, "tacrolimus", 30, 17.0)])
        self.assertTrue(r.hypoglycaemia and r.high_inr and r.high_tacrolimus and r.any_marker)

    def test_unmeasured_is_not_an_event(self):
        r = self.run_one([])
        self.assertFalse(r.any_measured or r.any_marker)

    def test_masking(self):
        self.assertEqual(lh.masked(4), "<11")
        self.assertEqual(lh.masked(0), 0)
        self.assertEqual(lh.masked(12), 12)

    def test_item_identification_by_label(self):
        d = pd.DataFrame({"itemid": [50912, 50971, 50822, 50931, 51237, 50986, 99999],
                          "label": ["Creatinine", "Potassium", "Potassium, Whole Blood", "Glucose", "INR(PT)", "Tacrolimus (FK506)", "Potassium"],
                          "fluid": ["Blood", "Blood", "Blood", "Blood", "Blood", "Blood", "Urine"]})
        found = lh.identify_items(d)
        self.assertEqual(found["potassium"], [50822, 50971])
        self.assertEqual(found["tacrolimus"], [50986])
        self.assertEqual(found["inr"], [51237])




class OptionalItemTests(unittest.TestCase):
    def test_tacrolimus_optional_and_any_fluid(self):
        d = pd.DataFrame({"itemid": [50912, 1, 2], "label": ["Creatinine", "tacroFK", "Glucose"], "fluid": ["Blood", "Whole Blood", "Blood"]})
        found = lh.identify_items(d)
        self.assertEqual(found["tacrolimus"], [1])
        self.assertTrue(lh.LAB_RULES["tacrolimus"]["optional"])
        empty = lh.identify_items(d[d.itemid != 1])
        self.assertEqual(empty["tacrolimus"], [])


if __name__ == "__main__":
    unittest.main()
