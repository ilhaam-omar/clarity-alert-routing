from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from clarity_v3.mimic import build_order_timeline, detect_signals_from_orders
from clarity_v3.reference import ReferenceStore

ROOT = Path(__file__).resolve().parents[1]


class ProviderOrderTimingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = ReferenceStore(ROOT / "reference")
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.cohort = pd.DataFrame({"subject_id":[1],"hadm_id":[10]})
        self.patients = pd.DataFrame({"subject_id":[1],"anchor_age":[50],"anchor_year":[2100],"gender":["M"]})
        self.admissions = pd.DataFrame({"subject_id":[1],"hadm_id":[10],"admittime":["2100-01-01"],"dischtime":["2100-01-10"]})
        self.labs = pd.DataFrame({
            "subject_id":[1,1],"hadm_id":[10,10],"itemid":[50912,50912],
            "charttime":["2100-01-01 09:00","2100-01-01 15:00"],"valuenum":[1.0,9.0]
        })
        self.procedures = pd.DataFrame({"subject_id":[1],"hadm_id":[10],"seq_num":[1],"chartdate":["2100-01-01"],"icd_code":["5569"],"icd_version":[9]})
        self.poe = pd.DataFrame({"poe_id":["P1","P2"],"subject_id":[1,1],"hadm_id":[10,10],"ordertime":["2100-01-01 10:00","2100-01-01 10:00"],"poe_seq":[1,2]})
        for name, frame in {
            "cohort.csv":self.cohort,"patients.csv":self.patients,"admissions.csv":self.admissions,
            "labs.csv":self.labs,"procedures.csv":self.procedures,"poe.csv":self.poe,
        }.items():
            frame.to_csv(self.base/name,index=False)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _build(self, rx: pd.DataFrame, poe: pd.DataFrame | None = None) -> pd.DataFrame:
        rx.to_csv(self.base/"rx.csv", index=False)
        if poe is not None:
            poe.to_csv(self.base/"poe.csv", index=False)
        return build_order_timeline(
            self.base/"cohort.csv", self.base/"patients.csv", self.base/"admissions.csv",
            self.base/"rx.csv", self.base/"labs.csv", self.base/"procedures.csv",
            None, None, None, self.reference, poe_file=self.base/"poe.csv",
            decision_time_mode="poe_ordertime_with_starttime_fallback",
        )

    def test_01_exact_link_uses_ordertime_and_excludes_future_lab(self) -> None:
        rx = pd.DataFrame({"subject_id":[1],"hadm_id":[10],"poe_id":["P1"],"drug":["Fluconazole"],"starttime":["2100-01-01 18:00"],"stoptime":["2100-01-03"],"route":["PO"]})
        timeline = self._build(rx)
        row = timeline.iloc[0]
        self.assertEqual(str(row["decision_time_source"]), "poe_ordertime")
        self.assertEqual(pd.Timestamp(row["timestamp"]), pd.Timestamp("2100-01-01 10:00"))
        self.assertEqual(float(row["current_creatinine"]), 1.0)

    def test_02_same_poe_multiple_starttimes_is_one_bundle_one_decision(self) -> None:
        rx = pd.DataFrame({"subject_id":[1,1],"hadm_id":[10,10],"poe_id":["P1","P1"],"drug":["Tacrolimus","Fluconazole"],"starttime":["2100-01-01 12:00","2100-01-02 12:00"],"stoptime":["2100-01-05","2100-01-05"],"route":["PO","PO"]})
        timeline = self._build(rx)
        self.assertEqual(timeline["bundle_id"].nunique(), 1)
        self.assertEqual(timeline["timestamp"].nunique(), 1)
        self.assertEqual(timeline["exposure_starttime"].nunique(), 2)

    def test_03_admission_mismatch_falls_back_and_stands_alone(self) -> None:
        bad_poe = self.poe.copy()
        bad_poe.loc[bad_poe["poe_id"].eq("P1"), "hadm_id"] = 11
        rx = pd.DataFrame({"subject_id":[1],"hadm_id":[10],"poe_id":["P1"],"drug":["Fluconazole"],"starttime":["2100-01-01 18:00"],"stoptime":["2100-01-03"],"route":["PO"]})
        timeline = self._build(rx, bad_poe)
        row = timeline.iloc[0]
        self.assertEqual(str(row["poe_link_status"]), "poe_id_found_other_admission")
        self.assertEqual(str(row["decision_time_source"]), "prescription_starttime_fallback")
        self.assertIn("RXROW", str(row["bundle_id"]))

    def test_04_separate_poe_same_ordertime_remain_separate_bundles(self) -> None:
        rx = pd.DataFrame({"subject_id":[1,1],"hadm_id":[10,10],"poe_id":["P1","P2"],"drug":["Tacrolimus","Fluconazole"],"starttime":["2100-01-01 12:00","2100-01-01 12:00"],"stoptime":["2100-01-05","2100-01-05"],"route":["PO","PO"]})
        timeline = self._build(rx)
        self.assertEqual(timeline["bundle_id"].nunique(), 2)
        signals = detect_signals_from_orders(timeline, self.reference)
        # No invented order sequence across separate provider orders with identical ordertime.
        self.assertFalse((signals["rule_family"] == "drug_drug_interaction").any() if not signals.empty else False)

    def test_05_prior_order_with_planned_overlap_fires(self) -> None:
        poe = pd.DataFrame({"poe_id":["P1","P2"],"subject_id":[1,1],"hadm_id":[10,10],"ordertime":["2100-01-01 08:00","2100-01-01 10:00"],"poe_seq":[1,2]})
        rx = pd.DataFrame({"subject_id":[1,1],"hadm_id":[10,10],"poe_id":["P1","P2"],"drug":["Tacrolimus","Fluconazole"],"starttime":["2100-01-02","2100-01-03"],"stoptime":["2100-01-10","2100-01-05"],"route":["PO","PO"]})
        timeline = self._build(rx, poe)
        signals = detect_signals_from_orders(timeline, self.reference)
        self.assertTrue((signals["rule_family"] == "drug_drug_interaction").any())

    def test_06_prior_order_without_planned_overlap_does_not_fire(self) -> None:
        poe = pd.DataFrame({"poe_id":["P1","P2"],"subject_id":[1,1],"hadm_id":[10,10],"ordertime":["2100-01-01 08:00","2100-01-01 10:00"],"poe_seq":[1,2]})
        rx = pd.DataFrame({"subject_id":[1,1],"hadm_id":[10,10],"poe_id":["P1","P2"],"drug":["Tacrolimus","Fluconazole"],"starttime":["2100-01-02","2100-01-06"],"stoptime":["2100-01-05","2100-01-10"],"route":["PO","PO"]})
        timeline = self._build(rx, poe)
        signals = detect_signals_from_orders(timeline, self.reference)
        self.assertFalse((signals["rule_family"] == "drug_drug_interaction").any() if not signals.empty else False)

    def test_07_same_provider_order_pair_fires_once(self) -> None:
        rx = pd.DataFrame({"subject_id":[1,1],"hadm_id":[10,10],"poe_id":["P1","P1"],"drug":["Tacrolimus","Fluconazole"],"starttime":["2100-01-02","2100-01-02"],"stoptime":["2100-01-05","2100-01-05"],"route":["PO","PO"]})
        timeline = self._build(rx)
        signals = detect_signals_from_orders(timeline, self.reference)
        ddi = signals[signals["rule_family"].eq("drug_drug_interaction")]
        self.assertEqual(len(ddi), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
