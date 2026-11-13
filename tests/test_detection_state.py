from __future__ import annotations

import unittest

from tests.helpers import demo_frames


class DetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference, cls.orders, cls.signals, cls.states, cls.design, cls.decisions = demo_frames()

    def test_01_reference_counts(self) -> None:
        counts = self.reference.active_rule_counts()
        self.assertEqual(counts["ddi_rules"], 226)
        self.assertEqual(counts["renal_rules"], 36)
        self.assertEqual(counts["transplant_phase_scope"], 11)

    def test_02_every_order_was_retained_in_timeline(self) -> None:
        self.assertEqual(len(self.orders), 15)
        self.assertEqual(self.orders["subject_id"].nunique(), 4)

    def test_03_same_order_rule_is_unique(self) -> None:
        self.assertFalse(self.signals.duplicated(["order_id", "rule_id"]).any())

    def test_04_later_repeat_firing_is_preserved(self) -> None:
        repeated = self.states[self.states["repeat_status"].ne("new_episode")]
        self.assertGreaterEqual(len(repeated), 1)
        self.assertIn("repeat_with_material_change", set(repeated["repeat_status"]))

    def test_05_one_state_per_signal(self) -> None:
        self.assertEqual(len(self.signals), len(self.states))
        self.assertEqual(set(self.signals["signal_id"]), set(self.states["signal_id"]))

    def test_06_phase_unknown_is_explicit(self) -> None:
        self.assertGreater((self.states["transplant_phase"] == "prevalent_phase_unknown").sum(), 0)

    def test_07_phase_unknown_does_not_claim_phase_specific_match(self) -> None:
        unknown = self.states[self.states["transplant_phase"].eq("prevalent_phase_unknown")]
        forbidden = unknown["protocol_fit"].eq("phase_specific_match")
        self.assertFalse(forbidden.any())

    def test_08_acute_renal_flag_reaches_state(self) -> None:
        self.assertGreater(self.states["acute_renal_deterioration"].astype(bool).sum(), 0)

    def test_09_high_severity_is_not_suppressible(self) -> None:
        high = self.states[self.states["base_severity"].ge(8)]
        self.assertTrue(high["admissible_actions"].astype(str).str.contains("suppress").eq(False).all())

    def test_10_repeat_is_not_tolerance(self) -> None:
        repeat = self.states[self.states["repeat_status"].ne("new_episode")]
        self.assertFalse(repeat["management_status"].astype(str).str.contains("toler", case=False).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
