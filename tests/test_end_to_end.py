from __future__ import annotations

import unittest

from clarity_v3.bundles import build_order_bundles
from tests.helpers import demo_frames


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference, cls.orders, cls.signals, cls.states, cls.design, cls.decisions = demo_frames()
        cls.bundles = build_order_bundles(cls.decisions)

    def test_01_denominators_are_preserved(self) -> None:
        self.assertEqual(len(self.signals), len(self.states))
        self.assertEqual(len(self.states), len(self.decisions))

    def test_02_demo_contains_incident_and_prevalent_transplant_states(self) -> None:
        phases = set(self.orders["transplant_phase"])
        self.assertIn("index_admission_postprocedure", phases)
        self.assertIn("prevalent_phase_unknown", phases)
        self.assertIn("maintenance_gt365d", phases)

    def test_03_demo_contains_repeat_episode(self) -> None:
        repeats = set(self.states.loc[self.states["repeat_status"].ne("new_episode"), "repeat_status"])
        self.assertTrue(repeats)
        self.assertIn("repeat_with_material_change", repeats)

    def test_04_demo_contains_high_safety_alert(self) -> None:
        protected = self.decisions[self.decisions["protected_alert"]]
        self.assertGreaterEqual(len(protected), 1)
        self.assertTrue(protected["pre_capacity_action"].eq("alert").all())

    def test_05_calibration_sensitivity_is_visible(self) -> None:
        self.assertGreater(self.decisions["calibration_sensitive"].astype(bool).sum(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
