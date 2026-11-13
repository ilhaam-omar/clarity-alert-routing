from __future__ import annotations

import unittest

import pandas as pd

from clarity_v3.comparators import (
    ACTIONS,
    apply_static_guardrails,
    build_static_signal_routes,
    static_matrix_action,
)


class StaticSeverityCertaintyComparatorTests(unittest.TestCase):
    def test_01_canonical_matrix_cells(self) -> None:
        self.assertEqual(static_matrix_action(False, False), "suppress")
        self.assertEqual(static_matrix_action(False, True), "passive")
        self.assertEqual(static_matrix_action(True, True), "alert")
        self.assertEqual(static_matrix_action(True, False), "escalate")

    def test_02_hard_safety_guardrails(self) -> None:
        action, _ = apply_static_guardrails(
            "suppress",
            safety_class="must_interrupt",
            admissible_actions=("alert",),
            high_certainty=False,
        )
        self.assertEqual(action, "alert")

        action, _ = apply_static_guardrails(
            "suppress",
            safety_class="no_suppress",
            admissible_actions=("passive", "alert", "escalate"),
            high_certainty=False,
        )
        self.assertEqual(action, "passive")

    def test_03_signal_denominator_is_preserved(self) -> None:
        states = pd.DataFrame(
            [
                {
                    "signal_id": "S1",
                    "bundle_id": "B1",
                    "subject_id": 1,
                    "hadm_id": 10,
                    "timestamp": "2024-01-01",
                    "base_severity": 2.0,
                    "certainty_class": "low",
                    "safety_class": "standard",
                    "admissible_actions": "suppress|passive|alert|escalate",
                },
                {
                    "signal_id": "S2",
                    "bundle_id": "B2",
                    "subject_id": 2,
                    "hadm_id": 20,
                    "timestamp": "2024-01-02",
                    "base_severity": 9.0,
                    "certainty_class": "high",
                    "safety_class": "must_interrupt",
                    "admissible_actions": "alert",
                },
            ]
        )
        routes = build_static_signal_routes(states, severity_threshold=8.0)
        self.assertEqual(len(routes), len(states))
        self.assertEqual(routes.loc[routes["signal_id"].eq("S1"), "pre_capacity_action"].iloc[0], "suppress")
        self.assertEqual(routes.loc[routes["signal_id"].eq("S2"), "pre_capacity_action"].iloc[0], "alert")


if __name__ == "__main__":
    unittest.main(verbosity=2)
