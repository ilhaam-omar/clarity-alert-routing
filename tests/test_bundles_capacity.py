from __future__ import annotations

import unittest

import pandas as pd

from clarity_v3.bundles import build_order_bundles
from clarity_v3.capacity import apply_dual_capacity, capacity_summary
from clarity_v3.models import CapacityScenario
from tests.helpers import demo_frames


def _scenario(
    name: str = "test",
    frontline_rate: float | None = 0.0,
    expert_fraction: float | None = 0.0,
    unconstrained: bool = False,
) -> CapacityScenario:
    return CapacityScenario(
        name=name,
        allocation_scope="cohort_normalized",
        frontline_alerts_per_100_order_bundles=frontline_rate,
        expert_review_coverage_fraction=expert_fraction,
        unconstrained=unconstrained,
        critical_reviews_spill_to_frontline=True,
    )


def _capacity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "bundle_id": "A",
                "pre_capacity_action": "alert",
                "fallback_non_resource_action": "passive",
                "admissible_actions": "alert",
                "protected_alert": True,
                "protected_review": False,
                "delta_frontline": 9.0,
                "delta_expert": 0.0,
            },
            {
                "bundle_id": "E1",
                "pre_capacity_action": "escalate",
                "fallback_non_resource_action": "passive",
                "admissible_actions": "alert|escalate",
                "protected_alert": False,
                "protected_review": True,
                "delta_frontline": 5.0,
                "delta_expert": 8.0,
            },
            {
                "bundle_id": "E2",
                "pre_capacity_action": "escalate",
                "fallback_non_resource_action": "passive",
                "admissible_actions": "alert|escalate",
                "protected_alert": False,
                "protected_review": True,
                "delta_frontline": 4.0,
                "delta_expert": 3.0,
            },
            {
                "bundle_id": "E3",
                "pre_capacity_action": "escalate",
                "fallback_non_resource_action": "passive",
                "admissible_actions": "passive|escalate",
                "protected_alert": False,
                "protected_review": False,
                "delta_frontline": 0.0,
                "delta_expert": 1.0,
            },
            {
                "bundle_id": "D1",
                "pre_capacity_action": "alert",
                "fallback_non_resource_action": "passive",
                "admissible_actions": "passive|alert",
                "protected_alert": False,
                "protected_review": False,
                "delta_frontline": 2.0,
                "delta_expert": 0.0,
            },
            {
                "bundle_id": "P",
                "pre_capacity_action": "passive",
                "fallback_non_resource_action": "passive",
                "admissible_actions": "passive|alert|escalate",
                "protected_alert": False,
                "protected_review": False,
                "delta_frontline": 7.0,
                "delta_expert": 7.0,
            },
        ]
    )


class BundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.orders, cls.signals, cls.states, cls.design, cls.decisions = demo_frames()
        cls.bundles = build_order_bundles(cls.decisions)

    def test_01_one_bundle_per_bundle_id(self) -> None:
        self.assertFalse(self.bundles["bundle_id"].duplicated().any())
        self.assertEqual(set(self.bundles["bundle_id"]), set(self.signals["bundle_id"]))

    def test_02_multi_signal_order_consumes_one_bundle(self) -> None:
        self.assertGreater((self.bundles["signal_count"] > 1).sum(), 0)
        self.assertLess(len(self.bundles), len(self.signals))

    def test_03_protected_flags_are_boolean(self) -> None:
        self.assertTrue(all(isinstance(value, bool) for value in self.bundles["protected_alert"]))
        self.assertTrue(all(isinstance(value, bool) for value in self.bundles["protected_review"]))

    def test_04_capacity_scenario_is_cohort_normalized(self) -> None:
        with self.assertRaises(ValueError):
            CapacityScenario(
                name="invalid",
                allocation_scope="eight_hour_shift",
                frontline_alerts_per_100_order_bundles=1.0,
                expert_review_coverage_fraction=0.5,
            ).validate()
        _scenario().validate()

    def test_05_protected_alert_is_never_rationed(self) -> None:
        final = apply_dual_capacity(
            _capacity_frame(),
            _scenario(frontline_rate=0.0, expert_fraction=0.0),
            total_order_bundles=100,
        )
        row = final.set_index("bundle_id").loc["A"]
        self.assertEqual(row["final_action"], "alert")
        self.assertEqual(row["capacity_status"], "protected_immediate_alert_overrun")

    def test_06_expert_fraction_ranks_by_escalation_benefit(self) -> None:
        # Three expert-demand bundles and 34% coverage gives one expert slot.
        final = apply_dual_capacity(
            _capacity_frame(),
            _scenario(frontline_rate=10.0, expert_fraction=0.34),
            total_order_bundles=100,
        ).set_index("bundle_id")
        self.assertEqual(final.loc["E1", "final_action"], "escalate")
        self.assertEqual(final.loc["E2", "final_action"], "alert")
        self.assertEqual(final.loc["E3", "final_action"], "passive")
        self.assertEqual(int(final.loc["E1", "expert_rank"]), 1)

    def test_07_passive_bundle_is_not_upgraded_by_capacity(self) -> None:
        final = apply_dual_capacity(
            _capacity_frame(),
            _scenario(frontline_rate=100.0, expert_fraction=1.0),
            total_order_bundles=100,
        ).set_index("bundle_id")
        self.assertEqual(final.loc["P", "final_action"], "passive")

    def test_08_unconstrained_preserves_pre_capacity_actions(self) -> None:
        frame = _capacity_frame()
        final = apply_dual_capacity(
            frame,
            _scenario(
                name="unconstrained",
                frontline_rate=None,
                expert_fraction=1.0,
                unconstrained=True,
            ),
            total_order_bundles=100,
        )
        observed = final.set_index("bundle_id")["final_action"].to_dict()
        expected = frame.set_index("bundle_id")["pre_capacity_action"].to_dict()
        self.assertEqual(observed, expected)

    def test_09_false_string_is_not_true(self) -> None:
        frame = _capacity_frame().query("bundle_id == 'P'").copy()
        frame["protected_alert"] = "False"
        frame["protected_review"] = "False"
        final = apply_dual_capacity(
            frame,
            _scenario(frontline_rate=0.0, expert_fraction=0.0),
            total_order_bundles=100,
        )
        self.assertFalse(final["protected_alert"].any())
        self.assertFalse(final["protected_review"].any())
        self.assertEqual(final.iloc[0]["final_action"], "passive")

    def test_10_summary_uses_prescribing_bundle_denominator(self) -> None:
        final = apply_dual_capacity(
            _capacity_frame(),
            _scenario(frontline_rate=1.0, expert_fraction=0.5),
            total_order_bundles=200,
        )
        summary = capacity_summary(final).iloc[0]
        self.assertEqual(int(summary["total_order_bundles"]), 200)
        self.assertAlmostEqual(
            float(summary["final_alerts_per_100_order_bundles"]),
            100.0 * int(summary["alert"]) / 200.0,
        )

    def test_11_alert_with_protected_review_is_not_reclassified_as_expert_demand(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "bundle_id": "AR",
                    "pre_capacity_action": "alert",
                    "fallback_non_resource_action": "passive",
                    "admissible_actions": "alert|escalate",
                    "protected_alert": False,
                    "protected_review": True,
                    "delta_frontline": 4.0,
                    "delta_expert": 5.0,
                },
                {
                    "bundle_id": "E",
                    "pre_capacity_action": "escalate",
                    "fallback_non_resource_action": "passive",
                    "admissible_actions": "alert|escalate",
                    "protected_alert": False,
                    "protected_review": True,
                    "delta_frontline": 3.0,
                    "delta_expert": 6.0,
                },
            ]
        )
        unconstrained = apply_dual_capacity(
            frame,
            _scenario(
                name="unconstrained",
                frontline_rate=None,
                expert_fraction=1.0,
                unconstrained=True,
            ),
            total_order_bundles=100,
        ).set_index("bundle_id")
        self.assertEqual(unconstrained.loc["AR", "final_action"], "alert")
        self.assertEqual(unconstrained.loc["E", "final_action"], "escalate")
        self.assertEqual(int(unconstrained.loc["AR", "pre_capacity_expert_demand"]), 1)

        constrained = apply_dual_capacity(
            frame,
            _scenario(frontline_rate=0.0, expert_fraction=0.0),
            total_order_bundles=100,
        ).set_index("bundle_id")
        self.assertEqual(constrained.loc["AR", "final_action"], "alert")
        self.assertEqual(
            constrained.loc["AR", "capacity_status"],
            "protected_review_alert_overrun",
        )
        self.assertEqual(constrained.loc["E", "final_action"], "alert")
        self.assertTrue(
            str(constrained.loc["E", "capacity_status"]).startswith(
                "protected_review_spillover"
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
