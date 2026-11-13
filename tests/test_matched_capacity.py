"""Independent synthetic checks for the absolute-budget experiment allocator."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from matched_capacity import allocate_absolute  # noqa: E402


def case(key, action, severity=8, expert=0, frontline=0, *, protected_alert=False,
         protected_review=False, allowed=None, fallback="passive", eligible=False):
    if allowed is None:
        allowed = "alert" if protected_alert else (
            "alert|escalate" if protected_review else "suppress|passive|alert|escalate"
        )
    return {
        "bundle_id": key, "pre_capacity_action": action,
        "highest_base_severity": severity, "delta_expert": expert,
        "delta_frontline": frontline, "protected_alert": protected_alert,
        "protected_review": protected_review, "admissible_actions": allowed,
        "fallback_non_resource_action": fallback, "bundle_suppression_eligible": eligible,
    }


def allocate(rows, F=10, K=1, ranking="regret"):
    return allocate_absolute(pd.DataFrame(rows), F, K, ranking, "synthetic", 100).set_index("bundle_id")


def mixed_rows():
    return [
        case("A", "alert", frontline=9, protected_alert=True),
        case("AR", "alert", frontline=7, protected_review=True),
        case("E1", "escalate", expert=8, frontline=5, protected_review=True),
        case("E2", "escalate", expert=3, frontline=4, protected_review=True),
        case("E3", "escalate", severity=5, expert=1),
        case("D1", "alert", severity=5, frontline=2),
        case("D2", "alert", severity=3, frontline=1),
        case("P", "passive", severity=3, expert=999, frontline=999),
        case("S", "suppress", severity=3, eligible=True, fallback="suppress"),
    ]


class AbsoluteCapacityTests(unittest.TestCase):
    def test_fixed_K_is_same_despite_different_policy_demand(self):
        small = [case(f"S{i}", "escalate", expert=i, protected_review=True) for i in range(3)]
        large = [case(f"L{i}", "escalate", expert=i, protected_review=True) for i in range(5)]
        for rows in (small, large):
            result = allocate(rows, F=0, K=2)
            self.assertEqual(int(result.final_action.eq("escalate").sum()), 2)
            self.assertTrue(result.expert_slots.eq(2).all())

    def test_ranking_changes_recipients_with_identical_totals(self):
        rows = [case("A", "escalate", severity=8, expert=1, protected_review=True),
                case("B", "escalate", severity=5, expert=9, protected_review=True)]
        simple = allocate(rows, F=0, K=1, ranking="severity")
        robust = allocate(rows, F=0, K=1, ranking="regret")
        self.assertEqual(simple.loc["A", "final_action"], "escalate")
        self.assertEqual(robust.loc["B", "final_action"], "escalate")
        self.assertEqual(simple.final_action.value_counts().to_dict(), robust.final_action.value_counts().to_dict())

    def test_severity_does_not_read_optimisation_scores(self):
        rows = [case("A", "escalate", severity=8, expert="not-a-score", protected_review=True),
                case("B", "escalate", severity=5, expert=1e12, protected_review=True)]
        result = allocate(rows, ranking="severity")
        self.assertEqual(result.loc["A", "final_action"], "escalate")
        frame = pd.DataFrame(rows).drop(columns=["delta_expert", "delta_frontline"])
        result = allocate_absolute(frame, 10, 1, "severity", "synthetic", 100)
        self.assertEqual(result.set_index("bundle_id").loc["A", "final_action"], "escalate")

    def test_identifier_breaks_ties_without_input_order_dependence(self):
        rows = [case("B", "escalate", expert=2, protected_review=True),
                case("A", "escalate", expert=2, protected_review=True)]
        for ranking in ("severity", "regret"):
            first = allocate(rows, ranking=ranking)
            second = allocate(rows[::-1], ranking=ranking)
            self.assertEqual(first.loc["A", "final_action"], "escalate")
            self.assertEqual(first.final_action.to_dict(), second.final_action.to_dict())

    def test_unsafe_suppression_fallback_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Suppression gate failed for fallback"):
            allocate([case("E", "escalate", fallback="suppress", eligible=False)], K=0)

    def test_unsafe_initial_suppression_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Suppression gate failed for final"):
            allocate([case("S", "suppress", eligible=False)])

    def test_eligible_suppression_and_common_fallback_retained(self):
        rows = [case("S", "suppress", eligible=True),
                case("E", "escalate", fallback="suppress", eligible=True)]
        result = allocate(rows, K=0)
        self.assertTrue(result.final_action.eq("suppress").all())

    def test_protected_spillover_exceeds_nominal_budget(self):
        result = allocate(mixed_rows(), F=0, K=0)
        self.assertEqual(result.loc["A", "capacity_status"], "protected_immediate_alert_overrun")
        self.assertEqual(result.loc["AR", "capacity_status"], "protected_review_alert_overrun")
        for key in ("E1", "E2"):
            self.assertEqual(result.loc[key, "final_action"], "alert")
            self.assertEqual(result.loc[key, "capacity_status"], "protected_review_spillover_overrun")
        self.assertEqual(result.loc["E3", "final_action"], "passive")

    def test_protected_review_prealert_does_not_consume_review_slot(self):
        rows = [case("AR", "alert", protected_review=True, expert=999),
                case("E", "escalate", protected_review=True, expert=1)]
        result = allocate(rows, F=0, K=1)
        self.assertEqual(result.loc["AR", "final_action"], "alert")
        self.assertTrue(pd.isna(result.loc["AR", "review_rank"]))
        self.assertEqual(result.loc["E", "final_action"], "escalate")

    def test_non_resource_routes_are_not_promoted_when_slots_exist(self):
        result = allocate(mixed_rows(), F=100, K=100)
        self.assertEqual(result.loc["P", "final_action"], "passive")
        self.assertEqual(result.loc["S", "final_action"], "suppress")

    def test_no_scarcity_preserves_every_initial_action_for_both_rankings(self):
        rows = mixed_rows()
        expected = {row["bundle_id"]: row["pre_capacity_action"] for row in rows}
        for ranking in ("severity", "regret"):
            self.assertEqual(allocate(rows, F=100, K=100, ranking=ranking).final_action.to_dict(), expected)

    def test_remaining_frontline_budget_used_for_discretionary_alerts(self):
        rows = [case("A", "alert", protected_alert=True),
                case("D1", "alert", severity=5, frontline=9),
                case("D2", "alert", severity=8, frontline=1)]
        result = allocate(rows, F=2, K=0)
        self.assertEqual(result.loc["D1", "final_action"], "alert")
        self.assertEqual(result.loc["D2", "final_action"], "passive")
        result = allocate(rows, F=2, K=0, ranking="severity")
        self.assertEqual(result.loc["D1", "final_action"], "passive")
        self.assertEqual(result.loc["D2", "final_action"], "alert")

    def test_false_string_does_not_protect_or_permit_suppression(self):
        row = case("P", "passive")
        row.update(protected_alert="False", protected_review="False", bundle_suppression_eligible="False")
        result = allocate([row], F=0, K=0)
        self.assertEqual(result.loc["P", "final_action"], "passive")
        self.assertFalse(result.loc["P", "protected_alert"])

    def test_missing_gate_duplicate_ids_and_invalid_budgets_fail(self):
        frame = pd.DataFrame(mixed_rows())
        with self.assertRaisesRegex(ValueError, "missing columns"):
            allocate_absolute(frame.drop(columns="bundle_suppression_eligible"), 1, 1, "regret", "test", 100)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            allocate(mixed_rows() + [mixed_rows()[0]])
        for F, K in [(-1, 0), (0, -1), (1.2, 1), (1, True)]:
            with self.assertRaises(ValueError):
                allocate_absolute(frame, F, K, "regret", "test", 100)

    def test_regret_mode_matches_original_normalised_allocator(self):
        project = ROOT
        if not (project / "clarity_v3" / "capacity.py").exists():
            self.skipTest("Original allocator source is not available in this environment")
        sys.path.insert(0, str(project))
        from clarity_v3.capacity import apply_dual_capacity
        from clarity_v3.models import CapacityScenario
        rows = pd.DataFrame(mixed_rows())
        # Three initial review candidates: each fraction gives exact integer K.
        for F, fraction, K in [(0, 0.0, 0), (2, 1/3, 1), (4, 2/3, 2), (10, 1.0, 3)]:
            scenario = CapacityScenario(
                name="reference", allocation_scope="cohort_normalized",
                frontline_alerts_per_100_order_bundles=F,
                expert_review_coverage_fraction=fraction,
                critical_reviews_spill_to_frontline=True,
            )
            old = apply_dual_capacity(rows, scenario, total_order_bundles=100).set_index("bundle_id")
            new = allocate_absolute(rows, F, K, "regret", "reference", 100).set_index("bundle_id")
            for column in ("final_action", "capacity_status", "frontline_slots", "expert_slots"):
                self.assertEqual(new[column].to_dict(), old[column].to_dict())
            self.assertEqual(new.frontline_rank.fillna(-1).to_dict(), old.frontline_rank.fillna(-1).to_dict())
            self.assertEqual(new.review_rank.fillna(-1).to_dict(), old.expert_rank.fillna(-1).to_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
