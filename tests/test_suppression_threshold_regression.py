"""Synthetic regression tests for the suppression-consensus boundary.

These tests exercise the real envelope regret/support calculation and policy
branches with deliberately controlled cost arrays. They contain no patient data
and do not constitute clinical validation.
"""
from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from clarity_v3.models import ACTIONS, parse_actions
from clarity_v3.uncertainty import (
    ActionSetConfig, MultiEnvelopeActionPolicy, _REQUIRED_PARAMETERS,
)
from tests import test_uncertainty_policy as original_policy_tests

ENVELOPES = (
    "clinical_safety_priority", "workflow_balanced", "alert_burden_priority",
)


class ControlledCostsPolicy(MultiEnvelopeActionPolicy):
    """Create a known regret winner with an exact support count out of 100."""

    def __init__(self, support_counts=(75, 75, 75), winner="suppress", config=None):
        frames = []
        for name in ENVELOPES:
            frame = pd.DataFrame({key: np.zeros(100) for key in _REQUIRED_PARAMETERS})
            frame["envelope"] = name
            frames.append(frame)
        super().__init__(pd.concat(frames, ignore_index=True),
                         {name: 0.95 for name in ENVELOPES}, config)
        self.support_counts = dict(zip(ENVELOPES, support_counts))
        self.winner = winner

    def _costs(self, state, frame):
        name = str(frame.iloc[0]["envelope"])
        count = self.support_counts[name]
        index = {action: i for i, action in enumerate(ACTIONS)}
        competitor = "passive" if self.winner == "suppress" else "alert"
        costs = np.tile([4.0, 3.0, 2.0, 2.5], (len(frame), 1))
        # The declared winner loses only by 0.1 when it loses, but its
        # competitor loses by 1.0 elsewhere. Thus it wins the Q95 regret
        # comparison even when its draw-wise support is below the threshold.
        costs[:, index[self.winner]] = 0.1
        costs[:, index[competitor]] = 0.0
        costs[:count, index[self.winner]] = 0.0
        costs[:count, index[competitor]] = 1.0
        for action in ACTIONS:
            if action not in parse_actions(state.admissible_actions):
                costs[:, index[action]] = np.inf
        return costs


class SuppressionThresholdRegressionTests(unittest.TestCase):
    @staticmethod
    def state(**changes):
        return original_policy_tests.UncertaintyTests.state(**changes)

    def decision(self, counts=(75, 75, 75), **state_changes):
        return ControlledCostsPolicy(counts).decide(self.state(**state_changes))

    def test_01_support_between_general_and_strict_threshold_is_not_suppressed(self):
        for count in (70, 75, 79):
            with self.subTest(support=count / 100):
                result = self.decision((count,) * 3)
                self.assertEqual(result.consensus_action, "suppress")
                self.assertAlmostEqual(result.consensus_support_floor, count / 100)
                self.assertEqual(result.pre_capacity_action, "passive")
                self.assertEqual(result.fallback_non_resource_action, "passive")

    def test_02_below_strict_suppression_threshold_is_calibration_sensitive(self):
        self.assertTrue(self.decision().calibration_sensitive)

    def test_03_exact_strict_boundary_is_accepted(self):
        result = self.decision((80, 80, 80))
        self.assertEqual(result.pre_capacity_action, "suppress")
        self.assertEqual(result.fallback_non_resource_action, "suppress")
        self.assertFalse(result.calibration_sensitive)

    def test_04_every_envelope_must_meet_strict_threshold(self):
        for counts in ((79, 90, 90), (90, 79, 90), (90, 90, 79)):
            with self.subTest(counts=counts):
                result = self.decision(counts)
                self.assertEqual(result.pre_capacity_action, "passive")
                self.assertTrue(result.calibration_sensitive)

    def test_05_changed_state_cannot_use_suppression_branch_or_fallback(self):
        result = self.decision((90, 90, 90), material_change_since_prior=True)
        self.assertEqual(result.pre_capacity_action, "passive")
        self.assertEqual(result.fallback_non_resource_action, "passive")

    def test_06_nonstandard_safety_class_cannot_use_suppression(self):
        result = self.decision((90, 90, 90), safety_class="no_suppress")
        self.assertEqual(result.pre_capacity_action, "passive")
        self.assertEqual(result.fallback_non_resource_action, "passive")

    def test_07_general_consensus_still_accepts_other_routes_at_70_percent(self):
        result = ControlledCostsPolicy((70,) * 3, winner="passive").decide(self.state())
        self.assertEqual(result.pre_capacity_action, "passive")
        self.assertFalse(result.calibration_sensitive)

    def test_08_general_consensus_below_70_percent_remains_sensitive(self):
        result = ControlledCostsPolicy((69,) * 3, winner="passive").decide(self.state())
        self.assertEqual(result.pre_capacity_action, "passive")
        self.assertTrue(result.calibration_sensitive)

    def test_09_configured_suppression_threshold_controls_selection_and_flag(self):
        config = replace(ActionSetConfig(), suppress_support_threshold=0.85)
        result = ControlledCostsPolicy((84,) * 3, config=config).decide(self.state())
        self.assertEqual(result.pre_capacity_action, "passive")
        self.assertTrue(result.calibration_sensitive)
        result = ControlledCostsPolicy((85,) * 3, config=config).decide(self.state())
        self.assertEqual(result.pre_capacity_action, "suppress")
        self.assertFalse(result.calibration_sensitive)

    def test_10_protected_routes_remain_protected(self):
        result = self.decision(safety_class="must_interrupt", admissible_actions=("alert",))
        self.assertEqual(result.pre_capacity_action, "alert")
        self.assertTrue(result.protected_alert)
        result = self.decision(safety_class="must_review_or_interrupt",
                               admissible_actions=("alert", "escalate"))
        self.assertIn(result.pre_capacity_action, ("alert", "escalate"))
        self.assertTrue(result.protected_review)

    def test_11_last_resort_cannot_reintroduce_unsafe_suppression(self):
        # Deliberately inconsistent upstream input must fail closed, never
        # turn a material change into a suppression-only final decision.
        with self.assertRaisesRegex(ValueError, "no safe admissible action"):
            self.decision((90,) * 3, material_change_since_prior=True,
                          admissible_actions=("suppress",))

    def test_12_non_resource_fallback_cannot_reintroduce_unsafe_suppression(self):
        # A resource action can be selected, yet its proposed fallback must
        # not be suppression when the strict safety conditions are absent.
        with self.assertRaisesRegex(ValueError, "no safe non-resource fallback"):
            self.decision((90,) * 3, safety_class="must_interrupt",
                          admissible_actions=("suppress", "alert"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
