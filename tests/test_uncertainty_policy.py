from __future__ import annotations

import unittest

import pandas as pd

from clarity_v3.config import load_uncertainty_config
from clarity_v3.models import ContextState
from clarity_v3.uncertainty import MultiEnvelopeActionPolicy, sample_design
from tests.helpers import ROOT, demo_frames


class UncertaintyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.envelopes, self.action = load_uncertainty_config(ROOT / "config/uncertainty_design.json")
        self.design = sample_design(self.envelopes)
        self.policy = MultiEnvelopeActionPolicy(
            self.design,
            {item.name: item.regret_quantile for item in self.envelopes},
            self.action,
        )

    def test_01_design_is_reproducible(self) -> None:
        second = sample_design(self.envelopes)
        pd.testing.assert_frame_equal(self.design, second)

    def test_02_structural_constraints(self) -> None:
        self.assertTrue((self.design["rho_alert"] < self.design["rho_passive"]).all())
        self.assertTrue((self.design["rho_escalate"] < self.design["rho_passive"]).all())
        self.assertTrue((self.design["burden_alert"] > self.design["burden_passive"]).all())
        self.assertTrue((self.design["uncertainty_escalate_fraction"] <= self.design["uncertainty_alert_fraction"]).all())

    @staticmethod
    def state(**overrides) -> ContextState:
        values = dict(
            signal_id="S", subject_id=1, hadm_id=1, order_id="O", bundle_id="B",
            timestamp=pd.Timestamp("2024-01-01"), rule_id="R", rule_family="ddi",
            base_severity=3.0, evidence_support_index=0.4, contextual_harm_index=3.0,
            evidence_completeness_index=0.8, certainty_class="moderate",
            current_creatinine=1.0, hours_since_current_creatinine=1.0,
            acute_renal_deterioration=False, acute_renal_deterioration_basis="none",
            crrt_active=False, transplant_patient=True, transplant_date_source="observed",
            transplant_phase="maintenance_gt365d", days_since_transplant=500,
            protocol_fit="none", monitoring_status="not_assessed", transition_status="none",
            episode_id="E", episode_occurrence_index=1, prior_lifetime_occurrences=0,
            prior_admission_occurrences=0, hours_since_prior_same_signal=None,
            material_change_since_prior=False, repeat_status="new_episode",
            note_context_usable=False, management_status="unmanaged", safety_class="standard",
            admissible_actions=("suppress","passive","alert","escalate"),
            missingness_flags=(), context_reason="",
        )
        values.update(overrides)
        return ContextState(**values)

    def test_03_must_interrupt_is_alert(self) -> None:
        decision = self.policy.decide(self.state(
            base_severity=8.0, contextual_harm_index=8.0, certainty_class="high",
            safety_class="must_interrupt", admissible_actions=("alert",),
        ))
        self.assertEqual(decision.pre_capacity_action, "alert")
        self.assertTrue(decision.protected_alert)
        self.assertIsInstance(decision.protected_review, bool)

    def test_04_must_review_is_resource_action(self) -> None:
        decision = self.policy.decide(self.state(
            base_severity=8.0, contextual_harm_index=8.0,
            safety_class="must_review_or_interrupt", admissible_actions=("alert","escalate"),
        ))
        self.assertIn(decision.pre_capacity_action, {"alert", "escalate"})
        self.assertTrue(decision.protected_review)

    def test_05_suppression_requires_standard_no_change(self) -> None:
        _, _, _, states, _, decisions = demo_frames()
        suppress = decisions[decisions["pre_capacity_action"].eq("suppress")]
        self.assertTrue(suppress["safety_class"].eq("standard").all())
        self.assertFalse(suppress["material_change_since_prior"].astype(bool).any())

    def test_06_design_support_is_not_forced_to_sum_across_minima(self) -> None:
        _, _, _, _, _, decisions = demo_frames()
        supports = decisions[[
            "action_support_suppress_min", "action_support_passive_min",
            "action_support_alert_min", "action_support_escalate_min",
        ]]
        self.assertTrue((supports >= 0).all().all())
        self.assertTrue((supports <= 1).all().all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
