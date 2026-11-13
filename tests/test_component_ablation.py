from __future__ import annotations

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from clarity_v3.component_ablation import (
    ABLATIONS,
    FULL_CONTEXT,
    build_component_variant,
    build_reusable_active_snapshots,
)
from clarity_v3.config import load_state_config
from tests.helpers import ROOT, demo_frames


class RebuildComponentAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference, cls.orders, cls.signals, cls.primary, _, _ = demo_frames()
        cls.config = load_state_config(ROOT / "config/state_builder.json")
        cls.snapshots = build_reusable_active_snapshots(cls.orders, cls.signals)
        cls.by_name = {item.name: item for item in ABLATIONS}

    def _build(self, name: str):
        return build_component_variant(
            self.by_name[name],
            self.signals,
            self.orders,
            self.reference,
            self.config,
            active_snapshots=self.snapshots,
        ).sort_values("signal_id").reset_index(drop=True)

    def test_01_analysis_full_context_reproduces_production_state(self) -> None:
        explicit = build_component_variant(
            FULL_CONTEXT,
            self.signals,
            self.orders,
            self.reference,
            self.config,
            active_snapshots=self.snapshots,
        ).sort_values("signal_id").reset_index(drop=True)
        default = self.primary.sort_values("signal_id").reset_index(drop=True)
        common = sorted(set(default.columns) & set(explicit.columns))
        assert_frame_equal(
            default[common],
            explicit[common],
            check_dtype=False,
            check_like=True,
        )

    def test_02_episode_ablation_rebuilds_every_signal_as_new(self) -> None:
        ablated = self._build("without_episode_memory")
        self.assertTrue(ablated["repeat_status"].eq("new_episode").all())
        self.assertTrue(ablated["episode_occurrence_index"].eq(1).all())
        self.assertTrue(ablated["prior_lifetime_occurrences"].eq(0).all())
        self.assertTrue(ablated["prior_admission_occurrences"].eq(0).all())
        self.assertFalse(ablated["material_change_since_prior"].astype(bool).any())

    def test_03_renal_ablation_retains_current_creatinine_but_removes_trajectory(self) -> None:
        ablated = self._build("without_acute_renal_context")
        primary = self.primary.sort_values("signal_id").reset_index(drop=True)
        self.assertTrue(
            pd.to_numeric(ablated["current_creatinine"], errors="coerce").equals(
                pd.to_numeric(primary["current_creatinine"], errors="coerce")
            )
        )
        self.assertFalse(ablated["acute_renal_deterioration"].astype(bool).any())
        self.assertTrue(
            pd.to_numeric(ablated["contextual_harm_index"], errors="coerce").equals(
                pd.to_numeric(ablated["base_severity"], errors="coerce")
            )
        )
        self.assertGreater(primary["acute_renal_deterioration"].astype(bool).sum(), 0)

    def test_04_protocol_ablation_retains_transition_interpretation(self) -> None:
        ablated = self._build("without_transplant_protocol_context")
        primary = self.primary.sort_values("signal_id").reset_index(drop=True)
        registered = ~primary["protocol_fit"].eq("no_matching_transplant_protocol")
        self.assertTrue(
            ablated.loc[registered, "protocol_fit"].eq(
                "no_matching_transplant_protocol"
            ).all()
        )
        self.assertTrue(
            ablated["transition_status"].astype(str).equals(
                primary["transition_status"].astype(str)
            )
        )
        self.assertIn("confirmed_steroid_decline", set(ablated["transition_status"]))

    def test_05_transition_ablation_retains_protocol_interpretation(self) -> None:
        ablated = self._build("without_taper_route_transition_context")
        primary = self.primary.sort_values("signal_id").reset_index(drop=True)
        observed_transition = ~primary["transition_status"].eq("none")
        self.assertTrue(ablated.loc[observed_transition, "transition_status"].eq("none").all())
        self.assertTrue(
            ablated["protocol_fit"].astype(str).equals(
                primary["protocol_fit"].astype(str)
            )
        )
        self.assertIn("phase_specific_match", set(ablated["protocol_fit"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
