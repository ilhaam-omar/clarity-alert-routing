from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clarity_v3.bundles import build_order_bundles
from clarity_v3.comparators import (
    build_snapshot_signal_routes,
    snapshot_matrix_action,
)


class SnapshotSeverityEvidenceComparatorTests(unittest.TestCase):
    def _signals(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "signal_id": "S1",
                    "bundle_id": "B1",
                    "subject_id": 1,
                    "hadm_id": 10,
                    "timestamp": "2024-01-01 08:00:00",
                    "rule_id": "R1",
                    "rule_family": "drug_drug_interaction",
                    "base_severity": 4.0,
                    "evidence_support_index": 0.4,
                },
                {
                    "signal_id": "S2",
                    "bundle_id": "B2",
                    "subject_id": 2,
                    "hadm_id": 20,
                    "timestamp": "2024-01-01 09:00:00",
                    "rule_id": "R2",
                    "rule_family": "renal_dose_adjustment",
                    "base_severity": 4.0,
                    "evidence_support_index": 0.8,
                },
                {
                    "signal_id": "S3",
                    "bundle_id": "B3",
                    "subject_id": 3,
                    "hadm_id": 30,
                    "timestamp": "2024-01-01 10:00:00",
                    "rule_id": "R3",
                    "rule_family": "drug_drug_interaction",
                    "base_severity": 9.0,
                    "evidence_support_index": 0.8,
                },
                {
                    "signal_id": "S4",
                    "bundle_id": "B4",
                    "subject_id": 4,
                    "hadm_id": 40,
                    "timestamp": "2024-01-01 11:00:00",
                    "rule_id": "R4",
                    "rule_family": "drug_drug_interaction",
                    "base_severity": 9.0,
                    "evidence_support_index": 0.4,
                },
            ]
        )

    def test_01_four_matrix_cells(self) -> None:
        self.assertEqual(snapshot_matrix_action(False, False), "suppress")
        self.assertEqual(snapshot_matrix_action(False, True), "passive")
        self.assertEqual(snapshot_matrix_action(True, True), "alert")
        self.assertEqual(snapshot_matrix_action(True, False), "escalate")

    def test_02_stage02_signal_denominator_and_routes(self) -> None:
        signals = self._signals()
        routes = build_snapshot_signal_routes(
            signals,
            severity_threshold=8.0,
            evidence_threshold=0.7,
        )
        self.assertEqual(len(routes), len(signals))
        actual = routes.set_index("signal_id")["pre_capacity_action"].to_dict()
        self.assertEqual(
            actual,
            {"S1": "suppress", "S2": "passive", "S3": "alert", "S4": "escalate"},
        )

    def test_03_context_columns_are_not_required(self) -> None:
        routes = build_snapshot_signal_routes(
            self._signals(),
            severity_threshold=8.0,
            evidence_threshold=0.7,
        )
        forbidden_inputs = {
            "certainty_class",
            "contextual_harm_index",
            "transplant_phase",
            "protocol_fit",
            "repeat_status",
            "material_change_since_prior",
            "management_status",
        }
        self.assertTrue(forbidden_inputs.isdisjoint(self._signals().columns))
        self.assertTrue(routes["snapshot_input_scope"].eq(
            "stage02_current_rule_severity_and_evidence_only"
        ).all())

    def test_04_bundle_consolidation_preserves_strongest_snapshot_route(self) -> None:
        signals = self._signals()
        extra = signals.iloc[[1]].copy()
        extra["signal_id"] = "S5"
        extra["bundle_id"] = "B3"  # passive signal joins an alert signal
        combined = pd.concat([signals, extra], ignore_index=True)
        routes = build_snapshot_signal_routes(
            combined,
            severity_threshold=8.0,
            evidence_threshold=0.7,
        )
        bundles = build_order_bundles(routes)
        b3 = bundles.loc[bundles["bundle_id"].astype(str).eq("B3")].iloc[0]
        self.assertEqual(b3["pre_capacity_action"], "alert")
        self.assertTrue(bool(b3["protected_alert"]))

    def test_05_review_ranking_uses_only_severity_and_evidence(self) -> None:
        routes = build_snapshot_signal_routes(
            self._signals(),
            severity_threshold=8.0,
            evidence_threshold=0.7,
        )
        review = routes.loc[routes["signal_id"].eq("S4")].iloc[0]
        self.assertEqual(review["pre_capacity_action"], "escalate")
        self.assertGreater(float(review["delta_expert"]), 0.0)
        self.assertEqual(float(review["delta_frontline"]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
