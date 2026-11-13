from __future__ import annotations

import unittest

import pandas as pd

from clarity_v3.temporal import EpisodeMemory, RenalTimeline
from clarity_v3.transplant_phase import (
    TransplantAnchor,
    classify_transplant_phase,
    is_kidney_transplant_procedure,
)


class RenalTemporalTests(unittest.TestCase):
    def test_01_future_creatinine_is_excluded(self) -> None:
        labs = pd.DataFrame({
            "charttime": pd.to_datetime(["2024-01-01 08:00", "2024-01-02 08:00"]),
            "valuenum": [1.0, 2.0],
            "hadm_id": [10, 10],
        })
        snapshot = RenalTimeline(labs).snapshot(pd.Timestamp("2024-01-01 12:00"), 10, 50, "M")
        self.assertEqual(snapshot.current_creatinine, 1.0)

    def test_02_kdigo_aligned_48h_flag(self) -> None:
        labs = pd.DataFrame({
            "charttime": pd.to_datetime(["2024-01-01 08:00", "2024-01-02 08:00"]),
            "valuenum": [1.0, 1.4],
            "hadm_id": [10, 10],
        })
        snapshot = RenalTimeline(labs).snapshot(pd.Timestamp("2024-01-02 09:00"), 10, 50, "F")
        self.assertTrue(snapshot.acute_deterioration_48h)
        self.assertAlmostEqual(snapshot.creatinine_delta_48h or 0, 0.4)

    def test_03_kdigo_aligned_7d_flag(self) -> None:
        labs = pd.DataFrame({
            "charttime": pd.to_datetime(["2024-01-01 08:00", "2024-01-06 08:00"]),
            "valuenum": [1.0, 1.6],
            "hadm_id": [10, 10],
        })
        snapshot = RenalTimeline(labs).snapshot(pd.Timestamp("2024-01-06 09:00"), 10, 50, "M")
        self.assertTrue(snapshot.acute_deterioration_7d)
        self.assertGreaterEqual(snapshot.creatinine_ratio_7d or 0, 1.5)


class TransplantPhaseTests(unittest.TestCase):
    def test_04_icd_procedure_codes(self) -> None:
        self.assertTrue(is_kidney_transplant_procedure("55.69", 9))
        self.assertTrue(is_kidney_transplant_procedure("0TY00Z0", 10))
        self.assertFalse(is_kidney_transplant_procedure("0FT44ZZ", 10))

    def test_05_incident_phase(self) -> None:
        anchor = TransplantAnchor(1, pd.Timestamp("2024-01-01"), 10, "observed", "0TY00Z0", 10)
        phase, days = classify_transplant_phase(pd.Timestamp("2024-04-15"), 11, anchor)
        self.assertEqual(phase, "transition_91_180d")
        self.assertEqual(days, 105)

    def test_06_prevalent_phase_remains_unknown(self) -> None:
        anchor = TransplantAnchor(1, None, None, "prevalent", None, None)
        phase, days = classify_transplant_phase(pd.Timestamp("2024-01-01"), 10, anchor)
        self.assertEqual(phase, "prevalent_phase_unknown")
        self.assertIsNone(days)


class EpisodeTests(unittest.TestCase):
    def test_07_repeat_within_active_overlap(self) -> None:
        memory = EpisodeMemory()
        first = memory.observe(1, 10, "DDI:a|b", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05"), ("stable",))
        second = memory.observe(1, 10, "DDI:a|b", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-05"), ("stable",))
        self.assertEqual(first["repeat_status"], "new_episode")
        self.assertEqual(second["repeat_status"], "repeat_unchanged_state")
        self.assertEqual(second["episode_occurrence_index"], 2)

    def test_08_material_change_realerts(self) -> None:
        memory = EpisodeMemory()
        memory.observe(1, 10, "DDI:a|b", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05"), ("stable",))
        second = memory.observe(1, 10, "DDI:a|b", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-05"), ("worsening",))
        self.assertEqual(second["repeat_status"], "repeat_with_material_change")
        self.assertTrue(second["material_change_since_prior"])

    def test_09_new_admission_is_new_episode_but_lifetime_history_remains(self) -> None:
        memory = EpisodeMemory()
        memory.observe(1, 10, "DDI:a|b", pd.Timestamp("2024-01-01"), None, ("stable",))
        second = memory.observe(1, 11, "DDI:a|b", pd.Timestamp("2024-02-01"), None, ("stable",))
        self.assertEqual(second["repeat_status"], "new_episode")
        self.assertEqual(second["prior_lifetime_occurrences"], 1)
        self.assertEqual(second["prior_admission_occurrences"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
