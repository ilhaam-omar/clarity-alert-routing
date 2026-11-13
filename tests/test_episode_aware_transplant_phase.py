from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from clarity_v3.mimic import build_order_timeline
from clarity_v3.reference import ReferenceStore
from clarity_v3.transplant_phase import (
    TransplantAnchor,
    classify_transplant_phase,
    load_transplant_episode_assignments,
    post_transplant_phase_context_allowed,
    protocol_phase_for_context,
)

ROOT = Path(__file__).resolve().parents[1]


def _episode_row(
    subject_id: int,
    hadm_id: int,
    episode_id: str,
    episode_number: int,
    transplant_hadm_id: int,
    transplant_date: str,
    *,
    quality: str = "exact_single_in_admission",
    quality_level: str = "high",
    allowed: bool = True,
    short_gap: bool = False,
) -> dict:
    return {
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "transplant_episode_id": episode_id,
        "episode_number": episode_number,
        "transplant_hadm_id": transplant_hadm_id,
        "transplant_date": transplant_date,
        "anchor_quality": quality,
        "anchor_quality_level": quality_level,
        "phase_specific_context_allowed": allowed,
        "short_gap_repeat_transplant_candidate": short_gap,
    }


class EpisodeAwarePhaseTests(unittest.TestCase):
    def _write_common_files(self, root: Path) -> tuple[Path, Path]:
        admissions = pd.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "hadm_id": [10, 20, 21],
                "admittime": [
                    "2020-01-01 00:00:00",
                    "2024-01-01 00:00:00",
                    "2024-02-15 00:00:00",
                ],
                "dischtime": [
                    "2020-01-10 00:00:00",
                    "2024-01-10 00:00:00",
                    "2024-02-20 00:00:00",
                ],
            }
        )
        procedures = pd.DataFrame(
            {
                "subject_id": [1, 1],
                "hadm_id": [10, 20],
                "chartdate": ["2020-01-02", "2024-01-02"],
                "icd_code": ["0TY00Z0", "0TY10Z0"],
                "icd_version": [10, 10],
            }
        )
        admissions_file = root / "admissions.csv"
        procedures_file = root / "procedures_icd.csv"
        admissions.to_csv(admissions_file, index=False)
        procedures.to_csv(procedures_file, index=False)
        return admissions_file, procedures_file

    def test_01_each_admission_uses_its_assigned_transplant_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admissions_file, procedures_file = self._write_common_files(root)
            cohort = pd.DataFrame(
                [
                    _episode_row(1, 10, "1-TX1", 1, 10, "2020-01-02"),
                    _episode_row(1, 20, "1-TX2", 2, 20, "2024-01-02"),
                    _episode_row(1, 21, "1-TX2", 2, 20, "2024-01-02"),
                ]
            )
            assignments = load_transplant_episode_assignments(
                cohort, procedures_file, admissions_file
            )
            self.assertEqual(assignments[10].transplant_episode_id, "1-TX1")
            self.assertEqual(assignments[20].transplant_episode_id, "1-TX2")
            self.assertEqual(assignments[21].transplant_episode_id, "1-TX2")
            phase, days = classify_transplant_phase(
                pd.Timestamp("2024-02-15 12:00:00"), 21, assignments[21]
            )
            self.assertEqual(phase, "early_31_90d")
            self.assertEqual(days, 44)

    def test_02_short_gap_candidate_disables_phase_specific_narrowing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admissions_file, procedures_file = self._write_common_files(root)
            cohort = pd.DataFrame(
                [
                    _episode_row(
                        1,
                        20,
                        "1-TX2",
                        2,
                        20,
                        "2024-01-02",
                        allowed=True,
                        short_gap=True,
                    )
                ]
            )
            assignment = load_transplant_episode_assignments(
                cohort, procedures_file, admissions_file
            )[20]
            self.assertTrue(assignment.short_gap_repeat_transplant_candidate)
            self.assertFalse(assignment.phase_specific_context_allowed)

    def test_03_low_quality_followup_respects_explicit_phase_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admissions_file, procedures_file = self._write_common_files(root)
            cohort = pd.DataFrame(
                [
                    _episode_row(
                        1,
                        21,
                        "1-TX2",
                        2,
                        20,
                        "2024-01-02",
                        quality="admission_date_proxy_invalid_chartdate",
                        quality_level="low",
                        allowed=False,
                    )
                ]
            )
            assignment = load_transplant_episode_assignments(
                cohort, procedures_file, admissions_file
            )[21]
            self.assertEqual(assignment.anchor_quality_level, "low")
            self.assertFalse(assignment.phase_specific_context_allowed)

    def test_04_timeline_resets_clock_at_retransplant_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admissions_file, procedures_file = self._write_common_files(root)
            cohort = pd.DataFrame(
                [
                    _episode_row(1, 10, "1-TX1", 1, 10, "2020-01-02"),
                    _episode_row(1, 20, "1-TX2", 2, 20, "2024-01-02"),
                ]
            )
            cohort_file = root / "cohort.csv"
            cohort.to_csv(cohort_file, index=False)
            pd.DataFrame(
                {
                    "subject_id": [1],
                    "anchor_age": [40],
                    "anchor_year": [2020],
                    "gender": ["F"],
                }
            ).to_csv(root / "patients.csv", index=False)
            pd.DataFrame(
                {
                    "subject_id": [1, 1],
                    "hadm_id": [10, 20],
                    "drug": ["Tacrolimus", "Tacrolimus"],
                    "starttime": ["2020-01-03 08:00:00", "2024-01-03 08:00:00"],
                    "stoptime": ["2020-01-09 08:00:00", "2024-01-09 08:00:00"],
                    "route": ["PO", "PO"],
                }
            ).to_csv(root / "prescriptions.csv", index=False)
            pd.DataFrame(
                {
                    "subject_id": [1, 1],
                    "hadm_id": [10, 20],
                    "itemid": [50912, 50912],
                    "charttime": ["2020-01-03 06:00:00", "2024-01-03 06:00:00"],
                    "valuenum": [1.0, 1.2],
                }
            ).to_csv(root / "labevents.csv", index=False)

            timeline = build_order_timeline(
                cohort_file,
                root / "patients.csv",
                admissions_file,
                root / "prescriptions.csv",
                root / "labevents.csv",
                procedures_file,
                None,
                None,
                None,
                ReferenceStore(ROOT / "reference"),
                expected_patients=1,
            )
            second = timeline[timeline["hadm_id"].eq(20)].iloc[0]
            self.assertEqual(second["transplant_episode_id"], "1-TX2")
            self.assertEqual(int(second["transplant_episode_number"]), 2)
            self.assertEqual(second["transplant_phase"], "index_admission_postprocedure")
            self.assertEqual(int(second["days_since_transplant"]), 1)
            self.assertTrue(bool(second["post_transplant_phase_context_allowed"]))
            self.assertEqual(second["transplant_protocol_phase"], "index_transplant_admission")

    def test_05_preprocedure_is_retained_but_posttransplant_narrowing_is_blocked(self) -> None:
        anchor = TransplantAnchor(
            1,
            pd.Timestamp("2024-01-02"),
            20,
            "observed",
            "0TY10Z0",
            10,
        )
        phase, days = classify_transplant_phase(
            pd.Timestamp("2024-01-01 12:00:00"), 20, anchor
        )
        self.assertEqual(phase, "index_admission_preprocedure")
        self.assertEqual(days, -1)
        self.assertFalse(post_transplant_phase_context_allowed(phase, anchor))
        self.assertEqual(protocol_phase_for_context(phase, False), "prevalent_phase_unknown")

    def test_06_transplant_day_is_time_uncertain_and_conservative(self) -> None:
        anchor = TransplantAnchor(
            1,
            pd.Timestamp("2024-01-02"),
            20,
            "observed",
            "0TY10Z0",
            10,
        )
        phase, days = classify_transplant_phase(
            pd.Timestamp("2024-01-02 18:00:00"), 20, anchor
        )
        self.assertEqual(phase, "transplant_day_time_uncertain")
        self.assertEqual(days, 0)
        self.assertFalse(post_transplant_phase_context_allowed(phase, anchor))
        self.assertEqual(protocol_phase_for_context(phase, False), "prevalent_phase_unknown")

    def test_07_postprocedure_index_phase_maps_to_registered_protocol_scope(self) -> None:
        anchor = TransplantAnchor(
            1,
            pd.Timestamp("2024-01-02"),
            20,
            "observed",
            "0TY10Z0",
            10,
        )
        phase, days = classify_transplant_phase(
            pd.Timestamp("2024-01-03 08:00:00"), 20, anchor
        )
        allowed = post_transplant_phase_context_allowed(phase, anchor)
        self.assertEqual(phase, "index_admission_postprocedure")
        self.assertEqual(days, 1)
        self.assertTrue(allowed)
        self.assertEqual(
            protocol_phase_for_context(phase, allowed),
            "index_transplant_admission",
        )

    def test_08_timeline_emits_explicit_peri_and_protocol_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admissions_file, procedures_file = self._write_common_files(root)
            cohort = pd.DataFrame(
                [_episode_row(1, 20, "1-TX2", 2, 20, "2024-01-02")]
            )
            cohort_file = root / "cohort.csv"
            cohort.to_csv(cohort_file, index=False)
            pd.DataFrame(
                {
                    "subject_id": [1],
                    "anchor_age": [40],
                    "anchor_year": [2020],
                    "gender": ["F"],
                }
            ).to_csv(root / "patients.csv", index=False)
            pd.DataFrame(
                {
                    "subject_id": [1, 1, 1],
                    "hadm_id": [20, 20, 20],
                    "drug": ["Tacrolimus", "Prednisone", "Mycophenolate"],
                    "starttime": [
                        "2024-01-01 08:00:00",
                        "2024-01-02 08:00:00",
                        "2024-01-03 08:00:00",
                    ],
                    "stoptime": [
                        "2024-01-09 08:00:00",
                        "2024-01-09 08:00:00",
                        "2024-01-09 08:00:00",
                    ],
                    "route": ["PO", "PO", "PO"],
                }
            ).to_csv(root / "prescriptions.csv", index=False)
            pd.DataFrame(
                {
                    "subject_id": [1],
                    "hadm_id": [20],
                    "itemid": [50912],
                    "charttime": ["2024-01-01 06:00:00"],
                    "valuenum": [1.0],
                }
            ).to_csv(root / "labevents.csv", index=False)

            timeline = build_order_timeline(
                cohort_file,
                root / "patients.csv",
                admissions_file,
                root / "prescriptions.csv",
                root / "labevents.csv",
                procedures_file,
                None,
                None,
                None,
                ReferenceStore(ROOT / "reference"),
                expected_patients=1,
            ).sort_values("timestamp")

            pre, day0, post = [row for _, row in timeline.iterrows()]
            self.assertEqual(pre["transplant_phase"], "index_admission_preprocedure")
            self.assertTrue(bool(pre["peri_transplant_order"]))
            self.assertFalse(bool(pre["post_transplant_phase_context_allowed"]))
            self.assertEqual(pre["transplant_protocol_phase"], "prevalent_phase_unknown")

            self.assertEqual(day0["transplant_phase"], "transplant_day_time_uncertain")
            self.assertTrue(bool(day0["peri_transplant_order"]))
            self.assertFalse(bool(day0["post_transplant_phase_context_allowed"]))
            self.assertEqual(day0["transplant_protocol_phase"], "prevalent_phase_unknown")

            self.assertEqual(post["transplant_phase"], "index_admission_postprocedure")
            self.assertFalse(bool(post["peri_transplant_order"]))
            self.assertTrue(bool(post["post_transplant_phase_context_allowed"]))
            self.assertEqual(post["transplant_protocol_phase"], "index_transplant_admission")


if __name__ == "__main__":
    unittest.main(verbosity=2)
