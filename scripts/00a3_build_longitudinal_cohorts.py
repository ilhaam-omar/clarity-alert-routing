from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clarity_v3.project import ensure_output_dirs, load_project_paths


def _clean_int(series: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_numeric(series, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{name} contains {int(parsed.isna().sum())} invalid value(s)")
    return parsed.astype("int64")


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_config(
    *,
    source_config: Path,
    destination: Path,
    root: Path,
    cohort_file: Path,
    output_dir: str,
    cohort_definition: str,
    scope_name: str,
) -> None:
    values: dict[str, Any] = json.loads(source_config.read_text(encoding="utf-8"))
    values["cohort_file"] = _relative_to_root(cohort_file, root)
    values["output_dir"] = output_dir
    values["expected_patients"] = 771
    values["cohort_definition"] = cohort_definition
    values["analysis_scope"] = scope_name
    values["transplant_episode_policy"] = (
        "Distinct exact transplant-coded admissions are separate candidate transplant episodes; "
        "each non-index admission is assigned to the most recent preceding episode."
    )
    values["transplant_anchor_quality_policy"] = (
        "Exact single in-admission chartdate preferred; earliest valid date used for multiple valid "
        "dates; admission date used as flagged proxy when chartdate lies outside the stay."
    )
    values["note_results_file"] = None
    values["discharge_notes_file"] = None
    values["note_targets"] = []
    values["note_context_mode"] = "disabled"
    destination.write_text(json.dumps(values, indent=2), encoding="utf-8")


def _prepare_cohort(assignments: pd.DataFrame, episodes: pd.DataFrame, mask: pd.Series, scope: str) -> pd.DataFrame:
    selected = assignments.loc[mask].copy()
    episode_detail = episodes[
        [
            "subject_id",
            "episode_number",
            "hadm_id",
            "candidate_anchor_day",
            "anchor_quality",
            "days_since_previous_transplant_episode",
        ]
    ].rename(
        columns={
            "hadm_id": "transplant_hadm_id",
            "candidate_anchor_day": "transplant_date",
        }
    )
    selected = selected.merge(
        episode_detail,
        on=["subject_id", "episode_number"],
        how="left",
        validate="many_to_one",
    )
    if selected["transplant_hadm_id"].isna().any():
        raise ValueError("Some selected admissions could not be linked to a transplant episode")

    selected["subject_id"] = _clean_int(selected["subject_id"], "subject_id")
    selected["hadm_id"] = _clean_int(selected["hadm_id"], "hadm_id")
    selected["transplant_hadm_id"] = _clean_int(
        selected["transplant_hadm_id"], "transplant_hadm_id"
    )
    selected["episode_number"] = _clean_int(selected["episode_number"], "episode_number")
    selected["transplant_date"] = pd.to_datetime(selected["transplant_date"], errors="coerce")
    if selected["transplant_date"].isna().any():
        raise ValueError("Some selected admissions have no usable transplant anchor date")

    selected["transplant_episode_id"] = (
        selected["subject_id"].astype(str)
        + "-TX"
        + selected["episode_number"].astype(str)
    )
    selected["cohort_scope"] = scope
    selected["anchor_quality_level"] = selected["anchor_quality"].map(
        {
            "exact_single_in_admission": "high",
            "exact_multiple_dates_same_admission_earliest_used": "moderate",
            "admission_date_proxy_invalid_chartdate": "low",
        }
    ).fillna("unknown")
    selected["short_gap_repeat_transplant_candidate"] = (
        selected["episode_number"].gt(1)
        & pd.to_numeric(
            selected["days_since_previous_transplant_episode"], errors="coerce"
        ).lt(90)
    )
    selected["phase_specific_context_allowed"] = (
        (
            selected["relation"].eq("transplant_episode_admission")
            | selected["anchor_quality"].eq("exact_single_in_admission")
        )
        & ~selected["short_gap_repeat_transplant_candidate"]
    )

    columns = [
        "subject_id",
        "hadm_id",
        "transplant_episode_id",
        "episode_number",
        "transplant_hadm_id",
        "transplant_date",
        "anchor_quality",
        "anchor_quality_level",
        "phase_specific_context_allowed",
        "days_since_previous_transplant_episode",
        "short_gap_repeat_transplant_candidate",
        "relation",
        "days_from_assigned_anchor",
        "assigned_phase",
        "cohort_scope",
    ]
    selected = selected[columns].drop_duplicates(["subject_id", "hadm_id"])
    return selected.sort_values(
        ["subject_id", "transplant_date", "hadm_id"], kind="stable"
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the index-only, first-year longitudinal, and all-post-transplant "
            "MIMIC Phase-1 cohorts from the audited transplant-episode assignments."
        )
    )
    parser.add_argument("--paths", required=True, help="Active real MIMIC configuration")
    args = parser.parse_args()

    config_path = Path(args.paths).resolve()
    paths = load_project_paths(config_path)
    ensure_output_dirs(paths)

    assignment_file = paths.private_dir / "00a2_all_admission_assignments_private.csv"
    episode_file = paths.private_dir / "00a2_transplant_episodes_private.csv"
    if not assignment_file.exists() or not episode_file.exists():
        raise FileNotFoundError(
            "Stage 00A.2 private assignment files are required. Run 00a2_transplant_episode_scope.py first."
        )

    assignments = pd.read_csv(assignment_file, low_memory=False)
    episodes = pd.read_csv(episode_file, low_memory=False)
    for column in ["subject_id", "hadm_id", "episode_number"]:
        assignments[column] = pd.to_numeric(assignments[column], errors="coerce")
    for column in ["subject_id", "hadm_id", "episode_number"]:
        episodes[column] = pd.to_numeric(episodes[column], errors="coerce")
    assignments = assignments.dropna(subset=["subject_id", "hadm_id", "episode_number"]).copy()
    episodes = episodes.dropna(subset=["subject_id", "hadm_id", "episode_number"]).copy()
    assignments = assignments.astype(
        {"subject_id": "int64", "hadm_id": "int64", "episode_number": "int64"}
    )
    episodes = episodes.astype(
        {"subject_id": "int64", "hadm_id": "int64", "episode_number": "int64"}
    )

    days = pd.to_numeric(assignments["days_from_assigned_anchor"], errors="coerce")
    index_mask = assignments["relation"].eq("transplant_episode_admission")
    first_year_mask = index_mask | (
        assignments["relation"].eq("post_transplant_nonindex_admission")
        & days.between(0, 365, inclusive="both")
    )
    all_post_mask = assignments["relation"].isin(
        ["transplant_episode_admission", "post_transplant_nonindex_admission"]
    )

    index = _prepare_cohort(assignments, episodes, index_mask, "index_transplant_only")
    first_year = _prepare_cohort(
        assignments, episodes, first_year_mask, "index_plus_post_transplant_0_365d"
    )
    all_post = _prepare_cohort(
        assignments, episodes, all_post_mask, "all_observed_post_transplant_admissions"
    )

    cohort_dir = paths.root / "data_processed" / "cohorts"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    index_file = cohort_dir / "kidney_transplant_cohort_index_only.csv"
    first_year_file = cohort_dir / "kidney_transplant_cohort_phase1_365d.csv"
    all_post_file = cohort_dir / "kidney_transplant_cohort_all_post_transplant.csv"
    index.to_csv(index_file, index=False)
    first_year.to_csv(first_year_file, index=False)
    all_post.to_csv(all_post_file, index=False)

    config_dir = config_path.parent
    index_config = config_dir / "mimic_paths.phase1.index.local.json"
    first_year_config = config_dir / "mimic_paths.phase1.365d.local.json"
    all_post_config = config_dir / "mimic_paths.phase1.all_post.local.json"

    _write_config(
        source_config=config_path,
        destination=index_config,
        root=paths.root,
        cohort_file=index_file,
        output_dir="outputs_mimic_phase1_index",
        cohort_definition=(
            "MIMIC-IV 3.1 kidney-transplant index-admission cohort: one row per exact "
            "transplant-coded admission; distinct transplant-coded admissions are separate episodes."
        ),
        scope_name="index_only_snapshot_sensitivity",
    )
    _write_config(
        source_config=config_path,
        destination=first_year_config,
        root=paths.root,
        cohort_file=first_year_file,
        output_dir="outputs_mimic_phase1_365d",
        cohort_definition=(
            "Primary episode-based longitudinal inpatient cohort: each exact transplant-coded "
            "index admission plus all subsequent MIMIC admissions through 365 days, assigned to "
            "the most recent preceding transplant episode."
        ),
        scope_name="primary_first_year_inpatient_longitudinal",
    )
    _write_config(
        source_config=config_path,
        destination=all_post_config,
        root=paths.root,
        cohort_file=all_post_file,
        output_dir="outputs_mimic_phase1_all_post",
        cohort_definition=(
            "Secondary extended episode-based cohort: every exact transplant-coded index admission "
            "plus all subsequent observed MIMIC admissions, assigned to the most recent preceding "
            "transplant episode."
        ),
        scope_name="secondary_all_observed_post_transplant",
    )

    summary = {
        "patients": int(first_year["subject_id"].nunique()),
        "transplant_episodes": int(index["transplant_episode_id"].nunique()),
        "index_only_admissions": int(len(index)),
        "primary_365d_admissions": int(len(first_year)),
        "primary_365d_nonindex_admissions": int(
            first_year["relation"].eq("post_transplant_nonindex_admission").sum()
        ),
        "primary_365d_patients_with_nonindex_followup": int(
            first_year.loc[
                first_year["relation"].eq("post_transplant_nonindex_admission"),
                "subject_id",
            ].nunique()
        ),
        "all_post_transplant_admissions": int(len(all_post)),
        "all_post_transplant_nonindex_admissions": int(
            all_post["relation"].eq("post_transplant_nonindex_admission").sum()
        ),
        "all_post_patients_with_nonindex_followup": int(
            all_post.loc[
                all_post["relation"].eq("post_transplant_nonindex_admission"),
                "subject_id",
            ].nunique()
        ),
        "episodes_high_quality_anchor": int(
            index["anchor_quality_level"].eq("high").sum()
        ),
        "episodes_moderate_quality_anchor": int(
            index["anchor_quality_level"].eq("moderate").sum()
        ),
        "episodes_low_quality_anchor": int(
            index["anchor_quality_level"].eq("low").sum()
        ),
        "short_gap_repeat_transplant_episodes": int(
            index["short_gap_repeat_transplant_candidate"].astype(bool).sum()
        ),
        "pretransplant_admissions_excluded_from_action_denominator": int(
            assignments["relation"].eq("admission_before_first_observed_transplant").sum()
        ),
    }

    if summary["patients"] != 771:
        raise RuntimeError(f"Expected 771 patients in the primary cohort, found {summary['patients']}")
    if summary["transplant_episodes"] != len(index):
        raise RuntimeError("Index cohort does not contain exactly one row per transplant episode")
    if first_year.duplicated(["subject_id", "hadm_id"]).any():
        raise RuntimeError("Primary cohort contains duplicate patient-admission pairs")
    if all_post.duplicated(["subject_id", "hadm_id"]).any():
        raise RuntimeError("Extended cohort contains duplicate patient-admission pairs")

    evidence_file = paths.evidence_dir / "00a3_longitudinal_cohort_build.json"
    evidence_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    phase_summary = (
        first_year.groupby(["relation", "assigned_phase", "anchor_quality_level"], dropna=False)
        .size()
        .reset_index(name="admission_count")
        .sort_values(["relation", "assigned_phase", "anchor_quality_level"], kind="stable")
    )
    phase_file = paths.evidence_dir / "00a3_primary_365d_phase_distribution.csv"
    phase_summary.to_csv(phase_file, index=False)

    print("=" * 78)
    print("STAGE 00A.3 - BUILD EPISODE-BASED LONGITUDINAL COHORTS")
    print("=" * 78)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nPrimary 365-day phase distribution:")
    print(phase_summary.to_string(index=False))
    print(f"\nIndex-only cohort: {index_file}")
    print(f"Primary first-year cohort: {first_year_file}")
    print(f"Extended all-post cohort: {all_post_file}")
    print(f"Primary configuration: {first_year_config}")
    print(f"Index sensitivity configuration: {index_config}")
    print(f"Long-term sensitivity configuration: {all_post_config}")
    print(f"Aggregate evidence: {evidence_file}")
    print(f"Aggregate phase distribution: {phase_file}")
    print("\nInterpretation:")
    print(" - The primary Phase-1 cohort is inpatient-only and covers index admission through 365 days.")
    print(" - Pretransplant admissions are excluded from action denominators but remain available as prior history.")
    print(" - Imperfect anchors are retained with explicit quality flags and must be included in sensitivity analysis.")
    print("\nSTAGE 00A.3: COMPLETE - PASS")


if __name__ == "__main__":
    main()
