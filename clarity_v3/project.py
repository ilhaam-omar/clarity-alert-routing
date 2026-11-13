from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .config import read_json


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    cohort_file: Path
    patients_file: Path
    admissions_file: Path
    prescriptions_file: Path
    poe_file: Optional[Path]
    labevents_file: Path
    procedures_icd_file: Optional[Path]
    d_icd_procedures_file: Optional[Path]
    diagnoses_icd_file: Optional[Path]
    transfers_file: Optional[Path]
    services_file: Optional[Path]
    procedureevents_file: Optional[Path]
    d_labitems_file: Optional[Path]
    omr_file: Optional[Path]
    pharmacy_file: Optional[Path]
    emar_file: Optional[Path]
    note_results_file: Optional[Path]
    discharge_notes_file: Optional[Path]
    note_targets: tuple[str, ...]
    note_context_mode: str
    reference_dir: Path
    output_dir: Path
    private_dir: Path
    evidence_dir: Path
    expected_patients: Optional[int]
    mimic_version: str
    cohort_definition: str
    decision_time_mode: str

    @property
    def cohort_audit_file(self) -> Path:
        return self.evidence_dir / "00a_cohort_audit.json"

    @property
    def design_freeze_file(self) -> Path:
        return self.evidence_dir / "00b_design_freeze.json"

    @property
    def orders_file(self) -> Path:
        return self.private_dir / "01_order_timeline.csv"

    @property
    def signals_file(self) -> Path:
        return self.private_dir / "02_detected_hazards.csv"


    @property
    def note_tasks_file(self) -> Path:
        return self.private_dir / "02a_note_extraction_tasks.csv"

    @property
    def generated_note_results_file(self) -> Path:
        return self.private_dir / "02b_note_context_facts.csv"

    @property
    def note_checkpoint_file(self) -> Path:
        return self.private_dir / "02b_note_context_checkpoint.jsonl"

    @property
    def note_validation_file(self) -> Path:
        return self.evidence_dir / "02c_note_context_validation.json"

    def effective_note_results_file(self) -> Optional[Path]:
        if self.note_results_file is not None and self.note_results_file.exists():
            return self.note_results_file
        if self.generated_note_results_file.exists():
            return self.generated_note_results_file
        return None

    @property
    def states_file(self) -> Path:
        return self.private_dir / "03_longitudinal_states.csv"

    @property
    def decisions_file(self) -> Path:
        return self.private_dir / "04_action_sets.csv"

    @property
    def bundles_file(self) -> Path:
        return self.private_dir / "05_order_bundles.csv"

    @property
    def final_bundles_file(self) -> Path:
        return self.private_dir / "06_final_bundle_decisions.csv"

    @property
    def explanations_file(self) -> Path:
        return self.private_dir / "08_explanations.csv"

    @property
    def parameter_design_file(self) -> Path:
        return self.evidence_dir / "uncertainty_design_points.csv"


def _resolve(root: Path, value: Any, required: bool = True) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        if required:
            raise ValueError("A required project path is empty")
        return None
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


def load_project_paths(config_path: str | Path) -> ProjectPaths:
    source = Path(config_path).resolve()
    values: Dict[str, Any] = read_json(source)
    root = _resolve(source.parent, values.get("project_root", "."))
    assert root is not None
    output = _resolve(root, values.get("output_dir", "outputs"))
    assert output is not None
    note_context_mode = str(values.get("note_context_mode", "disabled"))
    if note_context_mode not in {"disabled", "conservative", "secondary_analysis"}:
        raise ValueError(
            "note_context_mode must be disabled, conservative, or secondary_analysis"
        )
    return ProjectPaths(
        root=root,
        cohort_file=_resolve(root, values.get("cohort_file")),  # type: ignore[arg-type]
        patients_file=_resolve(root, values.get("patients_file")),  # type: ignore[arg-type]
        admissions_file=_resolve(root, values.get("admissions_file")),  # type: ignore[arg-type]
        prescriptions_file=_resolve(root, values.get("prescriptions_file")),  # type: ignore[arg-type]
        poe_file=_resolve(root, values.get("poe_file"), required=False),
        labevents_file=_resolve(root, values.get("labevents_file")),  # type: ignore[arg-type]
        procedures_icd_file=_resolve(root, values.get("procedures_icd_file"), required=False),
        d_icd_procedures_file=_resolve(root, values.get("d_icd_procedures_file"), required=False),
        diagnoses_icd_file=_resolve(root, values.get("diagnoses_icd_file"), required=False),
        transfers_file=_resolve(root, values.get("transfers_file"), required=False),
        services_file=_resolve(root, values.get("services_file"), required=False),
        procedureevents_file=_resolve(root, values.get("procedureevents_file"), required=False),
        d_labitems_file=_resolve(root, values.get("d_labitems_file"), required=False),
        omr_file=_resolve(root, values.get("omr_file"), required=False),
        pharmacy_file=_resolve(root, values.get("pharmacy_file"), required=False),
        emar_file=_resolve(root, values.get("emar_file"), required=False),
        note_results_file=_resolve(root, values.get("note_results_file"), required=False),
        discharge_notes_file=_resolve(root, values.get("discharge_notes_file"), required=False),
        note_targets=tuple(
            str(v) for v in values.get("note_targets", ["documented_interaction_awareness"])
        ),
        note_context_mode=note_context_mode,
        reference_dir=_resolve(root, values.get("reference_dir", "reference")),  # type: ignore[arg-type]
        output_dir=output,
        private_dir=output / "private",
        evidence_dir=output / "evidence",
        expected_patients=(
            int(values["expected_patients"])
            if values.get("expected_patients") is not None
            else None
        ),
        mimic_version=str(values.get("mimic_version", "UNSPECIFIED")),
        cohort_definition=str(
            values.get(
                "cohort_definition",
                "Kidney transplant cohort derived by user-specified ICD diagnosis/procedure logic",
            )
        ),
        decision_time_mode=str(values.get("decision_time_mode", "prescription_starttime")),
    )


def ensure_output_dirs(paths: ProjectPaths) -> None:
    paths.private_dir.mkdir(parents=True, exist_ok=True)
    paths.evidence_dir.mkdir(parents=True, exist_ok=True)
