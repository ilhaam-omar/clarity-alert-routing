from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from .models import DrugClassification


class ReferenceSchemaError(ValueError):
    """Raised when a reference file is missing or structurally invalid."""


REQUIRED_COLUMNS: Dict[str, tuple[str, ...]] = {
    "01_drug_dictionary.csv": (
        "raw_drug_name",
        "canonical_ingredient",
        "ingredient_key",
        "is_drug",
        "is_iv_fluid_or_diluent",
        "is_device_or_admin_artifact",
        "is_vaccine",
    ),
    "02_drug_normalization_corrections.csv": (
        "raw_drug_name",
        "canonical_ingredient",
        "ingredient_key",
        "normalization_status",
    ),
    "04_ddi_rules.csv": (
        "rule_id",
        "rule_family",
        "drug_a_key",
        "drug_b_key",
        "mechanism",
        "recommended_action",
        "source_id",
        "active_flag",
        "signal_confidence_numeric",
        "nccmerp_category",
        "severity_score_nccmerp",
    ),
    "05_renal_dose_rules.csv": (
        "rule_id",
        "rule_family",
        "ingredient_key",
        "threshold_operator",
        "threshold_value",
        "recommended_action",
        "source_id",
        "active_flag",
        "signal_confidence_numeric",
        "nccmerp_category",
        "severity_score_nccmerp",
    ),
    "07_duplicate_therapy_rules.csv": (
        "rule_id",
        "rule_family",
        "therapy_class",
        "member_ingredient_keys_pipe",
        "allowed_overlap_hours",
        "recommended_action",
        "source_id",
        "active_flag",
        "signal_confidence_numeric",
        "nccmerp_category",
        "severity_score_nccmerp",
    ),
    "08_beers_rules.csv": (
        "rule_id",
        "ingredient_key",
        "age_min_years",
        "rationale_short_paraphrased",
        "recommended_action",
        "source_id",
        "active_flag",
        "signal_confidence_numeric",
        "nccmerp_category",
        "severity_score_nccmerp",
    ),
    "20_rule_source_register.csv": ("source_id", "source_label", "source_url"),
    "transplant_protocol_rules.csv": (
        "rule_id",
        "drug_1",
        "drug_2",
        "protocol_expected_default",
        "required_monitoring",
        "monitoring_availability_flag",
    ),
    "21_transplant_context_overrides.csv": (
        "override_id",
        "transplant_rule_id",
        "target_rule_ids",
        "override_type",
        "override_condition",
        "resulting_action_when_condition_met",
    ),
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ReferenceSchemaError(f"Missing required reference file: {path}")
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    except Exception as exc:  # pragma: no cover - pandas exposes many parser types
        raise ReferenceSchemaError(f"Unable to parse {path.name}: {exc}") from exc


def _validate_columns(filename: str, df: pd.DataFrame) -> None:
    required = REQUIRED_COLUMNS.get(filename, ())
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ReferenceSchemaError(
            f"{filename} is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _active_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "active_flag" not in df.columns:
        return df.reset_index(drop=True)
    active = pd.to_numeric(df["active_flag"], errors="coerce").fillna(0).eq(1)
    return df.loc[active].reset_index(drop=True)


def _normalization_category(status: object) -> str:
    value = str(status).lower()
    if "iv_fluid" in value or "diluent" in value:
        return "IV_FLUID"
    if any(term in value for term in ("artifact", "device", "flush", "priming")):
        return "ADMIN_ARTIFACT"
    if "vaccine" in value:
        return "VACCINE"
    return "DRUG"


class ReferenceStore:
    """Loads the frozen rule/reference layer and exposes deterministic lookup APIs."""

    def __init__(self, reference_dir: str | Path):
        self.reference_dir = Path(reference_dir).resolve()

        self.drug_dictionary = self._load("01_drug_dictionary.csv")
        self.corrections = self._load("02_drug_normalization_corrections.csv")
        self.ddi_rules = _active_rows(self._load("04_ddi_rules.csv"))
        self.renal_rules = _active_rows(self._load("05_renal_dose_rules.csv"))
        self.duplicate_rules = _active_rows(self._load("07_duplicate_therapy_rules.csv"))
        self.beers_rules = _active_rows(self._load("08_beers_rules.csv"))
        self.source_register = self._load("20_rule_source_register.csv")
        self.transplant_protocol_rules = self._load("transplant_protocol_rules.csv")
        self.transplant_overrides = self._load("21_transplant_context_overrides.csv")
        phase_path = self.reference_dir / "22_transplant_phase_scope.csv"
        self.transplant_phase_scope = _read_csv(phase_path) if phase_path.exists() else pd.DataFrame()

        monitoring_path = self.reference_dir / "monitoring_requirements.csv"
        self.monitoring_requirements = (
            _read_csv(monitoring_path) if monitoring_path.exists() else pd.DataFrame()
        )
        mechanisms_path = self.reference_dir / "duplicate_mechanism_rules.csv"
        self.duplicate_mechanisms = (
            _read_csv(mechanisms_path) if mechanisms_path.exists() else pd.DataFrame()
        )

        self._validate_rule_values()
        self._validate_sources()
        self._classification_lookup = self._build_classification_lookup()
        self._rule_lookup = self._build_rule_lookup()
        self._transplant_target_lookup = self._build_transplant_target_lookup()

    def _load(self, filename: str) -> pd.DataFrame:
        df = _read_csv(self.reference_dir / filename)
        _validate_columns(filename, df)
        return df

    def _validate_rule_values(self) -> None:
        tables = {
            "04_ddi_rules.csv": self.ddi_rules,
            "05_renal_dose_rules.csv": self.renal_rules,
            "07_duplicate_therapy_rules.csv": self.duplicate_rules,
            "08_beers_rules.csv": self.beers_rules,
        }
        for filename, table in tables.items():
            if table["rule_id"].duplicated().any():
                ids = table.loc[table["rule_id"].duplicated(keep=False), "rule_id"].tolist()
                raise ReferenceSchemaError(f"Duplicate active rule_id values in {filename}: {ids}")
            for column in ("signal_confidence_numeric", "severity_score_nccmerp", "nccmerp_category"):
                if table[column].isna().any():
                    ids = table.loc[table[column].isna(), "rule_id"].tolist()
                    raise ReferenceSchemaError(
                        f"{filename} has null {column!r} for active rule(s): {ids}"
                    )
            confidence = pd.to_numeric(table["signal_confidence_numeric"], errors="coerce")
            if confidence.isna().any() or ((confidence < 0) | (confidence > 1)).any():
                raise ReferenceSchemaError(
                    f"{filename} has signal_confidence_numeric outside [0, 1] or nonnumeric values."
                )
            severity = pd.to_numeric(table["severity_score_nccmerp"], errors="coerce")
            if severity.isna().any() or (severity < 0).any():
                raise ReferenceSchemaError(
                    f"{filename} has invalid severity_score_nccmerp values."
                )

        self_pairs = self.ddi_rules[
            self.ddi_rules["drug_a_key"].astype(str)
            == self.ddi_rules["drug_b_key"].astype(str)
        ]
        if not self_pairs.empty:
            raise ReferenceSchemaError(
                "DDI table contains self-pairs: "
                + ", ".join(self_pairs["rule_id"].astype(str).tolist())
            )

    def _validate_sources(self) -> None:
        known = set(self.source_register["source_id"].dropna().astype(str))
        missing: list[str] = []
        for table in (
            self.ddi_rules,
            self.renal_rules,
            self.duplicate_rules,
            self.beers_rules,
        ):
            for value in table["source_id"].dropna().astype(str):
                if value not in known:
                    missing.append(value)
        if missing:
            raise ReferenceSchemaError(
                "Active rules reference unregistered source_id values: "
                + ", ".join(sorted(set(missing)))
            )

    @staticmethod
    def _pick_rows(df: pd.DataFrame, key_column: str, volume_column: str) -> pd.DataFrame:
        working = df.copy()
        working[key_column] = working[key_column].astype(str).str.strip().str.lower()
        if "is_canonical_row" in working.columns:
            canonical = pd.to_numeric(working["is_canonical_row"], errors="coerce").fillna(0)
            if (working.assign(_canonical=canonical).groupby(key_column)["_canonical"].sum() > 1).any():
                raise ReferenceSchemaError(
                    f"More than one is_canonical_row=1 value exists for a {key_column} group."
                )
            working["_canonical"] = canonical
        else:
            working["_canonical"] = 0
        if volume_column not in working.columns:
            working[volume_column] = 0
        working[volume_column] = pd.to_numeric(working[volume_column], errors="coerce").fillna(0)
        return (
            working.sort_values(
                [key_column, "_canonical", volume_column],
                ascending=[True, False, False],
                kind="stable",
            )
            .drop_duplicates(key_column, keep="first")
            .reset_index(drop=True)
        )

    def _build_classification_lookup(self) -> Dict[str, DrugClassification]:
        lookup: Dict[str, DrugClassification] = {}

        dd = self.drug_dictionary.copy()
        dd["_lookup_key"] = dd["raw_drug_name"].astype(str).str.strip().str.lower()
        for _, row in self._pick_rows(dd, "_lookup_key", "n_orders").iterrows():
            is_drug = int(row.get("is_drug", 0) or 0) == 1
            is_iv = int(row.get("is_iv_fluid_or_diluent", 0) or 0) == 1
            is_artifact = int(row.get("is_device_or_admin_artifact", 0) or 0) == 1
            is_vaccine = int(row.get("is_vaccine", 0) or 0) == 1
            if is_drug:
                category = "DRUG"
            elif is_iv:
                category = "IV_FLUID"
            elif is_artifact:
                category = "ADMIN_ARTIFACT"
            elif is_vaccine:
                category = "VACCINE"
            else:
                category = "UNMAPPED"
            lookup[str(row["_lookup_key"])] = DrugClassification(
                raw_name=str(row["raw_drug_name"]),
                ingredient_key=(
                    str(row["ingredient_key"]) if pd.notna(row["ingredient_key"]) else None
                ),
                canonical_ingredient=(
                    str(row["canonical_ingredient"])
                    if pd.notna(row["canonical_ingredient"])
                    else None
                ),
                category=category,
            )

        corrections = self.corrections.copy()
        corrections["_lookup_key"] = (
            corrections["raw_drug_name"].astype(str).str.strip().str.lower()
        )
        for _, row in self._pick_rows(corrections, "_lookup_key", "n_patients").iterrows():
            lookup[str(row["_lookup_key"])] = DrugClassification(
                raw_name=str(row["raw_drug_name"]),
                ingredient_key=(
                    str(row["ingredient_key"]) if pd.notna(row["ingredient_key"]) else None
                ),
                canonical_ingredient=(
                    str(row["canonical_ingredient"])
                    if pd.notna(row["canonical_ingredient"])
                    else None
                ),
                category=_normalization_category(row["normalization_status"]),
            )
        return lookup

    def _build_rule_lookup(self) -> Dict[str, pd.Series]:
        result: Dict[str, pd.Series] = {}
        for table in (
            self.ddi_rules,
            self.renal_rules,
            self.duplicate_rules,
            self.beers_rules,
        ):
            for _, row in table.iterrows():
                result[str(row["rule_id"])] = row
        return result

    def _build_transplant_target_lookup(self) -> Dict[str, pd.Series]:
        result: Dict[str, pd.Series] = {}
        for _, row in self.transplant_overrides.iterrows():
            value = row.get("target_rule_ids")
            if pd.isna(value):
                continue
            for rule_id in str(value).split("|"):
                key = rule_id.strip()
                if key:
                    result[key] = row
        return result

    def classify(self, raw_name: object) -> DrugClassification:
        key = str(raw_name).strip().lower()
        return self._classification_lookup.get(
            key,
            DrugClassification(
                raw_name=str(raw_name),
                ingredient_key=None,
                canonical_ingredient=None,
                category="UNMAPPED",
            ),
        )

    def rule(self, rule_id: str) -> pd.Series:
        try:
            return self._rule_lookup[str(rule_id)]
        except KeyError as exc:
            raise ReferenceSchemaError(f"Unknown rule_id: {rule_id}") from exc

    def transplant_override(self, rule_id: str) -> Optional[pd.Series]:
        return self._transplant_target_lookup.get(str(rule_id))

    def active_rule_counts(self) -> Dict[str, int]:
        return {
            "drug_dictionary": int(len(self.drug_dictionary)),
            "normalization_corrections": int(len(self.corrections)),
            "ddi_rules": int(len(self.ddi_rules)),
            "renal_rules": int(len(self.renal_rules)),
            "duplicate_rules": int(len(self.duplicate_rules)),
            "beers_rules": int(len(self.beers_rules)),
            "source_register": int(len(self.source_register)),
            "transplant_protocol_rules": int(len(self.transplant_protocol_rules)),
            "transplant_overrides": int(len(self.transplant_overrides)),
            "transplant_phase_scope": int(len(self.transplant_phase_scope)),
        }

    def all_rule_ids(self) -> set[str]:
        return set(self._rule_lookup)

    def required_files(self) -> Iterable[Path]:
        for filename in REQUIRED_COLUMNS:
            yield self.reference_dir / filename
