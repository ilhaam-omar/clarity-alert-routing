from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd

_TRUE = {"true", "1", "yes", "y", "t"}
CURRENT_INTENT_TARGETS = {
    "documented_clinical_intent",
    "documented_interaction_awareness",  # legacy synonym
}
CURRENT_MONITORING_TARGETS = {"documented_monitoring_plan"}
DURABLE_ADVERSE_TARGETS = {"prior_adverse_reaction_or_intolerance"}
PRIOR_MANAGEMENT_TARGETS = {"prior_interaction_management_history"}
ALL_CONTEXT_TARGETS = tuple(
    sorted(
        CURRENT_INTENT_TARGETS
        | CURRENT_MONITORING_TARGETS
        | DURABLE_ADVERSE_TARGETS
        | PRIOR_MANAGEMENT_TARGETS
    )
)


@dataclass(frozen=True)
class NoteFact:
    """A source-verified note fact available to the longitudinal state.

    ``hadm_id`` is the target decision admission. ``source_hadm_id`` records
    where the note originated. Current-management facts may only be used when
    both admissions match and the note was available before the decision.
    Durable adverse-reaction/intolerance facts may originate in an earlier
    admission and are used only to increase caution; they never justify
    suppression.
    """

    subject_id: int
    hadm_id: int
    extraction_target: str
    final_confidence: float
    note_time: Optional[pd.Timestamp]
    rule_id: Optional[str]
    drug_pair: Optional[str]
    supporting_quote: Optional[str]
    ingredient_key: Optional[str] = None
    source_hadm_id: Optional[int] = None
    note_id: Optional[str] = None
    temporal_scope: str = "same_admission_predecision"

    @property
    def origin_hadm_id(self) -> int:
        return int(self.source_hadm_id if self.source_hadm_id is not None else self.hadm_id)


class NoteFactStore:
    """Strict, time-causal lookup of accepted note-derived facts.

    A fact can influence a medication decision only when:
      * the model said a signal was found;
      * the supporting quote was verified;
      * final confidence is in (0, 1];
      * the fact has the required rule/drug/ingredient specificity; and
      * note_time is present and no later than the decision timestamp.

    Current intent/monitoring facts must originate in the current admission.
    Prior adverse-reaction/intolerance facts may come from an earlier
    admission, but can only remove suppression or increase review priority.
    Legacy admission-level rows lacking time or specificity are retained for
    coverage reporting and excluded from primary routing.
    """

    def __init__(self, facts: Iterable[NoteFact], legacy_rows: Optional[pd.DataFrame] = None):
        self._facts_by_subject: Dict[int, list[NoteFact]] = {}
        for fact in facts:
            self._facts_by_subject.setdefault(int(fact.subject_id), []).append(fact)
        for values in self._facts_by_subject.values():
            values.sort(key=lambda fact: fact.note_time or pd.Timestamp.max)
        self.legacy_rows = legacy_rows if legacy_rows is not None else pd.DataFrame()

    @staticmethod
    def _bool(series: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False)
        return series.astype("string").str.strip().str.lower().isin(_TRUE)

    @staticmethod
    def _first_column(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
        return next((name for name in names if name in df.columns), None)

    @classmethod
    def from_csv(cls, path: str | Path) -> "NoteFactStore":
        source = Path(path)
        if not source.exists():
            return cls([])
        if source.stat().st_size == 0:
            return cls([])
        try:
            df = pd.read_csv(source, low_memory=False)
        except pd.errors.EmptyDataError:
            return cls([])
        required = {
            "subject_id",
            "extraction_target",
            "quote_verified",
            "final_confidence",
        }
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Note extraction file is missing required columns: {missing}")
        if "hadm_id" not in df.columns and "target_hadm_id" not in df.columns:
            raise ValueError("Note extraction file requires hadm_id or target_hadm_id")
        if df.empty:
            return cls([])

        confidence = pd.to_numeric(df["final_confidence"], errors="coerce")
        accepted = cls._bool(df["quote_verified"]) & confidence.gt(0) & confidence.le(1)
        if "signal_found" in df.columns:
            accepted &= cls._bool(df["signal_found"])
        valid = df.loc[accepted].copy()
        valid["final_confidence"] = confidence.loc[accepted]

        time_column = cls._first_column(valid, ("note_time", "availability_time", "storetime", "charttime"))
        target_hadm_column = "target_hadm_id" if "target_hadm_id" in valid.columns else "hadm_id"
        has_rule_specificity = "rule_id" in valid.columns or "drug_pair" in valid.columns
        has_ingredient_specificity = "ingredient_key" in valid.columns

        def present(value: object) -> bool:
            return value is not None and not pd.isna(value) and bool(str(value).strip())

        legacy_mask = pd.Series(False, index=valid.index)
        if time_column is None:
            legacy_mask[:] = True
        for idx, row in valid.iterrows():
            target = str(row.get("extraction_target", ""))
            if target in CURRENT_INTENT_TARGETS | CURRENT_MONITORING_TARGETS | PRIOR_MANAGEMENT_TARGETS:
                if not has_rule_specificity or (
                    not present(row.get("rule_id")) and not present(row.get("drug_pair"))
                ):
                    legacy_mask.loc[idx] = True
            elif target in DURABLE_ADVERSE_TARGETS:
                if not has_ingredient_specificity or not present(row.get("ingredient_key")):
                    legacy_mask.loc[idx] = True
            else:
                legacy_mask.loc[idx] = True

        legacy = valid.loc[legacy_mask].copy()
        valid = valid.loc[~legacy_mask].copy()
        if valid.empty or time_column is None:
            return cls([], legacy_rows=legacy)

        valid[time_column] = pd.to_datetime(valid[time_column], errors="coerce")
        facts: list[NoteFact] = []
        for row in valid.itertuples(index=False):
            note_time = getattr(row, time_column)
            if pd.isna(note_time):
                continue
            subject = pd.to_numeric(pd.Series([getattr(row, "subject_id")]), errors="coerce").iloc[0]
            target_hadm = pd.to_numeric(
                pd.Series([getattr(row, target_hadm_column)]), errors="coerce"
            ).iloc[0]
            source_hadm_raw = getattr(row, "source_hadm_id", target_hadm)
            source_hadm = pd.to_numeric(pd.Series([source_hadm_raw]), errors="coerce").iloc[0]
            if pd.isna(subject) or pd.isna(target_hadm):
                continue
            if pd.isna(source_hadm):
                source_hadm = target_hadm

            def optional_text(name: str) -> Optional[str]:
                value = getattr(row, name, None)
                if value is None or pd.isna(value) or not str(value).strip():
                    return None
                return str(value).strip()

            facts.append(
                NoteFact(
                    subject_id=int(subject),
                    hadm_id=int(target_hadm),
                    extraction_target=str(getattr(row, "extraction_target")),
                    final_confidence=float(getattr(row, "final_confidence")),
                    note_time=pd.Timestamp(note_time),
                    rule_id=optional_text("rule_id"),
                    drug_pair=optional_text("drug_pair"),
                    supporting_quote=optional_text("supporting_quote"),
                    ingredient_key=optional_text("ingredient_key"),
                    source_hadm_id=int(source_hadm),
                    note_id=optional_text("note_id"),
                    temporal_scope=optional_text("temporal_scope") or (
                        "same_admission_predecision"
                        if int(source_hadm) == int(target_hadm)
                        else "prior_admission_history"
                    ),
                )
            )
        return cls(facts, legacy_rows=legacy)

    @staticmethod
    def _rule_or_pair_match(fact: NoteFact, rule_id: str, drug_pair: Optional[str]) -> bool:
        if fact.rule_id and fact.rule_id != str(rule_id):
            return False
        if fact.drug_pair and (not drug_pair or fact.drug_pair != str(drug_pair)):
            return False
        return bool(fact.rule_id or fact.drug_pair)

    def _current_fact(
        self,
        subject_id: int,
        hadm_id: int,
        decision_time: pd.Timestamp,
        rule_id: str,
        drug_pair: Optional[str],
        targets: set[str],
    ) -> Optional[NoteFact]:
        matches: list[NoteFact] = []
        for fact in self._facts_by_subject.get(int(subject_id), []):
            if fact.extraction_target not in targets:
                continue
            if fact.hadm_id != int(hadm_id) or fact.origin_hadm_id != int(hadm_id):
                continue
            if fact.note_time is None or fact.note_time > pd.Timestamp(decision_time):
                continue
            if not self._rule_or_pair_match(fact, rule_id, drug_pair):
                continue
            matches.append(fact)
        if not matches:
            return None
        return max(matches, key=lambda fact: (fact.final_confidence, fact.note_time))

    def matching_fact(
        self,
        subject_id: int,
        hadm_id: int,
        decision_time: pd.Timestamp,
        rule_id: str,
        drug_pair: Optional[str],
        extraction_target: str = "documented_interaction_awareness",
    ) -> Optional[NoteFact]:
        """Backward-compatible current-admission fact lookup."""
        targets = {extraction_target}
        if extraction_target == "documented_interaction_awareness":
            targets |= CURRENT_INTENT_TARGETS
        return self._current_fact(
            subject_id, hadm_id, decision_time, rule_id, drug_pair, targets
        )

    def current_intent(
        self,
        subject_id: int,
        hadm_id: int,
        decision_time: pd.Timestamp,
        rule_id: str,
        drug_pair: Optional[str],
    ) -> Optional[NoteFact]:
        return self._current_fact(
            subject_id,
            hadm_id,
            decision_time,
            rule_id,
            drug_pair,
            CURRENT_INTENT_TARGETS,
        )

    def current_monitoring_plan(
        self,
        subject_id: int,
        hadm_id: int,
        decision_time: pd.Timestamp,
        rule_id: str,
        drug_pair: Optional[str],
    ) -> Optional[NoteFact]:
        return self._current_fact(
            subject_id,
            hadm_id,
            decision_time,
            rule_id,
            drug_pair,
            CURRENT_MONITORING_TARGETS,
        )

    def prior_management_history(
        self,
        subject_id: int,
        hadm_id: int,
        decision_time: pd.Timestamp,
        rule_id: str,
        drug_pair: Optional[str],
    ) -> Optional[NoteFact]:
        matches: list[NoteFact] = []
        for fact in self._facts_by_subject.get(int(subject_id), []):
            if fact.extraction_target not in PRIOR_MANAGEMENT_TARGETS:
                continue
            if fact.note_time is None or fact.note_time > pd.Timestamp(decision_time):
                continue
            if fact.origin_hadm_id == int(hadm_id):
                # Historical management is deliberately a prior-admission concept.
                continue
            if fact.hadm_id not in {int(hadm_id), 0}:
                continue
            if not self._rule_or_pair_match(fact, rule_id, drug_pair):
                continue
            matches.append(fact)
        if not matches:
            return None
        return max(matches, key=lambda fact: (fact.note_time, fact.final_confidence))

    def prior_adverse_reaction(
        self,
        subject_id: int,
        hadm_id: int,
        decision_time: pd.Timestamp,
        ingredient_keys: Sequence[str],
    ) -> Optional[NoteFact]:
        keys = {str(value).strip() for value in ingredient_keys if str(value).strip()}
        if not keys:
            return None
        matches: list[NoteFact] = []
        for fact in self._facts_by_subject.get(int(subject_id), []):
            if fact.extraction_target not in DURABLE_ADVERSE_TARGETS:
                continue
            if fact.note_time is None or fact.note_time > pd.Timestamp(decision_time):
                continue
            if fact.hadm_id not in {int(hadm_id), 0}:
                # Target-specific extracted facts should not leak into another
                # decision admission. A value of 0 denotes patient-wide scope.
                continue
            if not fact.ingredient_key or fact.ingredient_key not in keys:
                continue
            matches.append(fact)
        if not matches:
            return None
        return max(matches, key=lambda fact: (fact.note_time, fact.final_confidence))

    def context_for_signal(
        self,
        subject_id: int,
        hadm_id: int,
        decision_time: pd.Timestamp,
        rule_id: str,
        drug_pair: Optional[str],
        ingredient_keys: Sequence[str],
    ) -> Dict[str, Optional[NoteFact]]:
        return {
            "current_clinical_intent": self.current_intent(
                subject_id, hadm_id, decision_time, rule_id, drug_pair
            ),
            "current_monitoring_plan": self.current_monitoring_plan(
                subject_id, hadm_id, decision_time, rule_id, drug_pair
            ),
            "prior_management_history": self.prior_management_history(
                subject_id, hadm_id, decision_time, rule_id, drug_pair
            ),
            "prior_adverse_reaction": self.prior_adverse_reaction(
                subject_id, hadm_id, decision_time, ingredient_keys
            ),
        }

    def coverage_summary(self) -> Dict[str, Any]:
        facts = [fact for group in self._facts_by_subject.values() for fact in group]
        by_target: Dict[str, int] = {}
        for fact in facts:
            by_target[fact.extraction_target] = by_target.get(fact.extraction_target, 0) + 1
        return {
            "timestamped_specific_accepted_facts": len(facts),
            "timestamped_specific_patients": len({fact.subject_id for fact in facts}),
            "timestamped_specific_target_admissions": len(
                {(fact.subject_id, fact.hadm_id) for fact in facts}
            ),
            "timestamped_specific_source_admissions": len(
                {(fact.subject_id, fact.origin_hadm_id) for fact in facts}
            ),
            "facts_by_target": by_target,
            "legacy_accepted_rows_excluded_from_primary": int(len(self.legacy_rows)),
            "legacy_patients": (
                int(self.legacy_rows["subject_id"].nunique())
                if not self.legacy_rows.empty and "subject_id" in self.legacy_rows.columns
                else 0
            ),
        }
