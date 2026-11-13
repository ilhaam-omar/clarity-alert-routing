from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def build_manifest(
    input_files: Iterable[str | Path],
    config_files: Iterable[str | Path],
    output_files: Iterable[str | Path],
    source_files: Iterable[str | Path] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def describe(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
        rows = []
        for value in paths:
            path = Path(value).resolve()
            rows.append(
                {
                    "path": str(path),
                    "exists": path.exists(),
                    "bytes": path.stat().st_size if path.exists() else None,
                    "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
                }
            )
        return rows

    payload: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "inputs": describe(input_files),
        "configs": describe(config_files),
        "outputs": describe(output_files),
        "source_files": describe(source_files or []),
    }
    if extra:
        payload["run_metadata"] = extra
    return payload


def assert_no_identifiers_in_evidence(path: str | Path) -> None:
    source = Path(path)
    if source.suffix.lower() != ".csv" or not source.exists():
        return
    columns = set(pd.read_csv(source, nrows=0).columns)
    forbidden = {"subject_id", "hadm_id", "stay_id", "note_id", "order_id", "signal_id"}
    overlap = sorted(columns & forbidden)
    if overlap:
        raise ValueError(
            f"Evidence file {source.name} contains patient/event identifiers: {overlap}"
        )


def ensure_no_identifier_columns(frame: pd.DataFrame, source_name: str = "evidence") -> None:
    """Reject patient/event identifiers from an aggregate evidence table."""
    forbidden = {
        "subject_id", "hadm_id", "stay_id", "note_id", "order_id", "signal_id",
        "bundle_id", "patient_id", "mrn",
    }
    overlap = sorted(set(frame.columns) & forbidden)
    if overlap:
        raise ValueError(
            f"Evidence table {source_name} contains patient/event identifiers: {overlap}"
        )
