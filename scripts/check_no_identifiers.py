"""Check files staged for release for patient-level identifiers and small counts.

Run from the repository root before every push:

    python scripts/check_no_identifiers.py

It lists every tracked or staged CSV, TSV, JSON or XLSX file whose column names
include a MIMIC-IV identifier, every file under a folder named "private", and
every text file that contains a local Windows user path (which reveals names).
It exits with status 1 if it finds any, so it can also run in CI.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

IDENTIFIER_COLUMNS = {
    "subject_id", "hadm_id", "stay_id", "poe_id", "poe_key", "order_id",
    "bundle_id", "signal_id", "pharmacy_id", "emar_id", "transplant_episode_id",
    "source_rx_row_index", "mrn", "patient_id",
}


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [Path(p) for p in out.splitlines() if p]


def header_columns(path: Path) -> set[str]:
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".tsv"}:
            with path.open(newline="", encoding="utf-8", errors="replace") as f:
                row = next(csv.reader(f, delimiter="\t" if suffix == ".tsv" else ","), [])
            return {c.strip().lower() for c in row}
        if suffix == ".json":
            text = path.read_text(encoding="utf-8", errors="replace")
            data = json.loads(text)
            keys: set[str] = set()

            def walk(obj: object) -> None:
                if isinstance(obj, dict):
                    keys.update(str(k).lower() for k in obj)
                    for v in obj.values():
                        walk(v)
                elif isinstance(obj, list):
                    for v in obj[:50]:
                        walk(v)

            walk(data)
            return keys
        if suffix == ".xlsx":
            from openpyxl import load_workbook  # optional dependency

            wb = load_workbook(path, read_only=True)
            cols: set[str] = set()
            for ws in wb.worksheets:
                for row in ws.iter_rows(max_row=3, values_only=True):
                    cols.update(str(c).strip().lower() for c in row if c is not None)
            return cols
    except Exception as exc:  # report unreadable files rather than skip them
        print(f"UNREADABLE  {path}: {exc}")
    return set()


LOCAL_PATH = re.compile(r"[A-Za-z]:[\\/]+(Users|Documents and Settings)[\\/]", re.I)
TEXT_SUFFIXES = {".csv", ".tsv", ".json", ".md", ".txt", ".py", ".toml", ".cff", ".ipynb", ".yml", ".yaml"}


def has_local_path(path: Path) -> bool:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return False
    try:
        return bool(LOCAL_PATH.search(path.read_text(encoding="utf-8", errors="replace")))
    except Exception:
        return False


def main() -> int:
    problems = 0
    for path in tracked_files():
        if "private" in path.parts:
            print(f"PRIVATE FOLDER  {path}")
            problems += 1
            continue
        if path.name != "check_no_identifiers.py" and has_local_path(path):
            print(f"LOCAL PATH  {path}")
            problems += 1
        hits = header_columns(path) & IDENTIFIER_COLUMNS
        if hits:
            print(f"IDENTIFIER COLUMNS {sorted(hits)}  {path}")
            problems += 1
    if problems:
        print(f"\n{problems} file(s) need review before release.")
        return 1
    print("No identifier columns, private folders or local paths found in tracked files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
