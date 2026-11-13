from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parser(description: str) -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=description)
    value.add_argument(
        "--paths",
        default=str(ROOT / "config" / "synthetic.json"),
        help=(
            "Path to the structured Phase-1 configuration JSON. "
            "Copy config/mimic_paths.example.json and set local paths."
        ),
    )
    return value


def banner(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def progress_printer(label: str, width: int = 32) -> Callable[[int, int, int], None]:
    started = time.perf_counter()

    def callback(done: int, total: int, produced: int) -> None:
        fraction = 1.0 if total == 0 else min(1.0, done / total)
        filled = int(round(width * fraction))
        elapsed = max(1e-9, time.perf_counter() - started)
        rate = done / elapsed
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"\r{label}: [{bar}] {fraction:6.1%} | {done:,}/{total:,} | "
            f"output={produced:,} | {rate:,.1f}/s",
            end="\n" if done >= total else "",
            flush=True,
        )

    return callback