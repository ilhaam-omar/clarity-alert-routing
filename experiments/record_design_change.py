"""Record the final weight-design change in the repository's version check.

The final analysis uses the grounded (judgement-defined) weight design with 4,096 weightings
per viewpoint. This changes clarity_v3/uncertainty.py and config/uncertainty_design.json, which
the version check in experiments/experiment_source_guard.json fingerprints. This script records
the new fingerprints (keeping the previous ones) and writes experiments/design_change_note.json,
so that every guarded script accepts the final setting and the change is documented.

Usage
  python experiments/record_design_change.py --project .
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

REASON = ("Final analysis: grounded weight design (weightings limited by definitional domains, structural "
          "orders and four reference-case judgements, so that each response is the intended response for at least one reference concern; no declared ranges) with 4,096 weightings per viewpoint, "
          "after the convergence test showed 1,024 left sampling noise in the review ranking. The earlier "
          "declared-range design is kept as config/uncertainty_design_declared_ranges.json and tested as an "
          "alternative.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project", type=Path, required=True)
    a = ap.parse_args()
    root = a.project.expanduser().resolve()
    guard_path = root / "experiments" / "experiment_source_guard.json"
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    changed = {}
    for name, entry in guard["source_files"].items():
        data = (root / name).read_bytes()
        raw = hashlib.sha256(data).hexdigest()
        norm = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
        if raw in {entry.get("sha256"), entry.get("publication_sha256")} or norm == entry.get("newline_normalized_sha256"):
            continue
        entry.setdefault("previous_sha256", entry.get("sha256"))
        entry["sha256"], entry["newline_normalized_sha256"], entry["publication_sha256"] = raw, norm, raw
        changed[name] = raw
    guard_path.write_text(json.dumps(guard, indent=2) + "\n", encoding="utf-8")
    cfg = json.loads((root / "config" / "uncertainty_design.json").read_text(encoding="utf-8"))
    note = {"recorded": datetime.datetime.now().isoformat(timespec="seconds"), "reason": REASON,
            "weightings_per_viewpoint": sorted({e["n_points"] for e in cfg["envelopes"]}),
            "grounded": all(bool(e.get("anchor_rules")) for e in cfg["envelopes"]),
            "fingerprints_updated": changed}
    (root / "experiments" / "design_change_note.json").write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(note, indent=2))


if __name__ == "__main__":
    main()
