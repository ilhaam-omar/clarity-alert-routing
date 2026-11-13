"""Set the number of weight combinations per envelope and record the new configuration fingerprint.

The convergence test showed that priority scores converge from 4,096 combinations per envelope.
This script (1) sets n_points in config/uncertainty_design.json for all three envelopes and
(2) records the new file's fingerprint in experiments/experiment_source_guard.json, keeping the
previous fingerprint, so that the guarded pipeline accepts the new primary setting. A note of the
change is written to experiments/design_points_change.json.

Usage
  python experiments/set_design_points.py --project . --points 4096
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--points", type=int, default=4096)
    a = ap.parse_args()
    root = a.project.expanduser().resolve()
    cfg_path = root / "config" / "uncertainty_design.json"
    guard_path = root / "experiments" / "experiment_source_guard.json"
    raw_before = cfg_path.read_bytes()
    cfg = json.loads(raw_before)
    before = sorted({e["n_points"] for e in cfg["envelopes"]})
    for e in cfg["envelopes"]:
        e["n_points"] = int(a.points)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    data = cfg_path.read_bytes()
    new_hash = hashlib.sha256(data).hexdigest()
    norm_hash = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    entry = guard["source_files"]["config/uncertainty_design.json"]
    entry.setdefault("previous_sha256", entry.get("sha256"))
    entry["sha256"] = new_hash
    entry["newline_normalized_sha256"] = norm_hash
    entry["publication_sha256"] = new_hash
    guard_path.write_text(json.dumps(guard, indent=2) + "\n", encoding="utf-8")
    note = {"changed": datetime.datetime.now().isoformat(timespec="seconds"), "n_points_before": before,
            "n_points_after": a.points, "reason": "Priority scores converge from 4,096 combinations per envelope "
            "(convergence_test.py); 1,024 left seed-dependent sampling noise in the review ranking.",
            "previous_config_sha256": hashlib.sha256(raw_before).hexdigest(), "new_config_sha256": new_hash}
    (root / "experiments" / "design_points_change.json").write_text(json.dumps(note, indent=2) + "\n")
    print(json.dumps(note, indent=2))


if __name__ == "__main__":
    main()
