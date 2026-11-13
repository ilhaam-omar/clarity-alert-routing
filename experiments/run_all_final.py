"""Run the complete final analysis in order and collect the aggregate results into one ZIP.

Steps
   1  record the final weight design in the version check
   2  rebuild routing from the frozen Stage 01-03 inputs (Stages 04-07, 09, 10, 10A, 10B, matched comparison)
   3  allocation robustness, parameter sensitivity, robustness classification
   4  laboratory harm markers
   5  comparator harm alignment and suppression check
   6  weight designs, history check, predictive check, feasibility frontier, cohort profile
   7  convergence confirmation (4,096 and 16,384 weightings)
   8  final_results_<date>.zip with every aggregate output and no patient-level file

With --clean, results from earlier runs in this folder (outputs_corrected, the dated result
folders and earlier final_results ZIPs) are deleted first, so only one set of results remains.
The frozen Stage 01-03 inputs must be outside outputs_corrected (for example outputs_corrected_4096).
Progress is written to final_run_log.txt; if a step fails, the run stops and says which.

Usage
  python experiments/run_all_final.py --clean --frozen-input-dir "outputs_corrected_4096\\private" ^
      --labevents "...\\labevents.csv" --d-labitems "...\\d_labitems.csv"
"""
from __future__ import annotations

import argparse
import datetime
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "final_run_log.txt"


def log(msg: str) -> None:
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(label: str, args: list[str]) -> None:
    log(f"START {label}")
    t0 = time.time()
    r = subprocess.run([sys.executable, *args], cwd=ROOT)
    if r.returncode != 0:
        log(f"FAILED {label} (exit code {r.returncode}). Fix the problem and rerun this script in a fresh copy.")
        raise SystemExit(1)
    log(f"DONE  {label} ({(time.time() - t0) / 60:.1f} min)")


def newest(prefix: str) -> Path:
    found = sorted(p for p in ROOT.glob(prefix + "*") if p.is_dir())
    if not found:
        log(f"FAILED: no output folder starting with {prefix}")
        raise SystemExit(1)
    return found[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frozen-input-dir", type=Path, required=True)
    ap.add_argument("--labevents", type=Path, required=True)
    ap.add_argument("--d-labitems", type=Path, required=True)
    ap.add_argument("--clean", "--overwrite", dest="clean", action="store_true",
                    help="Delete results of earlier runs in this folder first, so only one set remains")
    a = ap.parse_args()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    frozen = Path(a.frozen_input_dir).resolve()
    current = (ROOT / "outputs_corrected").resolve()
    if frozen == current or current in frozen.parents:
        print("STOPPED: the frozen inputs must not be inside outputs_corrected. Use outputs_corrected_4096\\private.")
        raise SystemExit(1)
    if not (frozen / "03_longitudinal_states.csv").exists():
        print(f"STOPPED: frozen inputs not found in {frozen} (expected the 01-03 CSV files).")
        raise SystemExit(1)
    if a.clean:
        prefixes = ("matched_comparison_", "allocation_robustness_", "parameter_sensitivity_", "robust_tier_",
                    "lab_harm_markers_", "feasibility_frontier_", "cohort_profile_", "predictive_check_",
                    "weight_grounding_", "history_check_", "convergence_test_", "information_ceiling_",
                    "comparator_harm_alignment", "suppression_condition_overlay", "final_results_")
        removed = 0
        for p in list(ROOT.iterdir()):
            if p.name == "outputs_corrected" or p.name.startswith(prefixes):
                shutil.rmtree(p) if p.is_dir() else p.unlink()
                removed += 1
        if LOG.exists():
            LOG.unlink()
        print(f"Cleaned {removed} result folders and files from earlier runs.")
    if (ROOT / "outputs_corrected").exists():
        print("STOPPED: outputs_corrected already exists. Rename it (for example to outputs_corrected_previous) "
              "or use a fresh copy of the repository, then rerun.")
        raise SystemExit(1)
    for path in (a.frozen_input_dir, a.labevents, a.d_labitems):
        if not Path(path).exists():
            print(f"STOPPED: not found: {path}")
            raise SystemExit(1)
    # fixed-name output folders from earlier runs are set aside so results are never mixed
    for fixed in ("comparator_harm_alignment", "suppression_condition_overlay"):
        if (ROOT / fixed).exists():
            (ROOT / fixed).rename(ROOT / f"{fixed}_previous_{stamp}")
    log("FINAL ANALYSIS RUN " + stamp)

    run("1 record final weight design", ["experiments/record_design_change.py", "--project", "."])
    run("2a check frozen inputs", ["scripts/reproduce_corrected.py", "--frozen-input-dir", str(a.frozen_input_dir), "--dry-run"])
    run("2b rebuild routing", ["scripts/reproduce_corrected.py", "--frozen-input-dir", str(a.frozen_input_dir), "--extended"])
    run("3a allocation robustness", ["experiments/allocation_robustness.py", "--project", "."])
    run("3b parameter sensitivity", ["experiments/parameter_sensitivity_allocation.py", "--project", "."])
    run("3c robustness classification", ["experiments/robust_tier.py", "--project", "."])
    scores = newest("robust_tier_") / "private" / "specification_scores.csv"
    run("4 laboratory harm markers", ["experiments/lab_harm_markers.py", "--project", ".", "--labevents", str(a.labevents),
                                      "--d-labitems", str(a.d_labitems), "--scores", str(scores)])
    markers = newest("lab_harm_markers_") / "private" / "bundle_markers.csv"
    run("5a comparator harm alignment", ["experiments/comparator_harm_alignment.py", "--project", ".", "--markers", str(markers)])
    run("5b suppression check", ["experiments/suppression_condition_overlay.py", "--project", ".", "--markers", str(markers)])
    run("6a feasibility frontier", ["experiments/feasibility_frontier.py", "--project", ".", "--outputs", "outputs_corrected"])
    run("6b cohort profile", ["experiments/cohort_profile.py", "--project", ".", "--outputs", "outputs_corrected"])
    run("6c predictive check", ["experiments/predictive_check.py", "--project", ".", "--outputs", "outputs_corrected", "--markers", str(markers)])
    run("6d weight designs", ["experiments/weight_grounding.py", "--project", ".", "--outputs", "outputs_corrected", "--scores", str(scores)])
    run("6e history check", ["experiments/history_check.py", "--project", ".", "--outputs", "outputs_corrected", "--markers", str(markers)])
    run("7 convergence confirmation", ["experiments/convergence_test.py", "--project", ".", "--outputs", "outputs_corrected", "--points", "4096,16384"])

    # 8 collect aggregate outputs only
    prefixes = ("matched_comparison_", "allocation_robustness_", "parameter_sensitivity_", "robust_tier_",
                "lab_harm_markers_", "feasibility_frontier_", "cohort_profile_", "predictive_check_",
                "weight_grounding_", "history_check_", "convergence_test_")
    folders = [newest(p) for p in prefixes]
    folders += [ROOT / "comparator_harm_alignment", ROOT / "suppression_condition_overlay"]
    zpath = ROOT / f"final_results_{stamp}.zip"
    written = set()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        def add(f: Path):
            rel = f.relative_to(ROOT).as_posix()
            if rel not in written:
                z.write(f, rel); written.add(rel)
        for folder in folders:
            for f in folder.rglob("*"):
                rel = f.relative_to(ROOT)
                if f.is_file() and "private" not in rel.parts and f.suffix.lower() in {".csv", ".json", ".png", ".pdf", ".txt"}:
                    add(f)
        evidence = ROOT / "outputs_corrected" / "evidence"
        for f in evidence.rglob("*"):
            if f.is_file() and f.suffix.lower() in {".csv", ".json"}:
                add(f)
        for extra in ("outputs_corrected/reproduction_manifest.json", "config/uncertainty_design.json",
                      "config/uncertainty_design_declared_ranges.json", "experiments/design_change_note.json",
                      "final_run_log.txt"):
            if (ROOT / extra).exists():
                add(ROOT / extra)
    added = len(written)
    log(f"COMPLETE. {added} aggregate files collected in {zpath.name} (no private folders included).")
    log("Send this one ZIP file. Keep every private folder on your own computer.")


if __name__ == "__main__":
    main()
