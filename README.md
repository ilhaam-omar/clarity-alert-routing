# Routing medication safety concerns under finite capacity

Code and aggregate results for the article:

> Omar IA, Simsekler MCE, Sleptchenko A, Salah K, Omar M. *Routing Medication Safety Concerns Under Finite Capacity: A Pre-Deployment Diagnostic of Capacity, Priority and Information Limits.* Submitted to Reliability Engineering & System Safety, 2026.

The study routes each medication safety concern to one of four responses (suppression, passive notification, interruptive alert or pharmacist review) and applies three tests before deployment:

| Test | Limit it addresses | Question |
|---|---|---|
| Feasibility test | Capacity limit | Can the safety rules be met within the interruption allowance and the review slots? |
| Robustness test | Priority limit | Which review choices hold under every plausible weighting and model setting? |
| Data-adequacy test | Information limit | Can the recorded data separate orders followed by harm from the rest? |

The analysis uses MIMIC-IV version 3.1. This repository contains no patient-level data. It holds the code, the catalogue of screening rules, synthetic test data and the aggregate outputs behind every table and figure in the article.

The software is for research only. It does not give treatment recommendations and does not show clinical benefit.

## Repository contents

| Folder | Contents |
|---|---|
| `clarity_v3/` | Concern detection, patient state, safety rules, response selection, order bundles and capacity allocation |
| `config/` | Fixed analysis settings, capacity scenarios, weight design and a template for local data paths |
| `reference/` | The catalogue of screening rules and the source register for every rule |
| `scripts/` | Pipeline stages 01-10 and the synthetic run |
| `experiments/` | The three tests and the supporting analyses, and `run_all_final.py`, which runs them all |
| `tests/` | Automated software tests on synthetic data |
| `results/final_results_20261103_134909/` | Aggregate outputs of the final run, with no patient-level files, and the run log |
| `figures/` | The data behind Figures 2 and 3 (`Figures_data.xlsx`) and the editable charts |
| `docs/` | Data access and reproduction scope |

Supplementary Material S13 (Table S24) maps each table and figure in the article to its script and output folder.

## Install

The analyses used Python 3.13.11, NumPy 2.3.5 and pandas 2.2.3. Install the pinned packages in `requirements.txt`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows, use `py -3.13 -m venv .venv` and `.venv\Scripts\python.exe`.

## Check the installation on synthetic data

The synthetic data are fictional and are generated from values written in `scripts/make_demo_data.py`. They are not MIMIC-IV records.

```bash
.venv/bin/python scripts/make_demo_data.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/run_synthetic.py
```

## Reproduce the article's results

1. Obtain credentialed access to MIMIC-IV 3.1 through PhysioNet (https://doi.org/10.13026/kpb9-mt58) and accept its data-use agreement.
2. Rebuild the three input files (order timeline, detected concerns and patient states) with scripts 00a-03, as described in `docs/DATA_AND_REPRODUCTION.md`. Keep them in restricted local storage.
3. Run the full analysis with one command, pointing it to the frozen inputs and to two MIMIC-IV laboratory tables:

```bash
.venv/bin/python experiments/run_all_final.py --clean \
    --frozen-input-dir /path/to/restricted/inputs \
    --labevents /path/to/mimic-iv/hosp/labevents.csv \
    --d-labitems /path/to/mimic-iv/hosp/d_labitems.csv
```

The run takes about a day on a laptop and logs its progress to `final_run_log.txt`.

Every script checks the SHA-256 fingerprints of its source files and inputs against `experiments/experiment_source_guard.json` and stops on any mismatch. Patient-level outputs are written only to local `private/` folders, which are never committed. The run writes every aggregate output into one archive, `final_results_<date>.zip`, whose contents should match `results/final_results_20261103_134909/`.

## Data protection

- No raw MIMIC-IV tables, patient identifiers, admission identifiers or per-order decisions are included.
- Counts from 1 to 10 in the released tables are shown as <11.
- `.gitignore` excludes every `private/` folder and local configuration file.

## Citation

Please cite the article and this software release (see `CITATION.cff`).

## Licence

The code is released under the MIT Licence (see `LICENSE`). MIMIC-IV remains subject to its own licence and data-use agreement.
